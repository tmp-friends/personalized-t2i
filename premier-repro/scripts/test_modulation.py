"""Sanity check on the GPU:
 1. premier_transformer_forward(Δ=None) == stock FluxTransformer2DModel.forward
 2. forward + backward with Δ through the float8-quantised transformer fits in 24 GB
    at 512 px, and gradients reach the adapters and the user embeddings.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from premier_repro.model.loading import (load_transformer, pack_latents, latent_image_ids, text_ids, count_params)
from premier_repro.model.flux_modulation import premier_transformer_forward
from premier_repro.model.premier import PremierModel, PremierConfig
from premier_repro.model.user_embedding import UserEmbeddingBank
from premier_repro.model.losses import flow_matching_loss, dispersion_loss, sample_timesteps

dev = "cuda"
quant = sys.argv[1] if len(sys.argv) > 1 else "quanto_fp8"
res = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B, L = 2, 256
t0 = time.time()
tr = load_transformer(quant=quant, device=dev, gradient_checkpointing=True)
print(f"transformer loaded ({quant}) in {time.time()-t0:.0f}s; cuda mem {torch.cuda.memory_allocated()/2**30:.1f} GB")

h = w = res // 8
lat = pack_latents(torch.randn(B, 16, h, w, device=dev)).to(torch.bfloat16)
t5 = torch.randn(B, L, 4096, device=dev, dtype=torch.bfloat16)
pooled = torch.randn(B, 768, device=dev, dtype=torch.bfloat16)
img_ids = latent_image_ids(h, w, dev, torch.bfloat16); txt_ids = text_ids(L, dev, torch.bfloat16)
t = torch.rand(B, device=dev); guid = torch.ones(B, device=dev)

# 1. equivalence
tr.gradient_checkpointing = False
with torch.no_grad():
    ref = tr(hidden_states=lat, encoder_hidden_states=t5, pooled_projections=pooled, timestep=t, img_ids=img_ids,
             txt_ids=txt_ids, guidance=guid, return_dict=False)[0]
    out = premier_transformer_forward(tr, lat, t5, pooled, t, img_ids, txt_ids, guid)
    zero = torch.zeros(B, L, 3072, device=dev)
    out0 = premier_transformer_forward(tr, lat, t5, pooled, t, img_ids, txt_ids, guid, delta_shared=zero)
print("max|ref-out| (Δ=None):", (ref - out).abs().max().item(), " (Δ=0):", (ref - out0).abs().max().item(),
      " |ref| mean:", ref.abs().mean().item())
tr.gradient_checkpointing = True

# 2. training step
cfg = PremierConfig()
model = PremierModel(cfg).to(dev)
bank = UserEmbeddingBank([f"u{i}" for i in range(8)]).to(dev)
print(f"adapter params: shared {count_params(model.adapter_shared)/1e6:.1f}M, distinct {count_params(model.adapter_distinct)/1e6:.1f}M, groups {model.n_groups}")
params = list(model.parameters()) + list(bank.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)
torch.cuda.reset_peak_memory_stats()
for step in range(3):
    t0 = time.time()
    uidx = torch.tensor([0, 1], device=dev)
    d = model.compute_deltas(t5, bank(uidx))
    noise = torch.randn_like(lat); x0 = torch.randn_like(lat)
    tt = sample_timesteps(B, lat.shape[1], dev)
    zt = ((1 - tt)[:, None, None] * x0.float() + tt[:, None, None] * noise.float()).to(torch.bfloat16)
    pred = premier_transformer_forward(tr, zt, t5, pooled, tt, img_ids, txt_ids, guid, delta_shared=d["shared"],
                                       delta_distinct=d["distinct"], block_group=model.block_group)
    loss_flow = flow_matching_loss(pred, noise, x0)
    dd = model.compute_deltas(t5[:1, :16].expand(8, -1, -1), bank(torch.arange(8, device=dev)))
    loss_disp = dispersion_loss(dd["shared"]) + dispersion_loss(dd["distinct"])
    loss = loss_flow + 0.1 * loss_disp
    loss.backward()
    gn_a = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5
    gn_u = bank.emb.grad.norm().item()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"step {step}: loss_flow {loss_flow.item():.4f} loss_disp {loss_disp.item():.4f} |g_adapter| {gn_a:.3e} |g_user| {gn_u:.3e} "
          f"time {time.time()-t0:.2f}s peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
print("OK")
