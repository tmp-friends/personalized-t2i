#!/usr/bin/env python
"""Learn a preference embedding for a *new* user with the official Premier code on a 24 GB GPU
(stage 2 of the paper: adapters frozen, only the user representation is trained).

  --mode linear : e_new = Σ_k α_k e_k over the 1000 released training-user embeddings,
                  only α (1000 numbers) is trained  -> paper default, robust for few images
  --mode direct : a fresh 30 x 1024 embedding is trained from scratch (paper ablation)

The training step is a line-by-line port of `OminiModelUserEmbedding.training_step` in
scripts/train_flux/train_user_embedding(_linear).py (same t sampling, loss, guidance,
optimiser defaults), minus Lightning and with the memory handling of premier_repro.official.

Input images (preferred images of the user + their prompts), one of:
  --csv data.csv [--data-path root]       columns positive_image, caption   (official format)
  --json items.json                       [{"image": ..., "caption": ...}, ...]
  --images a.png b.png --captions "..." "..."

Example
  python scripts/train_official_user.py --name alice --json my_images.json --mode linear --steps 1000 --out outputs/official/users
  python scripts/run_official.py --users none file:outputs/official/users/alice/user_combination_alice.safetensors --prompts "a cat" --out outputs/official/alice_gen
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import save_file

from premier_repro.official import (N_TRAIN_USERS, TOKEN_NUM, USER_DIM, WEIGHTS,
                                    EmbeddingLinearCombination, cuda_gb, encode_images,
                                    load_adapter_config, load_components, load_train_bank,
                                    log, make_pipeline, quantize_int8,
                                    require_patched_upstream)

require_patched_upstream()

from scripts.pipeline.flux_adapter import transformer_forward_verse  # official
from scripts.pipeline.mod_adapters import load_modulation_adapter  # official


def read_items(a) -> list[tuple[Path, str]]:
    items = []
    if a.csv:
        root = Path(a.data_path or Path(a.csv).parent)
        for row in csv.DictReader(open(a.csv)):
            items.append((root / row["positive_image"], row["caption"]))
    elif a.json:
        root = Path(a.data_path or Path(a.json).parent)
        for r in json.load(open(a.json)):
            p = Path(r["image"])
            items.append((p if p.is_absolute() else root / p, r["caption"]))
    else:
        assert a.images and a.captions and len(a.images) == len(a.captions), "--images and --captions must pair up"
        items = [(Path(i), c) for i, c in zip(a.images, a.captions)]
    for p, _ in items:
        assert p.exists(), f"missing image {p}"
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--csv"); ap.add_argument("--json"); ap.add_argument("--data-path")
    ap.add_argument("--images", nargs="*"); ap.add_argument("--captions", nargs="*")
    ap.add_argument("--mode", default="linear", choices=["linear", "direct"])
    ap.add_argument("--steps", type=int, default=1000, help="official config: 5000 (~2 h on a 4090)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--optimizer", default=None, help="AdamW | Prodigy (official: AdamW for linear, Prodigy for direct)")
    ap.add_argument("--lr", type=float, default=None, help="official: 0.01 (AdamW, linear) / 1.0 (Prodigy, direct)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--memory", default="int8", choices=["int8", "bf16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--out", default=str(ROOT / "outputs/official/users"))
    a = ap.parse_args()
    weights = Path(a.weights)
    dev, dtype = a.device, torch.bfloat16
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    items = read_items(a)
    log(f"{len(items)} preferred images for user '{a.name}' (mode={a.mode})")

    adapter_config = load_adapter_config(weights)
    is_uncond = adapter_config["model"]["modulation"].get("uncond", False)

    # ---- text embeddings once, then drop T5 / CLIP (9.7 GB)
    c = load_components(dtype=dtype, with_transformer=False)
    pipe = make_pipeline(c).to(dev)
    with torch.no_grad():
        text_cache = {}
        for _, cap in items:
            if cap not in text_cache:
                pe, ppe, tid = pipe.encode_prompt(prompt=cap, prompt_2=None, device=dev, num_images_per_prompt=1,
                                                  max_sequence_length=a.max_seq_len)
                text_cache[cap] = (pe, ppe, tid)
        uncond_embeds, _, _ = pipe.encode_prompt(prompt="", prompt_2="", device=dev, num_images_per_prompt=1,
                                                 max_sequence_length=a.max_seq_len)
    vae = c["vae"]
    del pipe, c["text_encoder"], c["text_encoder_2"]
    torch.cuda.empty_cache()

    # ---- frozen transformer (int8) + frozen adapter
    c2 = load_components(dtype=dtype, with_text_encoders=False, with_vae=False)
    transformer = c2["transformer"]
    if a.memory == "int8":
        quantize_int8(transformer, dev)
    else:
        transformer.to(dev)
    transformer.requires_grad_(False)
    transformer.gradient_checkpointing = not a.no_grad_ckpt
    transformer.train()   # the official forward checks `transformer.training and transformer.gradient_checkpointing`
    pipe = make_pipeline({"scheduler": c2["scheduler"], "vae": vae, "transformer": transformer})
    mod_adapter = load_modulation_adapter(adapter_config, dtype, dev, ckpt_dir=str(weights), is_training=False)
    mod_adapter.eval()
    log(f"models ready ({a.memory}, grad ckpt={not a.no_grad_ckpt}); cuda {cuda_gb()}")

    # ---- trainable user representation (official initialisation)
    if a.mode == "linear":
        bank = load_train_bank(dev, dtype, weights)
        comb = EmbeddingLinearCombination(embedding_num=N_TRAIN_USERS, combination_size=1, use_softmax=False).to(dev, dtype)
        params = list(comb.parameters())
        opt_name, lr = a.optimizer or "AdamW", a.lr if a.lr is not None else 0.01
        get_user = lambda b: comb(bank, input_ids=torch.zeros(b, dtype=torch.long, device=dev)).view(-1, TOKEN_NUM, USER_DIM)
    else:
        user_emb = nn.Embedding(1, TOKEN_NUM * USER_DIM).to(dev, dtype)
        user_emb.weight.data.uniform_(-0.001, 0.001)
        params = list(user_emb.parameters())
        opt_name, lr = a.optimizer or "Prodigy", a.lr if a.lr is not None else 1.0
        get_user = lambda b: user_emb(torch.zeros(b, dtype=torch.long, device=dev)).view(-1, TOKEN_NUM, USER_DIM)
    for p in params:
        p.requires_grad_(True)
    if opt_name == "AdamW":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.weight_decay)
    elif opt_name == "Prodigy":
        import prodigyopt
        opt = prodigyopt.Prodigy(params, lr=lr, weight_decay=a.weight_decay, use_bias_correction=True, safeguard_warmup=True)
    else:
        raise ValueError(opt_name)
    log(f"optimizer {opt_name} lr={lr}, {sum(p.numel() for p in params)} trainable values")

    to_tensor = T.ToTensor()
    pil_cache = {p: to_tensor(Image.open(p).convert("RGB").resize((a.size, a.size))) for p, _ in items}
    logf = open(out / "log.jsonl", "a")
    ema = None
    t0 = time.time()
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(a.accum):
            batch = random.sample(items, k=min(a.batch_size, len(items)))
            imgs = torch.stack([pil_cache[p] for p, _ in batch])
            caps = [cap for _, cap in batch]
            b = imgs.shape[0]
            with torch.no_grad():
                x_0, img_ids = encode_images(pipe, imgs)                                     # official
                prompt_embeds = torch.cat([text_cache[cp][0] for cp in caps])
                pooled = torch.cat([text_cache[cp][1] for cp in caps])
                text_ids = text_cache[caps[0]][2]
                t = torch.sigmoid(torch.randn((b,), device=dev) + 1.0)                     # official t sampling
                x_1 = torch.randn_like(x_0)
                t_ = t.unsqueeze(1).unsqueeze(1)
                x_t = ((1 - t_) * x_0 + t_ * x_1).to(dtype)
                guidance = torch.ones_like(t) if transformer.config.guidance_embeds else None
            user_pref = get_user(b)
            if is_uncond:
                delta_emb, delta_emb_pblock = mod_adapter(t, prompt_embeds, user_pref, uncond_embeds.repeat(b, 1, 1))
            else:
                delta_emb, delta_emb_pblock = mod_adapter(t, prompt_embeds, user_pref)
            pred = transformer_forward_verse(transformer, model_config=adapter_config, hidden_states=x_t,
                                             encoder_hidden_states=prompt_embeds, pooled_projections=pooled,
                                             img_ids=img_ids, txt_ids=text_ids, guidance=guidance, timestep=t,
                                             return_dict=False, delta_emb=delta_emb, delta_emb_pblock=delta_emb_pblock)[0]
            loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
            (loss / a.accum).backward()
            loss_acc += loss.item() / a.accum
        if a.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        opt.step()
        ema = loss_acc if ema is None else ema * 0.95 + loss_acc * 0.05
        if step % a.log_every == 0 or step == 1:
            rec = {"step": step, "loss": round(loss_acc, 5), "loss_ema": round(ema, 5), "t": round(t.mean().item(), 3),
                   "sec_per_step": round((time.time() - t0) / step, 2)}
            if a.mode == "linear":
                w = comb.get_combination_weights().detach().float()[0]
                top = torch.topk(w.abs(), 3)
                rec["top_coef"] = [(int(i), round(float(w[i]), 4)) for i in top.indices]
            if opt_name == "Prodigy":
                rec["lr_eff"] = float(opt.param_groups[0]["d"] * opt.param_groups[0]["lr"])
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save_user(out, a, comb if a.mode == "linear" else user_emb, get_user, items, step, ema, opt_name, lr)
    log(f"done in {(time.time() - t0) / 60:.1f} min; cuda {cuda_gb()}; saved to {out}")


def save_user(out: Path, a, module, get_user, items, step, ema, opt_name, lr):
    with torch.no_grad():
        emb = get_user(1).reshape(1, TOKEN_NUM * USER_DIM).float().cpu()
    if a.mode == "linear":
        save_file(module.state_dict(), str(out / f"user_combination_{a.name}.safetensors"))   # official format
    save_file({"weight": emb.contiguous()}, str(out / f"user_embedding_{a.name}.safetensors"))  # official (direct) format
    meta = {"name": a.name, "mode": a.mode, "steps": step, "loss_ema": ema, "optimizer": opt_name, "lr": lr,
            "images": [str(p) for p, _ in items], "captions": [c for _, c in items], "size": a.size}
    if a.mode == "linear":
        w = module.get_combination_weights().detach().float()[0].cpu()
        top = torch.topk(w.abs(), 10)
        meta["top_train_users"] = [(int(i), round(float(w[i]), 4)) for i in top.indices]
    json.dump(meta, open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
