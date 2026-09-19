"""Stage 2: preference learning for a *new* user from k preferred images.

  mode = "linear": e_new = Σ_k α_k e_k  (α trained, adapters + bank frozen)   — paper default
  mode = "direct": e_new trained from scratch                                  — ablation
Only the flow-matching loss is used.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from ..config import Cfg
from ..data.cache import FeatureCache
from ..model.flux_modulation import premier_transformer_forward
from ..model.loading import latent_image_ids, pack_latents, text_ids
from ..model.losses import flow_matching_loss, sample_timesteps
from ..model.user_embedding import DirectUser, LinearCombinationUser
from .common import JsonlLogger, build_optimizer, prodigy_effective_lr


def train_new_user(cfg: Cfg, uid: str, manifest: dict, cache: FeatureCache, premier, bank, transformer,
                   out_dir: Path, k_shot: int | None = None, mode: str | None = None, steps: int | None = None,
                   seed: int = 0) -> Path:
    dev = cfg.device
    k_shot = k_shot or cfg.stage2.k_shot
    mode = mode or cfg.stage2.mode
    steps = steps or cfg.stage2.steps
    torch.manual_seed(seed)
    rng = random.Random(seed)

    pairs = [p for p in manifest["users"][uid]["pos"][:k_shot] if cache.has_latent(p["image"]) and cache.has_text(p["prompt"])]
    if not pairs:
        raise RuntimeError(f"user {uid}: no cached preference samples")
    user_out = out_dir / "users" / uid
    user_out.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(user_out / "log.jsonl")

    pcfg = premier.cfg
    if mode == "linear":
        user = LinearCombinationUser(bank.emb.detach(), init=cfg.stage2.coef_init).to(dev)
    elif mode == "direct":
        user = DirectUser(pcfg.user_tokens, pcfg.user_dim, cfg.premier.user_init_std).to(dev)
    elif mode == "mean":  # no training: mean of the bank (sanity baseline)
        user = LinearCombinationUser(bank.emb.detach(), init="uniform").to(dev)
        steps = 0
    else:
        raise ValueError(mode)
    premier.eval()
    premier.requires_grad_(False)
    opt = build_optimizer(user.parameters(), cfg.stage2.optimizer, cfg.stage2.lr, cfg.stage2.weight_decay) if steps else None

    res = cfg.data.resolution
    h = w = res // 8
    img_ids = latent_image_ids(h, w, dev, torch.bfloat16)
    txt_ids = text_ids(cfg.data.max_len, dev, torch.bfloat16)
    image_seq_len = (h // 2) * (w // 2)
    empty_t5 = cache.load_empty_text()[0][None].to(dev)
    bs = cfg.stage2.batch_size
    t0 = time.time()
    losses = []
    for step in tqdm(range(1, steps + 1), desc=f"user {uid} ({mode}, k={len(pairs)})", leave=False, dynamic_ncols=True):
        sel = [rng.choice(pairs) for _ in range(bs)]
        lat = pack_latents(torch.stack([cache.load_latent(p["image"]) for p in sel]).to(dev)).to(torch.bfloat16)
        t5 = torch.stack([cache.load_text(p["prompt"])[0] for p in sel]).to(dev)
        pooled = torch.stack([cache.load_text(p["prompt"])[1] for p in sel]).to(dev)
        noise = torch.randn_like(lat)
        t = sample_timesteps(bs, image_seq_len, dev, cfg.train.timestep_scheme, shift=cfg.train.timestep_shift)
        zt = ((1 - t)[:, None, None] * lat.float() + t[:, None, None] * noise.float()).to(torch.bfloat16)
        guid = torch.full((bs,), float(cfg.train.guidance), device=dev)
        ue = user().expand(bs, -1, -1)
        d = premier.compute_deltas(t5, ue, empty_text_emb=empty_t5)
        pred = premier_transformer_forward(transformer, zt, t5, pooled, t, img_ids, txt_ids, guid,
                                           delta_shared=d["shared"], delta_distinct=d["distinct"],
                                           block_group=premier.block_group)
        loss = flow_matching_loss(pred, noise, lat)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % cfg.stage2.log_every == 0 or step == 1:
            rec = dict(step=step, flow=round(loss.item(), 5), lr_eff=prodigy_effective_lr(opt))
            if mode == "linear":
                c = user.coef.detach()
                rec.update(coef_abs_max=round(c.abs().max().item(), 4), coef_sum=round(c.sum().item(), 4),
                           coef_entropy=round(float(-(c.softmax(0) * c.log_softmax(0)).sum()), 3))
            logger.log(**rec)

    with torch.no_grad():
        emb = user().detach().float().cpu()
    tensors = {"emb": emb.contiguous()}
    meta = {"user_id": uid, "mode": mode, "k_shot": len(pairs), "steps": steps, "seed": seed,
            "train_images": [p["image"] for p in pairs], "train_prompts": [p["prompt"] for p in pairs],
            "final_loss_mean_last20": (sum(losses[-20:]) / len(losses[-20:])) if losses else None,
            "seconds": round(time.time() - t0, 1)}
    if mode in ("linear", "mean"):
        tensors["coef"] = user.coef.detach().float().cpu().contiguous()
        top = user.top_users(8)
        meta["top_train_users"] = [(bank.user_ids[i], round(c, 4)) for i, c in top]
    save_file(tensors, str(user_out / "user.safetensors"))
    json.dump(meta, open(user_out / "meta.json", "w"), indent=1)
    return user_out
