"""Stage 1: jointly train the preference adapters and the training-user embeddings.

    L = L_flow + λ_shared · L_disp^shared + λ_distinct · L_disp^distinct
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Cfg
from ..data.cache import FeatureCache
from ..data.dataset import MultiUserBatchSampler, PreferenceDataset, collate
from ..data.prefbench import load_manifest
from ..model.flux_modulation import premier_transformer_forward
from ..model.loading import (count_params, latent_image_ids, load_transformer, load_vae, pack_latents, text_ids)
from ..model.losses import dispersion_loss, flow_matching_loss, sample_timesteps
from ..model.premier import PremierConfig, PremierModel
from ..model.user_embedding import UserEmbeddingBank
from .common import (JsonlLogger, build_optimizer, latest_checkpoint, load_bank, preview_grid, prodigy_effective_lr,
                     save_checkpoint)


def ensure_texts_cached(cache: FeatureCache, prompts: list[str], cfg: Cfg):
    missing = [p for p in prompts if not cache.has_text(p)]
    if not missing and cache.empty_text_path.exists():
        return
    from ..data.cache import cache_texts
    from ..model.loading import TextEncoders
    enc = TextEncoders(cfg.model.aux_id, cfg.device, max_len=cfg.data.max_len)
    cache_texts(cache, missing, enc)
    enc.unload()


def train_stage1(cfg: Cfg):
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config.yaml")
    logger = JsonlLogger(out_dir / "log.jsonl")

    manifest = load_manifest(cfg.paths.manifest)
    train_users = manifest["train_users"][: cfg.data.n_train_users]
    cache = FeatureCache(cfg.paths.cache_dir, cfg.data.resolution, cfg.data.max_len)
    ensure_texts_cached(cache, list(cfg.train.sample_prompts), cfg)

    # ---- data
    user_index = {u: i for i, u in enumerate(train_users)}
    ds = PreferenceDataset(manifest, train_users, cache, user_index, per_user_limit=cfg.data.per_user_limit)
    if len(ds) == 0:
        raise RuntimeError("no cached samples found; run scripts/01_cache_features.py first")
    missing_users = [u for u in train_users if u not in ds.by_user]
    print(f"train samples: {len(ds)} from {len(ds.by_user)} users ({len(missing_users)} users without cached data)")
    sampler = MultiUserBatchSampler(ds, cfg.train.batch_size, seed=cfg.seed)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.train.num_workers, collate_fn=collate,
                    pin_memory=True, persistent_workers=cfg.train.num_workers > 0)
    empty_t5 = cache.load_empty_text()[0][None].to(dev)              # [1, L, 4096]
    disp_t5 = empty_t5[:, : cfg.train.disp_tokens]

    # ---- model
    pcfg = PremierConfig.from_cfg(cfg.premier.to_dict())
    premier = PremierModel(pcfg).to(dev)
    bank = UserEmbeddingBank(train_users, pcfg.user_tokens, pcfg.user_dim, cfg.premier.user_init_std).to(dev)
    print(f"adapters: {count_params(premier)/1e6:.1f}M params (shared {0 if premier.adapter_shared is None else count_params(premier.adapter_shared)/1e6:.1f}M, "
          f"distinct {0 if premier.adapter_distinct is None else count_params(premier.adapter_distinct)/1e6:.1f}M over {premier.n_groups} blocks); "
          f"user bank: {len(bank)} x {pcfg.user_tokens} x {pcfg.user_dim}")
    opt = build_optimizer(list(premier.parameters()) + list(bank.parameters()), cfg.train.optimizer, cfg.train.lr,
                          cfg.train.weight_decay)
    step = 0
    resume = cfg.train.resume
    if resume == "auto":
        resume = latest_checkpoint(out_dir)
    if resume:
        ck = Path(resume)
        premier.load_state_dict(PremierModel.load(ck).state_dict())
        bank.emb.data.copy_(load_bank(ck).emb.data)
        state = torch.load(ck / "optimizer.pt", map_location="cpu")
        opt.load_state_dict(state["optimizer"])
        step = int(state["step"])
        print(f"resumed from {ck} at step {step}")

    transformer = load_transformer(cfg.model.transformer_id, cfg.model.quant, dev,
                                   gradient_checkpointing=cfg.model.gradient_checkpointing)
    vae = load_vae(cfg.model.aux_id, dev) if cfg.train.sample_every else None
    print(f"cuda memory after loading: {torch.cuda.memory_allocated()/2**30:.1f} GB")

    res = cfg.data.resolution
    h = w = res // 8
    img_ids = latent_image_ids(h, w, dev, torch.bfloat16)
    txt_ids = text_ids(cfg.data.max_len, dev, torch.bfloat16)
    image_seq_len = (h // 2) * (w // 2)
    n_train = len(bank)

    def do_preview(tag: str):
        users = [None] + [bank.emb[i].detach()[None] for i in range(min(cfg.train.sample_users, n_train))]
        labels = ["base"] + train_users[: cfg.train.sample_users]
        preview_grid(transformer, vae, premier, cache, list(cfg.train.sample_prompts), users, labels, res,
                     cfg.train.sample_steps, cfg.eval.guidance, cfg.seed, out_dir / "previews" / f"{tag}.jpg")

    if cfg.train.sample_every and step == 0 and cfg.train.get("preview_at_start", True):
        do_preview("step-000000")

    premier.train()
    it = iter(dl)
    pbar = tqdm(total=cfg.train.steps, initial=step, dynamic_ncols=True)
    t_last = time.time()
    while step < cfg.train.steps:
        opt.zero_grad(set_to_none=True)
        acc = {"flow": 0.0, "disp_shared": 0.0, "disp_distinct": 0.0}
        for _ in range(cfg.train.grad_accum):
            batch = next(it)
            lat = pack_latents(batch["latent"].to(dev)).to(torch.bfloat16)
            t5 = batch["t5"].to(dev)
            pooled = batch["pooled"].to(dev)
            uidx = batch["user_idx"].to(dev)
            b = lat.shape[0]
            noise = torch.randn_like(lat)
            t = sample_timesteps(b, image_seq_len, dev, cfg.train.timestep_scheme, shift=cfg.train.timestep_shift)
            zt = ((1 - t)[:, None, None] * lat.float() + t[:, None, None] * noise.float()).to(torch.bfloat16)
            guid = torch.full((b,), float(cfg.train.guidance), device=dev)

            d = premier.compute_deltas(t5, bank(uidx), empty_text_emb=empty_t5)
            pred = premier_transformer_forward(transformer, zt, t5, pooled, t, img_ids, txt_ids, guid,
                                               delta_shared=d["shared"], delta_distinct=d["distinct"],
                                               block_group=premier.block_group)
            loss_flow = flow_matching_loss(pred, noise, lat)
            loss = loss_flow
            # dispersion loss on the empty prompt over (batch users ∪ random users)
            ls = ld = torch.zeros((), device=dev)
            if cfg.train.lambda_shared > 0 or cfg.train.lambda_distinct > 0:
                n_disp = min(cfg.train.disp_users, n_train)
                extra = torch.randperm(n_train, device=dev)
                disp_idx = torch.unique(torch.cat([uidx, extra]), sorted=False)
                # keep batch users first, then fill up with random ones
                disp_idx = torch.cat([uidx.unique(), extra[~torch.isin(extra, uidx)]])[:n_disp]
                dd = premier.compute_deltas(disp_t5.expand(len(disp_idx), -1, -1), bank(disp_idx))
                if dd["shared"] is not None and cfg.train.lambda_shared > 0:
                    ls = dispersion_loss(dd["shared"], cfg.train.disp_tau)
                    loss = loss + cfg.train.lambda_shared * ls
                if dd["distinct"] is not None and cfg.train.lambda_distinct > 0:
                    ld = dispersion_loss(dd["distinct"], cfg.train.disp_tau)
                    loss = loss + cfg.train.lambda_distinct * ld
            (loss / cfg.train.grad_accum).backward()
            acc["flow"] += loss_flow.item() / cfg.train.grad_accum
            acc["disp_shared"] += ls.item() / cfg.train.grad_accum
            acc["disp_distinct"] += ld.item() / cfg.train.grad_accum
        if cfg.train.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(list(premier.parameters()) + list(bank.parameters()), cfg.train.grad_clip)
        opt.step()
        step += 1
        pbar.update(1)

        if step % cfg.train.log_every == 0 or step == 1:
            dt = (time.time() - t_last) / (cfg.train.log_every if step > 1 else 1)
            t_last = time.time()
            rec = dict(step=step, **{k: round(v, 5) for k, v in acc.items()}, lr_eff=prodigy_effective_lr(opt),
                       sec_per_step=round(dt, 2), mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       delta_shared_rms=None if d["shared"] is None else round(d["shared"].detach().float().pow(2).mean().sqrt().item(), 5),
                       delta_distinct_rms=None if d["distinct"] is None else round(d["distinct"].detach().float().pow(2).mean().sqrt().item(), 5))
            logger.log(**rec)
            pbar.set_postfix({k: rec[k] for k in ("flow", "disp_shared", "disp_distinct", "sec_per_step")})

        if cfg.train.save_every and step % cfg.train.save_every == 0:
            save_checkpoint(out_dir / f"checkpoint-{step:06d}", premier, bank, opt, step, cfg.to_dict())
        if cfg.train.sample_every and step % cfg.train.sample_every == 0:
            premier.eval()
            do_preview(f"step-{step:06d}")
            premier.train()

    save_checkpoint(out_dir / f"checkpoint-{step:06d}", premier, bank, opt, step, cfg.to_dict())
    if cfg.train.sample_every:
        premier.eval()
        do_preview(f"step-{step:06d}")
    json.dump({"final_step": step, "train_users": train_users}, open(out_dir / "done.json", "w"))
    print(f"done. final checkpoint: {out_dir / f'checkpoint-{step:06d}'}")
