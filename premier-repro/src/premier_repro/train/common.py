"""Shared training utilities: optimiser, checkpoints, logging, preview generation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def build_optimizer(params, name: str = "prodigy", lr: float = 1.0, weight_decay: float = 0.01):
    params = [p for p in params if p.requires_grad]
    if name == "prodigy":
        from prodigyopt import Prodigy
        return Prodigy(params, lr=lr, weight_decay=weight_decay, decouple=True, use_bias_correction=True,
                       safeguard_warmup=True)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    raise ValueError(name)


def prodigy_effective_lr(opt) -> float | None:
    g = opt.param_groups[0]
    if "d" in g:
        return float(g["d"] * g["lr"])
    return None


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a")
        self.t0 = time.time()

    def log(self, **kw):
        kw.setdefault("time", round(time.time() - self.t0, 1))
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()


def save_checkpoint(path: Path, premier, bank, optimizer, step: int, cfg: dict, extra: dict | None = None):
    path.mkdir(parents=True, exist_ok=True)
    premier.save(path, extra={"step": step, "train_config": cfg})
    if bank is not None:
        save_file({"emb": bank.emb.detach().cpu().contiguous()}, str(path / "user_bank.safetensors"))
        json.dump({"user_ids": bank.user_ids, "step": step}, open(path / "user_bank.json", "w"))
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": step, **(extra or {})}, path / "optimizer.pt")


def load_bank(path: Path, device="cpu"):
    from ..model.user_embedding import UserEmbeddingBank
    meta = json.load(open(path / "user_bank.json"))
    emb = load_file(str(path / "user_bank.safetensors"))["emb"]
    bank = UserEmbeddingBank(meta["user_ids"], n_tokens=emb.shape[1], dim=emb.shape[2])
    bank.emb.data.copy_(emb)
    return bank.to(device)


def latest_checkpoint(out_dir: Path) -> Path | None:
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return cks[-1] if cks else None


@torch.no_grad()
def preview_grid(transformer, vae, premier, cache, prompts: list[str], user_embs: list[torch.Tensor | None],
                 labels: list[str], resolution: int, steps: int, guidance: float, seed: int, out_path: Path):
    """rows = users (first row: base model, user_emb=None), cols = prompts."""
    from ..infer.sampler import generate, image_grid, to_pil
    empty_t5 = cache.load_empty_text()[0][None].to(next(premier.parameters()).device)
    pils = []
    for ue in user_embs:
        for p in prompts:
            t5, pooled = cache.load_text(p)
            img = generate(transformer, vae, t5[None], pooled[None], resolution, resolution, steps=steps,
                           guidance=guidance, seed=seed, premier=premier, user_emb=None if ue is None else ue,
                           empty_t5=empty_t5)
            pils += to_pil(img)
    grid = image_grid(pils, cols=len(prompts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    json.dump({"rows": labels, "cols": prompts}, open(out_path.with_suffix(".json"), "w"), indent=1)
    return grid
