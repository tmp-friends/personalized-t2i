"""PremierModel = block-shared adapter + block-distinct adapter (+ helpers to save/load)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .adapter import PreferenceAdapter


@dataclass
class PremierConfig:
    user_tokens: int = 30
    user_dim: int = 1024
    text_dim: int = 4096          # T5-XXL hidden size
    mod_dim: int = 3072           # FLUX inner dim (modulation vector size)
    adapter_dim: int = 1024
    adapter_heads: int = 16
    adapter_blocks: int = 3
    use_shared: bool = True
    use_distinct: bool = True
    distinct_blocks: str = "all"  # "all" | "double" | "single" | "every2"
    n_double: int = 19
    n_single: int = 38
    adapter_text: str = "prompt"  # "prompt" (PPM) | "empty" (ablation "w/o PPM")
    zero_init: bool = True
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_cfg(d: dict) -> "PremierConfig":
        keys = PremierConfig.__dataclass_fields__.keys()
        return PremierConfig(**{k: v for k, v in d.items() if k in keys})


def _block_groups(cfg: PremierConfig) -> list[int | None]:
    n = cfg.n_double + cfg.n_single
    if cfg.distinct_blocks == "all":
        sel = list(range(n))
    elif cfg.distinct_blocks == "double":
        sel = list(range(cfg.n_double))
    elif cfg.distinct_blocks == "single":
        sel = list(range(cfg.n_double, n))
    elif cfg.distinct_blocks == "every2":
        sel = list(range(0, n, 2))
    else:
        raise ValueError(cfg.distinct_blocks)
    groups: list[int | None] = [None] * n
    for g, j in enumerate(sel):
        groups[j] = g
    return groups


class PremierModel(nn.Module):
    def __init__(self, cfg: PremierConfig):
        super().__init__()
        self.cfg = cfg
        self.block_group = _block_groups(cfg)
        self.n_groups = sum(g is not None for g in self.block_group)
        kw = dict(text_dim=cfg.text_dim, user_dim=cfg.user_dim, dim=cfg.adapter_dim, n_heads=cfg.adapter_heads,
                  n_blocks=cfg.adapter_blocks, out_dim=cfg.mod_dim, zero_init=cfg.zero_init)
        self.adapter_shared = PreferenceAdapter(n_out_groups=1, **kw) if cfg.use_shared else None
        self.adapter_distinct = PreferenceAdapter(n_out_groups=self.n_groups, **kw) if cfg.use_distinct else None

    # --------------------------------------------------------------------- deltas
    def compute_deltas(self, text_emb: torch.Tensor, user_emb: torch.Tensor,
                       empty_text_emb: torch.Tensor | None = None, use_shared: bool = True,
                       use_distinct: bool = True) -> dict[str, torch.Tensor | None]:
        """text_emb [B, T, 4096]; user_emb [B, M, D] -> {"shared": [B,T,3072]|None, "distinct": [B,T,J,3072]|None}"""
        if self.cfg.adapter_text == "empty":
            assert empty_text_emb is not None, "adapter_text='empty' needs the cached empty-prompt embedding"
            text_emb = empty_text_emb.to(text_emb.device).expand(text_emb.shape[0], -1, -1)
        out = {"shared": None, "distinct": None}
        if self.adapter_shared is not None and use_shared:
            out["shared"] = self.adapter_shared(text_emb, user_emb)
        if self.adapter_distinct is not None and use_distinct:
            out["distinct"] = self.adapter_distinct(text_emb, user_emb)
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # --------------------------------------------------------------------- io
    def save(self, path: str | Path, extra: dict | None = None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(path / "adapters.safetensors"))
        meta = {"premier_config": asdict(self.cfg)}
        if extra:
            meta.update(extra)
        json.dump(meta, open(path / "premier_config.json", "w"), indent=1)

    @staticmethod
    def load(path: str | Path, device="cpu") -> "PremierModel":
        path = Path(path)
        meta = json.load(open(path / "premier_config.json"))
        cfg = PremierConfig(**meta["premier_config"])
        m = PremierModel(cfg)
        m.load_state_dict(load_file(str(path / "adapters.safetensors")))
        return m.to(device)
