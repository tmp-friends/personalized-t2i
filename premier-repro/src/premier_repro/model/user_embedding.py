"""Learnable user preference embeddings (Premier).

* UserEmbeddingBank      : one trainable tensor (n_tokens x dim, default 30 x 1024) per training user (stage 1).
* LinearCombinationUser  : new user = Σ_k α_k e_k over the frozen training-user bank; only α is trained (stage 2, paper's default).
* DirectUser             : new user embedding trained from scratch (stage-2 ablation "direct training").
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UserEmbeddingBank(nn.Module):
    def __init__(self, user_ids: list[str], n_tokens: int = 30, dim: int = 1024, init_std: float = 1.0):
        super().__init__()
        self.user_ids = list(user_ids)
        self.index = {u: i for i, u in enumerate(self.user_ids)}
        self.emb = nn.Parameter(torch.randn(len(self.user_ids), n_tokens, dim) * init_std)

    def __len__(self) -> int:
        return len(self.user_ids)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb[idx]

    def ids_to_idx(self, ids: list[str], device=None) -> torch.Tensor:
        return torch.tensor([self.index[u] for u in ids], dtype=torch.long, device=device)


class LinearCombinationUser(nn.Module):
    """e_new = Σ_k α_k · e_k  (bank frozen).  α initialised uniformly (= mean user)."""

    def __init__(self, bank: torch.Tensor, init: str = "uniform", init_coef: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("bank", bank.detach().clone(), persistent=False)
        n = bank.shape[0]
        if init_coef is not None:
            coef = init_coef.clone().float()
        elif init == "uniform":
            coef = torch.full((n,), 1.0 / n)
        elif init == "zeros":
            coef = torch.zeros(n)
        else:
            raise ValueError(init)
        self.coef = nn.Parameter(coef)

    def forward(self) -> torch.Tensor:
        return torch.einsum("n,nmd->md", self.coef.to(self.bank.dtype), self.bank)[None]

    def top_users(self, k: int = 5) -> list[tuple[int, float]]:
        vals, idx = self.coef.detach().abs().topk(min(k, self.coef.numel()))
        return [(int(i), float(self.coef[i])) for i in idx]


class DirectUser(nn.Module):
    def __init__(self, n_tokens: int = 30, dim: int = 1024, init_std: float = 1.0, init: torch.Tensor | None = None):
        super().__init__()
        if init is not None:
            self.emb = nn.Parameter(init.detach().clone().float().view(1, n_tokens, dim))
        else:
            self.emb = nn.Parameter(torch.randn(1, n_tokens, dim) * init_std)

    def forward(self) -> torch.Tensor:
        return self.emb
