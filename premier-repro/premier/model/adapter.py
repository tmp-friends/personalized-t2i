"""Preference Adapter (Premier Sec. 4.1).

A stack of cross-attention blocks in which the *text tokens are the queries* and the
*learnable user preference embedding e_u (30 x 1024) provides keys / values*.  The
output is a preference-aware modulation direction Δ for every text token:

    block-shared   adapter : Δ_shared(e_u, e_{p_i})      -> [B, T, 3072]
    block-distinct adapter : Δ^j_distinct(e_u, e_{p_i})  -> [B, T, J, 3072]   (J = #DiT blocks modulated)

The block-distinct adapter is the same network whose final projection is widened to
J * 3072 ("expands the output dimension in the final layer").
The final projection is zero-initialised so that training starts exactly at the base
model (Δ = 0), as in adaLN-Zero / ControlNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.d = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        m = ctx.shape[1]
        q = self.q(x).view(b, t, self.h, self.d).transpose(1, 2)
        k, v = self.kv(ctx).view(b, m, 2, self.h, self.d).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(b, t, self.h * self.d))


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, n_heads)
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Linear(ffn_mult * dim, dim))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_q(x), self.ln_kv(ctx))
        x = x + self.ff(self.ln_ff(x))
        return x


class PreferenceAdapter(nn.Module):
    def __init__(self, text_dim: int = 4096, user_dim: int = 1024, dim: int = 1024, n_heads: int = 16,
                 n_blocks: int = 3, out_dim: int = 3072, n_out_groups: int = 1, zero_init: bool = True):
        super().__init__()
        self.out_dim, self.n_out_groups = out_dim, n_out_groups
        self.ln_in = nn.LayerNorm(text_dim)
        self.in_proj = nn.Linear(text_dim, dim)
        self.user_proj = nn.Linear(user_dim, dim)
        self.blocks = nn.ModuleList([CrossAttnBlock(dim, n_heads) for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim * n_out_groups)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, text_emb: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        """text_emb [B, T, text_dim] (T5 hidden states), user_emb [B, M, user_dim]
        -> [B, T, out_dim] if n_out_groups == 1 else [B, T, n_out_groups, out_dim]"""
        x = self.in_proj(self.ln_in(text_emb.to(self.in_proj.weight.dtype)))
        ctx = self.user_proj(user_emb.to(self.user_proj.weight.dtype))
        for blk in self.blocks:
            x = blk(x, ctx)
        delta = self.out_proj(self.ln_out(x))
        if self.n_out_groups == 1:
            return delta
        b, t, _ = delta.shape
        return delta.view(b, t, self.n_out_groups, self.out_dim)
