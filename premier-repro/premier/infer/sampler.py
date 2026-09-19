"""Euler flow-matching sampler for FLUX.1-dev with Premier preference modulation.

The preference adapters do not depend on the timestep, so Δ is computed once per
generation and reused for every denoising step (paper: ~1 s overhead per image).
"""
from __future__ import annotations

import torch

from ..model.flux_modulation import premier_transformer_forward
from ..model.loading import (flux_shift, latent_image_ids, pack_latents, shift_sigmas, text_ids, unpack_latents,
                             vae_decode)


@torch.no_grad()
def generate(transformer, vae, t5: torch.Tensor, pooled: torch.Tensor, height: int = 512, width: int = 512,
             steps: int = 28, guidance: float = 3.5, seed: int | None = 0, premier=None,
             user_emb: torch.Tensor | None = None, empty_t5: torch.Tensor | None = None,
             use_shared: bool = True, use_distinct: bool = True, delta_scale: float = 1.0,
             return_latents: bool = False):
    device = next(transformer.parameters()).device
    dtype = torch.bfloat16
    b, t_len = t5.shape[0], t5.shape[1]
    h, w = height // 8, width // 8
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    lat = torch.randn(b, 16, h, w, generator=gen, device=device, dtype=torch.float32)
    lat = pack_latents(lat).to(dtype)
    img_ids = latent_image_ids(h, w, device, dtype)
    txt_ids = text_ids(t_len, device, dtype)
    seq_len = lat.shape[1]
    sig = shift_sigmas(torch.linspace(1.0, 1.0 / steps, steps, device=device, dtype=torch.float32), flux_shift(seq_len))
    sig = torch.cat([sig, sig.new_zeros(1)])
    guid = torch.full((b,), guidance, device=device, dtype=torch.float32)

    t5 = t5.to(device, dtype)
    pooled = pooled.to(device, dtype)
    delta_shared = delta_distinct = block_group = None
    if premier is not None and user_emb is not None:
        ue = user_emb.to(device)
        if ue.shape[0] == 1 and b > 1:
            ue = ue.expand(b, -1, -1)
        d = premier.compute_deltas(t5, ue, empty_text_emb=empty_t5, use_shared=use_shared, use_distinct=use_distinct)
        delta_shared = None if d["shared"] is None else d["shared"] * delta_scale
        delta_distinct = None if d["distinct"] is None else d["distinct"] * delta_scale
        block_group = premier.block_group

    for i in range(steps):
        t = sig[i].expand(b)
        pred = premier_transformer_forward(transformer, lat, t5, pooled, t, img_ids, txt_ids, guid,
                                           delta_shared=delta_shared, delta_distinct=delta_distinct,
                                           block_group=block_group)
        lat = (lat.float() + (sig[i + 1] - sig[i]) * pred.float()).to(dtype)

    latents = unpack_latents(lat, h, w)
    images = vae_decode(vae, latents)
    if return_latents:
        return images, latents
    return images


def to_pil(images: torch.Tensor):
    from PIL import Image
    arr = (images.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in arr]


def image_grid(pils, cols: int, pad: int = 4, bg=(255, 255, 255)):
    from PIL import Image
    if not pils:
        return None
    w, h = pils[0].size
    rows = (len(pils) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), bg)
    for i, im in enumerate(pils):
        grid.paste(im, ((i % cols) * (w + pad), (i // cols) * (h + pad)))
    return grid
