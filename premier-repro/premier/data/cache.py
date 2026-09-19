"""Pre-compute VAE latents and T5 / CLIP text embeddings so that training never has to
hold the 9.5 GB T5-XXL next to the 12 GB transformer.

Layout (cache_dir):
  latents_{res}/{sha1(image_path)}.safetensors   {"mean": [16,h,w] bf16, "std": [16,h,w] bf16}
  text_{L}/{sha1(prompt)}.safetensors            {"t5": [L,4096] bf16, "pooled": [768] bf16}
  text_{L}/empty.safetensors                     embedding of the empty prompt ""  (for the dispersion loss / "w/o PPM")
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def key_of(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_image_square(path: str | Path, resolution: int) -> torch.Tensor:
    """RGB -> resize shortest side -> center crop -> tensor in [-1, 1], [3, res, res]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = resolution / min(w, h)
    nw, nh = max(resolution, round(w * s)), max(resolution, round(h * s))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - resolution) // 2, (nh - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))
    x = torch.from_numpy(__import__("numpy").asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1
    return x


class FeatureCache:
    def __init__(self, cache_dir: str | Path, resolution: int, max_len: int):
        self.root = Path(cache_dir)
        self.res, self.max_len = resolution, max_len
        self.lat_dir = self.root / f"latents_{resolution}"
        self.txt_dir = self.root / f"text_{max_len}"
        self.lat_dir.mkdir(parents=True, exist_ok=True)
        self.txt_dir.mkdir(parents=True, exist_ok=True)

    # paths
    def latent_path(self, image_rel: str) -> Path:
        return self.lat_dir / f"{key_of(image_rel)}.safetensors"

    def text_path(self, prompt: str) -> Path:
        return self.txt_dir / f"{key_of(prompt)}.safetensors"

    @property
    def empty_text_path(self) -> Path:
        return self.txt_dir / "empty.safetensors"

    def has_latent(self, image_rel: str) -> bool:
        return self.latent_path(image_rel).exists()

    def has_text(self, prompt: str) -> bool:
        return self.text_path(prompt).exists()

    # loading
    def load_latent(self, image_rel: str, sample: bool = True, generator=None) -> torch.Tensor:
        d = load_file(str(self.latent_path(image_rel)))
        mean = d["mean"].float()
        if not sample:
            return mean
        return mean + d["std"].float() * torch.randn(mean.shape, generator=generator)

    def load_text(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = load_file(str(self.text_path(prompt)))
        return d["t5"], d["pooled"]

    def load_empty_text(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = load_file(str(self.empty_text_path))
        return d["t5"], d["pooled"]


@torch.no_grad()
def cache_latents(cache: FeatureCache, image_root: Path, images: list[str], vae, batch_size: int = 8):
    from ..model.loading import raw_to_model_latents, vae_encode_dist
    todo = [p for p in images if not cache.has_latent(p)]
    if not todo:
        return 0
    for i in tqdm(range(0, len(todo), batch_size), desc="latents"):
        chunk = todo[i:i + batch_size]
        px = torch.stack([load_image_square(image_root / p, cache.res) for p in chunk])
        mean, std = vae_encode_dist(vae, px)
        mean = raw_to_model_latents(mean, vae)
        std = std * vae.config.scaling_factor
        for p, m, s in zip(chunk, mean, std):
            save_file({"mean": m.to(torch.bfloat16).cpu().contiguous(), "std": s.to(torch.bfloat16).cpu().contiguous()},
                      str(cache.latent_path(p)))
    return len(todo)


@torch.no_grad()
def cache_texts(cache: FeatureCache, prompts: list[str], encoders, batch_size: int = 16):
    todo = sorted({p for p in prompts if not cache.has_text(p)})
    n = 0
    if not cache.empty_text_path.exists():
        t5, pooled = encoders.encode([""])
        save_file({"t5": t5[0].cpu().contiguous(), "pooled": pooled[0].cpu().contiguous()}, str(cache.empty_text_path))
    for i in tqdm(range(0, len(todo), batch_size), desc="text"):
        chunk = todo[i:i + batch_size]
        t5, pooled = encoders.encode(chunk)
        for p, a, b in zip(chunk, t5, pooled):
            save_file({"t5": a.cpu().contiguous(), "pooled": b.cpu().contiguous()}, str(cache.text_path(p)))
            n += 1
    return n
