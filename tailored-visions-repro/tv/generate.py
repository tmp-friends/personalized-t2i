"""Text-to-image generation with Stable Diffusion v1-5.

Paper settings (Sec. 5.1): SD v1-5, PNDM scheduler, 50 steps, CFG scale 7.0.

``runwayml/stable-diffusion-v1-5`` was taken down in August 2024; the identical
weights are re-hosted at ``stable-diffusion-v1-5/stable-diffusion-v1-5``.

Seeding matters for the image metrics: every method must see the *same* seed for
a given test sample, otherwise sampling noise -- not the rewrite -- drives the
differences between methods. :func:`seed_for` derives that seed from the sample
key alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_FALLBACK = "sd-legacy/stable-diffusion-v1-5"


def seed_for(key: str, salt: str = "") -> int:
    """Deterministic per-sample seed, stable across processes and runs."""
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class GenConfig:
    steps: int = 50
    guidance_scale: float = 7.0
    height: int = 512
    width: int = 512
    batch_size: int = 8


class SDGenerator:
    def __init__(
        self,
        model_name: str = SD15_MODEL,
        device: str | None = None,
        dtype: str = "float16",
        config: GenConfig | None = None,
        disable_safety_checker: bool = True,
    ):
        import torch
        from diffusers import PNDMScheduler, StableDiffusionPipeline

        self.torch = torch
        self.config = config or GenConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)

        kwargs = dict(torch_dtype=torch_dtype)
        if disable_safety_checker:
            # The PIP dataset ships an NSFW flag of its own; the pipeline's
            # checker would blank images and make the metrics meaningless.
            kwargs.update(safety_checker=None, requires_safety_checker=False)
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, variant="fp16", **kwargs)
        except Exception:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(model_name, **kwargs)
            except Exception:
                pipe = StableDiffusionPipeline.from_pretrained(SD15_FALLBACK, **kwargs)
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(self.device)
        # SD v1-5's text encoder holds 77 tokens; anything past that is dropped.
        # Rewritten prompts routinely reach that limit, so the tail of a long
        # rewrite never reaches the model. diffusers warns once per call, which
        # is unusable at this scale -- silence it and count instead.
        self._n_prompts = 0
        self._n_truncated = 0
        import logging

        logging.getLogger("diffusers.pipelines.stable_diffusion").setLevel(logging.ERROR)

    def _count_truncation(self, prompts: Sequence[str]) -> None:
        tok = self.pipe.tokenizer
        limit = tok.model_max_length
        for p in prompts:
            self._n_prompts += 1
            if len(tok(p, truncation=False)["input_ids"]) > limit:
                self._n_truncated += 1

    @property
    def truncation_rate(self) -> float:
        """Share of prompts longer than the 77-token CLIP context."""
        return self._n_truncated / self._n_prompts if self._n_prompts else 0.0

    def generate(self, prompts: Sequence[str], seeds: Sequence[int]):
        """Generate one image per prompt. Returns a list of PIL images."""
        torch = self.torch
        assert len(prompts) == len(seeds)
        images = []
        cfg = self.config
        self._count_truncation(prompts)
        for start in range(0, len(prompts), cfg.batch_size):
            chunk = list(prompts[start : start + cfg.batch_size])
            chunk_seeds = list(seeds[start : start + cfg.batch_size])
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in chunk_seeds]
            out = self.pipe(
                chunk,
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height,
                width=cfg.width,
                generator=generators,
            )
            images.extend(out.images)
        return images

    def unload(self) -> None:
        del self.pipe
        self.torch.cuda.empty_cache()


def save_grid(images: Sequence, path: str | Path, cols: int | None = None) -> None:
    from PIL import Image

    if not images:
        return
    cols = cols or len(images)
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
