"""Text-to-image generation for the evaluation pipeline.

Paper settings (Sec. 5.1): SD v1-5, PNDM scheduler, 50 steps, CFG scale 7.0.
The default here is **SDXL** instead, at 1024x1024 with the repo's own scheduler
-- PNDM is a v1-5 setting and SDXL is not tuned for it. Pass
``model_name=SD15_MODEL`` for the paper-faithful configuration; the resolution,
scheduler and batch defaults follow whichever model is selected.

Metrics computed under different models are not comparable with each other, so a
run records its model id in the metadata and results produced under one model
should not be mixed with another's.

``runwayml/stable-diffusion-v1-5`` was taken down in August 2024; the identical
weights are re-hosted at ``stable-diffusion-v1-5/stable-diffusion-v1-5``.

Seeding matters for the image metrics: every method must see the *same* seed for
a given test sample, otherwise sampling noise -- not the rewrite -- drives the
differences between methods. :func:`seed_for` derives that seed from the sample
key alone.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_FALLBACK = "sd-legacy/stable-diffusion-v1-5"
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_MODEL = os.environ.get("TV_SD_MODEL", SDXL_MODEL)


def is_sdxl(model_name: str) -> bool:
    return "xl" in model_name.rsplit("/", 1)[-1].lower()


def _enable_vae_slicing(pipe) -> None:
    """[2026 patch] Slice the VAE decode, whichever diffusers API is present.

    Decoding a whole batch of 1024x1024 latents at once is the peak allocation
    of an SDXL run. ``pipe.enable_vae_slicing()`` was removed in diffusers 0.40
    in favour of ``pipe.vae.enable_slicing()``.
    """
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    elif hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()


def seed_for(key: str, salt: str = "") -> int:
    """Deterministic per-sample seed, stable across processes and runs."""
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class GenConfig:
    steps: int = 50
    guidance_scale: float = 7.0
    # None means "whatever the model was trained for" -- 512 for v1-5, 1024 for
    # SDXL. Resolved in SDGenerator once the model is known.
    height: int | None = None
    width: int | None = None
    batch_size: int | None = None

    def resolve_for(self, model_name: str) -> "GenConfig":
        """Fill in the per-model defaults, leaving explicit values alone."""
        xl = is_sdxl(model_name)
        side = 1024 if xl else 512
        return GenConfig(
            steps=self.steps,
            guidance_scale=self.guidance_scale,
            height=self.height or side,
            width=self.width or side,
            # An SDXL batch costs ~4x a v1-5 one at its native resolution.
            batch_size=self.batch_size or (2 if xl else 8),
        )


class SDGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str = "float16",
        config: GenConfig | None = None,
        disable_safety_checker: bool = True,
    ):
        import torch
        from diffusers import AutoPipelineForText2Image, PNDMScheduler

        self.torch = torch
        self.model_name = model_name
        self.config = (config or GenConfig()).resolve_for(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)

        kwargs = dict(torch_dtype=torch_dtype)
        if disable_safety_checker and not is_sdxl(model_name):
            # The PIP dataset ships an NSFW flag of its own; the pipeline's
            # checker would blank images and make the metrics meaningless.
            # SDXL has no safety checker, and its pipeline rejects the argument.
            kwargs.update(safety_checker=None, requires_safety_checker=False)
        # AutoPipelineForText2Image resolves v1-5 vs SDXL from model_index.json;
        # the fp16 variant halves the download where a repo ships one.
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(model_name, variant="fp16", **kwargs)
        except Exception:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(model_name, **kwargs)
            except Exception:
                pipe = AutoPipelineForText2Image.from_pretrained(SD15_FALLBACK, **kwargs)
        if not is_sdxl(model_name):
            # PNDM is the paper's v1-5 setting. SDXL ships EulerDiscrete and is
            # not tuned for PNDM, so leave its own scheduler in place.
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        # Costs nothing in quality, removes the VAE decode memory spike.
        _enable_vae_slicing(pipe)
        self.pipe = pipe.to(self.device)
        # SD v1-5's text encoder holds 77 tokens; anything past that is dropped.
        # Rewritten prompts routinely reach that limit, so the tail of a long
        # rewrite never reaches the model. diffusers warns once per call, which
        # is unusable at this scale -- silence it and count instead.
        self._n_prompts = 0
        self._n_truncated = 0
        import logging

        logging.getLogger("diffusers.pipelines.stable_diffusion").setLevel(logging.ERROR)
        logging.getLogger("diffusers.pipelines.stable_diffusion_xl").setLevel(logging.ERROR)

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
