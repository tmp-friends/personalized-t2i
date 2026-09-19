#!/usr/bin/env python
"""Generate preference-aware images with the official FAN implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHED_UPSTREAM = ROOT / ".work/upstream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Foundation Encoders Are All You Need for Preference-Aware Personalization"
    )
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev", help="T2I model id")
    parser.add_argument("--sample_size", type=float, default=0.0, help="Sampling ratio for profiling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=["A photograph of an astronaut riding a horse"],
        help="Prompt for personalized generation",
    )
    parser.add_argument(
        "--ref",
        nargs="+",
        default=["A retro-futuristic space exploration movie poster with bold, vibrant colors"],
        help="Reference descriptions for conditioning",
    )
    parser.add_argument("--weight", nargs="+", type=float, default=[1], help="Reference weights")
    parser.add_argument("--alpha", type=float, default=0.4, help="Preference strength (0-1)")
    parser.add_argument("--size", type=int, default=0, help="Image size; 0 uses the model default")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=0,
        help="Number of inference steps; 0 uses the model default",
    )
    parser.add_argument("--skip", type=int, default=0, help="CLIP skip layer; 0 uses the model default")
    parser.add_argument("--save_path", default=str(ROOT / "outputs"), help="Output directory")
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype")
    parser.add_argument("--variant", default=None, help="Hugging Face weight variant, e.g. fp16")
    parser.add_argument(
        "--ref_weight_dir",
        default=str(PATCHED_UPSTREAM / "weight"),
        help="Directory containing L.pth and bigG.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (PATCHED_UPSTREAM / "fan").is_dir():
        raise SystemExit("run `python scripts/prepare_upstream.py` first")
    sys.path.insert(0, str(PATCHED_UPSTREAM))

    import torch
    from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
    from fan import personalized_t2i_encoder

    sample_size = 0.1 if args.sample_size else 0
    args.weight = list(args.weight) * len(args.ref) if len(args.weight) == 1 else args.weight
    generator = torch.Generator("cuda").manual_seed(args.seed)
    dtype = getattr(torch, args.dtype, torch.bfloat16)

    if "flux" in args.model.lower():
        pipeline = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype, variant=args.variant)
        size = 1024 if args.size == 0 else args.size
        num_inference_steps = 28 if args.num_inference_steps == 0 else args.num_inference_steps
        skip = -2 if args.skip == 0 else args.skip
    elif "stable-diffusion-3" in args.model.lower():
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.model, torch_dtype=dtype, variant=args.variant
        )
        size = 1024 if args.size == 0 else args.size
        num_inference_steps = 28 if args.num_inference_steps == 0 else args.num_inference_steps
        skip = -2 if args.skip == 0 else args.skip
    else:
        pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, variant=args.variant)
        if "xl-" in args.model.lower():
            size = 1024 if args.size == 0 else args.size
            skip = -2 if args.skip == 0 else args.skip
        else:
            size = 512 if args.size == 0 else args.size
            skip = -1 if args.skip == 0 else args.skip
        num_inference_steps = 50 if args.num_inference_steps == 0 else args.num_inference_steps
    pipeline = pipeline.to("cuda")

    encoder = personalized_t2i_encoder(pipeline, args.ref_weight_dir)
    with torch.no_grad():
        cond, pool_cond = encoder(
            args.prompt,
            args.ref,
            weight=args.weight,
            alpha=args.alpha,
            skip=skip,
            sample_size=sample_size,
        )
        images = pipeline(
            prompt_embeds=cond.type(dtype),
            pooled_prompt_embeds=pool_cond.type(dtype) if pool_cond is not None else None,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=size,
            width=size,
        ).images

    output = Path(args.save_path)
    output.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(output / f"{index:05d}.png")
    print(f"Inference complete. Images saved to {output}")


if __name__ == "__main__":
    main()
