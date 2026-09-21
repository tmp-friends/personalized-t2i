#!/usr/bin/env python3
"""G0 probe: can FAN personalize the exhibit's Illustrious XL v2.0 pipeline?

Loads the pinned single-file checkpoint exactly like ``exhibit.workers.generate``,
builds the FAN SDXL encoder on the pipeline's own text encoders, checks numerical
equivalence against ``pipe.encode_prompt``, then generates a preference matrix and
records timings, VRAM and pixel MAE.

FAN monkey-patches CLIP attention and needs transformers>=4.57,<5, so this must run
in the fan-repro environment, not the exhibit GPU environment:

    cd <repo root>
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=fan-repro/.work/upstream:exhibit/src \
    fan-repro/.venv/bin/python exhibit/scripts/fan_probe.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from exhibit.config import CONFIG, OUTPUTS, REPO, write_json
from PIL import Image

# Danbooru/Illustrious tag style, matching the vocabulary the exhibit already uses.
PREFERENCE_REFS = [
    "cool color palette, blue and teal tones, soft lighting",
    "dramatic lighting, strong shadows, rim light, night",
    "watercolor painting, soft brushwork, painterly texture, muted colors",
]
OPPOSITE_REFS = [
    "warm color palette, amber and orange tones, sunlight",
    "cel shading, clean lineart, anime coloring, flat colors",
    "open mouth, laughing, cheerful expression, lively atmosphere",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topic", default="cat", help="topic id from configs/demo.json")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument(
        "--upstream",
        default=str(
            REPO / CONFIG.get("fan", {}).get("upstream", "fan-repro/.work/upstream")
        ),
        help="materialized FAN upstream tree (contains fan/ and weight/)",
    )
    p.add_argument("--seeds", type=int, default=4, help="how many config seeds to use")
    p.add_argument("--alpha", type=float, default=0.4, help="main preference strength")
    p.add_argument(
        "--alpha-sweep",
        type=float,
        nargs="*",
        default=[0.2, 0.6],
        help="extra alphas rendered on the first seed only",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=CONFIG.get("fan", {}).get("skip", -2),
        help="CLIP skip layer",
    )
    p.add_argument("--no-generate", action="store_true", help="encoder checks only")
    return p.parse_args()


def vram(reset=False):
    out = {
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


def compare(a, b):
    """Numerical agreement between two embedding tensors of the same shape."""
    x, y = a.detach().float(), b.detach().float()
    cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return {
        "shape": list(a.shape),
        "max_abs_diff": round((x - y).abs().max().item(), 8),
        "mean_abs_diff": round((x - y).abs().mean().item(), 8),
        "rel_max_abs_diff": round(
            ((x - y).abs().max() / y.abs().max().clamp_min(1e-12)).item(), 8
        ),
        "mean_cosine": round(cos.mean().item(), 8),
        "min_cosine": round(cos.min().item(), 8),
    }


def pixel_mae(a, b):
    left = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    right = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    return round(float(np.abs(left - right).mean()), 4)


def load_pipeline(settings):
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download

    # Same path as exhibit.workers.generate(): weights come from the pinned
    # single-file checkpoint, only architecture/tokenizers from the config repo.
    checkpoint = hf_hub_download(
        settings["model"],
        filename=settings["checkpoint"],
        revision=settings["revision"],
        local_files_only=True,
    )
    config = settings["pipeline_config"]
    config_path = str(
        Path(
            hf_hub_download(
                config["model"],
                filename="model_index.json",
                revision=config["revision"],
                local_files_only=True,
            )
        ).parent
    )
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint,
        config=config_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_encoder(pipe, weight_dir):
    """FAN SDXL encoder on the pipeline's own text encoders.

    ``fan.wrapper.personalized_t2i_encoder`` dispatches on ``pipeline.name_or_path``,
    which ``from_single_file`` leaves empty, so the SDXL branch is wired by hand.
    """
    from fan import FAN
    from fan.wrapper import stable_diffusion_xl

    large = FAN(
        pipe.text_encoder, pipe.tokenizer, decoder=str(Path(weight_dir) / "L.pth")
    ).eval()
    bigG = FAN(
        pipe.text_encoder_2,
        pipe.tokenizer_2,
        decoder=str(Path(weight_dir) / "bigG.pth"),
    ).eval()
    return stable_diffusion_xl(large, bigG), large, bigG


def encode_fan(encoder, prompt, refs, weights, alpha, skip):
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.no_grad():
        cond, pool = (
            encoder(prompt, refs, weight=weights, alpha=alpha, skip=skip)
            if refs
            else encoder(prompt, None, skip=skip)
        )
    torch.cuda.synchronize()
    return (
        cond.to(torch.float16),
        pool.to(torch.float16),
        round(time.monotonic() - started, 4),
    )


def class_token_report(pipe, large, bigG, texts):
    """Where FAN's class-token detector thinks the pooled token is.

    In the reference branch FAN pools with ``ClassTokenDecoder`` instead of the EOS
    position, and it feeds the detector the *pre* final-layer-norm hidden state.
    On long Danbooru-style tag prompts that lands on a padding token, so the
    resulting ``pooled_prompt_embeds`` does not match ``pipe.encode_prompt``.
    """
    rows = []
    for name, fan, tok, model in (
        ("L", large, pipe.tokenizer, pipe.text_encoder),
        ("bigG", bigG, pipe.tokenizer_2, pipe.text_encoder_2),
    ):
        for text in texts:
            batch = tok(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            ids = batch["input_ids"].to("cuda")
            with torch.no_grad():
                states = model(ids, output_hidden_states=True).hidden_states
                rows.append(
                    {
                        "encoder": name,
                        "text": text[:48],
                        "eos_index": int(ids.argmax(-1).item()),
                        "detector_on_unnormalized": int(fan.decoder(states[-1]).item()),
                        "detector_on_normalized": int(
                            fan.decoder(
                                fan.normalize_text_hidden_state(states[-1])
                            ).item()
                        ),
                    }
                )
    return rows


def main():
    args = parse_args()
    settings = CONFIG["generation"]
    topic = next(t for t in CONFIG["topics"] if t["id"] == args.topic)
    out_dir = Path(args.out) if args.out else OUTPUTS / "fan-probe"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.upstream)

    prompt = topic["basic_prompt_en"] + ", " + settings["positive_prompt_tail"] + "."
    negative = settings["negative_prompt"]
    seeds = CONFIG["seeds"][: args.seeds]

    import accelerate
    import diffusers
    import transformers

    report = {
        "probe": "fan-g0",
        "topic": topic["id"],
        "prompt": prompt,
        "negative_prompt": negative,
        "seeds": seeds,
        "skip": args.skip,
        "references": {"preference": PREFERENCE_REFS, "opposite": OPPOSITE_REFS},
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "generation_settings": {
            k: settings[k]
            for k in (
                "model",
                "revision",
                "checkpoint",
                "width",
                "height",
                "steps",
                "guidance_scale",
            )
        },
    }

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    pipe = load_pipeline(settings)
    torch.cuda.synchronize()
    report["load_seconds"] = round(time.monotonic() - started, 3)
    report["vram_after_load"] = vram(reset=True)
    report["dtypes"] = {
        "text_encoder": str(pipe.text_encoder.dtype),
        "text_encoder_2": str(pipe.text_encoder_2.dtype),
        "unet": str(pipe.unet.dtype),
        "pipeline_name_or_path": str(getattr(pipe, "name_or_path", "")),
    }

    started = time.monotonic()
    encoder, large, bigG = build_encoder(pipe, Path(args.upstream) / "weight")
    report["encoder_build_seconds"] = round(time.monotonic() - started, 3)
    report["dtypes"]["fan_decoder_L"] = str(large.decoder.classifier[0].weight.dtype)
    report["dtypes"]["fan_decoder_bigG"] = str(bigG.decoder.classifier[0].weight.dtype)

    # (g) token lengths -- FAN truncates to 77.
    tokenizers = {"tokenizer": pipe.tokenizer, "tokenizer_2": pipe.tokenizer_2}
    report["token_lengths"] = {
        name: {
            "model_max_length": tok.model_max_length,
            "prompt": len(tok(prompt, truncation=False)["input_ids"]),
            "negative_prompt": len(tok(negative, truncation=False)["input_ids"]),
            "preference_refs": [
                len(tok(r, truncation=False)["input_ids"]) for r in PREFERENCE_REFS
            ],
            "opposite_refs": [
                len(tok(r, truncation=False)["input_ids"]) for r in OPPOSITE_REFS
            ],
        }
        for name, tok in tokenizers.items()
    }

    # (c) equivalence against the pipeline's own encoder.
    with torch.no_grad():
        ref_cond, _, ref_pool, _ = pipe.encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    plain_cond, plain_pool, plain_seconds = encode_fan(
        encoder, prompt, None, None, 0.0, args.skip
    )
    zero_cond, zero_pool, zero_seconds = encode_fan(
        encoder, prompt, PREFERENCE_REFS[:1], [1.0], 0.0, args.skip
    )
    report["equivalence"] = {
        "fan_plain_vs_pipeline": {
            "hidden": compare(plain_cond, ref_cond),
            "pooled": compare(plain_pool, ref_pool),
            "encode_seconds": plain_seconds,
        },
        "fan_alpha0_oneref_vs_pipeline": {
            "hidden": compare(zero_cond, ref_cond),
            "pooled": compare(zero_pool, ref_pool),
            "encode_seconds": zero_seconds,
        },
        "fan_alpha0_oneref_vs_fan_plain": {
            "hidden": compare(zero_cond, plain_cond),
            "pooled": compare(zero_pool, plain_pool),
        },
    }
    alpha_cond, alpha_pool, alpha_seconds = encode_fan(
        encoder, prompt, PREFERENCE_REFS, [1.0, 1.0, 1.0], args.alpha, args.skip
    )
    report["equivalence"][f"fan_alpha{args.alpha}_vs_fan_plain"] = {
        "hidden": compare(alpha_cond, plain_cond),
        "pooled": compare(alpha_pool, plain_pool),
        "encode_seconds": alpha_seconds,
    }
    report["class_token_detector"] = class_token_report(
        pipe, large, bigG, [prompt, negative, *PREFERENCE_REFS]
    )

    # (d) negative prompt through the same FAN path.
    with torch.no_grad():
        _, ref_neg_cond, _, ref_neg_pool = pipe.encode_prompt(
            prompt="",
            negative_prompt=negative,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
    neg_cond, neg_pool, neg_seconds = encode_fan(
        encoder, negative, None, None, 0.0, args.skip
    )
    report["equivalence"]["negative_fan_vs_pipeline"] = {
        "hidden": compare(neg_cond, ref_neg_cond),
        "pooled": compare(neg_pool, ref_neg_pool),
        "encode_seconds": neg_seconds,
    }
    report["vram_after_encode"] = vram(reset=True)

    if args.no_generate:
        write_json(out_dir / "report.json", report)
        print(json.dumps(report["equivalence"], indent=2))
        return

    common = {
        "num_inference_steps": settings["steps"],
        "guidance_scale": settings["guidance_scale"],
        "width": settings["width"],
        "height": settings["height"],
    }
    first = seeds[0]
    jobs = (
        [{"name": "a-baseline-string", "seed": s, "mode": "string"} for s in seeds]
        + [
            {"name": "b-fan-plain", "seed": s, "refs": None, "alpha": 0.0}
            for s in seeds
        ]
        + [
            {
                "name": f"c-fan-a{args.alpha}",
                "seed": s,
                "refs": PREFERENCE_REFS,
                "weights": [1.0, 1.0, 1.0],
                "alpha": args.alpha,
            }
            for s in seeds
        ]
        + [
            {
                "name": f"c2-fan-a{args.alpha}-plainpool",
                "seed": s,
                "refs": PREFERENCE_REFS,
                "weights": [1.0, 1.0, 1.0],
                "alpha": args.alpha,
                "pooled": "plain",
            }
            for s in seeds
        ]
        + [
            {
                "name": f"d-fan-a{a}",
                "seed": first,
                "refs": PREFERENCE_REFS,
                "weights": [1.0, 1.0, 1.0],
                "alpha": a,
            }
            for a in args.alpha_sweep
        ]
        + [
            {
                "name": f"e-fan-a{args.alpha}-w211",
                "seed": first,
                "refs": PREFERENCE_REFS,
                "weights": [2.0, 1.0, 1.0],
                "alpha": args.alpha,
            },
            {
                "name": f"f-fan-a{args.alpha}-no-ref2",
                "seed": first,
                "refs": [PREFERENCE_REFS[0], PREFERENCE_REFS[2]],
                "weights": [1.0, 1.0],
                "alpha": args.alpha,
            },
            {
                "name": f"g-fan-a{args.alpha}-opposite",
                "seed": first,
                "refs": OPPOSITE_REFS,
                "weights": [1.0, 1.0, 1.0],
                "alpha": args.alpha,
            },
            # (f) patch persistence: a plain string call after every FAN call.
            {"name": "h-baseline-string-after-fan", "seed": first, "mode": "string"},
        ]
    )

    records = []
    for job in jobs:
        seed = job["seed"]
        generator = torch.Generator(device="cuda").manual_seed(seed)
        encode_seconds = None
        if job.get("mode") == "string":
            call = {"prompt": prompt, "negative_prompt": negative}
        else:
            cond, pool, encode_seconds = encode_fan(
                encoder,
                prompt,
                job.get("refs"),
                job.get("weights"),
                job.get("alpha", 0.0),
                args.skip,
            )
            if job.get("pooled") == "plain":
                # FAN's reference branch pools with its class-token detector; on long
                # tag prompts that misses the EOS token, so optionally substitute the
                # unpersonalized pooled embedding.
                pool = plain_pool
            call = {
                "prompt_embeds": cond,
                "pooled_prompt_embeds": pool,
                "negative_prompt_embeds": neg_cond,
                "negative_pooled_prompt_embeds": neg_pool,
            }
        torch.cuda.synchronize()
        started = time.monotonic()
        image = pipe(**call, **common, generator=generator).images[0]
        torch.cuda.synchronize()
        seconds = round(time.monotonic() - started, 3)
        path = images_dir / f"{job['name']}_seed{seed}.png"
        image.save(path)
        records.append(
            {
                **{k: v for k, v in job.items() if k not in ("mode", "pooled")},
                "mode": job.get("mode", "fan-embeds"),
                "pooled": None
                if job.get("mode") == "string"
                else job.get("pooled", "fan"),
                "file": path.name,
                "seconds": seconds,
                "encode_seconds": encode_seconds,
                "vram": vram(),
            }
        )
        print(f"{job['name']} seed={seed} {seconds}s", flush=True)

    baselines = {
        r["seed"]: images_dir / r["file"]
        for r in records
        if r["name"] == "a-baseline-string"
    }
    fan_plains = {
        r["seed"]: images_dir / r["file"] for r in records if r["name"] == "b-fan-plain"
    }
    for record in records:
        base = baselines.get(record["seed"])
        plain = fan_plains.get(record["seed"])
        record["pixel_mae_vs_baseline"] = (
            pixel_mae(base, images_dir / record["file"]) if base else None
        )
        record["pixel_mae_vs_fan_plain"] = (
            pixel_mae(plain, images_dir / record["file"]) if plain else None
        )
    report["images"] = records
    report["vram_peak_overall"] = vram()
    report["patch_persistence"] = {
        "description": "plain string call repeated after all FAN calls, same seed",
        "pixel_mae_vs_first_baseline": next(
            r["pixel_mae_vs_baseline"]
            for r in records
            if r["name"] == "h-baseline-string-after-fan"
        ),
    }

    # (f) also compare the encoder numerically before/after the FAN calls.
    after_cond, after_pool, _ = encode_fan(encoder, prompt, None, None, 0.0, args.skip)
    with torch.no_grad():
        after_ref_cond, _, after_ref_pool, _ = pipe.encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    report["patch_persistence"]["pipeline_encode_prompt_before_vs_after"] = {
        "hidden": compare(after_ref_cond, ref_cond),
        "pooled": compare(after_ref_pool, ref_pool),
    }
    report["patch_persistence"]["fan_plain_before_vs_after"] = {
        "hidden": compare(after_cond, plain_cond),
        "pooled": compare(after_pool, plain_pool),
    }

    write_json(out_dir / "report.json", report)
    print(f"report: {out_dir / 'report.json'}")
    print(f"images: {images_dir}")


if __name__ == "__main__":
    main()
