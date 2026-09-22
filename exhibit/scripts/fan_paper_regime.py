#!/usr/bin/env python3
"""A2: reproduce the official FAN regime on base SDXL 1.0 (control experiment).

The exhibit's own FAN policies run on a fine-tuned anime checkpoint with
Danbooru-tag prompts and centroid-averaged references (see
``exhibit/scripts/fan_probe.py`` and ``docs/superpowers/plans/
2026-09-22-fan-strength-experiments.md``, section A2). Any of those choices
could be why the personalization effect looks weak in the exhibit. This
script removes them: it runs the official FAN encoder on
``stabilityai/stable-diffusion-xl-base-1.0`` with natural-language prompts and
natural-language references, at a small alpha sweep, so we get an upper bound
on how strong FAN's visible effect is in its own regime. It is a diagnostic,
not a change to exhibit policy: results are recorded as their own
``experiment_kind`` and are not subject to the exhibit's screen/refine rules.

This is a controller+worker in one process (unlike ``evaluate_fan.py``, which
spawns a worker under the GPU lease): it must hold ``exhibit.gpu.gpu_lease``
for the whole GPU section itself, exactly the way ``fan_probe.py`` is run
under the FAN environment.

Run the real thing from the repository root, in the fan-repro environment
(FAN monkey-patches CLIP attention and needs the pinned transformers/diffusers
versions there, not the exhibit GPU environment)::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    PYTHONPATH=fan-repro/.work/upstream:exhibit/src \\
    fan-repro/.venv/bin/python exhibit/scripts/fan_paper_regime.py

``--dry-run`` only builds and prints the job list as JSON; it does not import
torch/diffusers/transformers/numpy/PIL and does not touch the GPU, so it also
works from the exhibit venv for a CPU-only test::

    PYTHONPATH=exhibit/src exhibit/.venv/bin/python \\
        exhibit/scripts/fan_paper_regime.py --dry-run

Outputs land under ``exhibit/outputs/fan-strength/a2-paper-regime/``:
``images/<topic>-<history>-alpha<alpha>-seed<seed>.png`` (the alpha-0 baseline
is named ``<topic>-plain-alpha0.0-seed<seed>.png``), ``report.json``, and
``contact-sheet.jpg``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from exhibit.config import OUTPUTS, ROOT, write_json
from exhibit.gpu import gpu_lease

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_HISTORIES_FILE = ROOT / "configs/evaluation-cases/histories-strength.json"
DEFAULT_HISTORY_IDS = ["warm-sentence", "cool-sentence"]
DEFAULT_ALPHAS = [0.0, 0.4, 0.7]
DEFAULT_SEEDS = [42]
DEFAULT_OUT = OUTPUTS / "fan-strength" / "a2-paper-regime"

TOPICS = [
    {
        "id": "cat",
        "prompt": (
            "A brown-haired girl with green eyes wearing a sweater, "
            "holding a cat by a window."
        ),
    },
    {
        "id": "tokyo",
        "prompt": (
            "A young man with short black hair wearing a jacket, "
            "with the Tokyo skyline at night."
        ),
    },
]

# Official generation settings (README regime): 1024x1024, 50 steps, pipeline
# default scheduler and guidance scale, no negative prompt, skip -2 (SDXL).
DEFAULT_SIZE = 1024
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_SKIP = -2
DEFAULT_SKIP_PA = [0]
DEFAULT_SAMPLE_SIZE = 0.0


def format_alpha(alpha):
    """0.0 -> '0.0', 0.4 -> '0.4', 0.70 -> '0.7' -- stable, no trailing zeros."""
    text = f"{float(alpha):.3f}".rstrip("0").rstrip(".")
    return text if "." in text else text + ".0"


def load_histories(path, history_ids):
    """Read named histories from ``histories-strength.json`` verbatim.

    Returns a list of ``{"id": ..., "refs": [{"ref_id", "text", "weight"}, ...]}``
    in the order of ``history_ids``, using exactly the ref texts and weights
    the fixture declares (no copy kept here).
    """
    raw = json.loads(Path(path).read_text())
    by_id = {entry["id"]: entry for entry in raw.get("histories", [])}
    missing = [history_id for history_id in history_ids if history_id not in by_id]
    if missing:
        raise ValueError(f"unknown history id(s) in {path}: {missing}")
    return [by_id[history_id] for history_id in history_ids]


def build_jobs(topics, histories, alphas, seeds):
    """The full job list: one reference-free baseline plus one job per
    (topic, history, non-zero alpha, seed). The alpha-0 baseline is rendered
    once per (topic, seed) with ``ref_prompt=None`` -- it doesn't depend on
    the history, so it is not repeated per history.
    """
    nonzero_alphas = sorted({round(float(a), 6) for a in alphas if float(a) != 0.0})
    jobs = []
    for topic in topics:
        for seed in seeds:
            jobs.append(
                {
                    "kind": "plain",
                    "topic": topic["id"],
                    "prompt": topic["prompt"],
                    "history": None,
                    "refs": None,
                    "weights": None,
                    "alpha": 0.0,
                    "seed": seed,
                    "file": f"{topic['id']}-plain-alpha0.0-seed{seed}.png",
                }
            )
        for history in histories:
            refs = [ref["text"] for ref in history["refs"]]
            weights = [ref["weight"] for ref in history["refs"]]
            for alpha in nonzero_alphas:
                for seed in seeds:
                    jobs.append(
                        {
                            "kind": "personalized",
                            "topic": topic["id"],
                            "prompt": topic["prompt"],
                            "history": history["id"],
                            "refs": refs,
                            "weights": weights,
                            "alpha": alpha,
                            "seed": seed,
                            "file": (
                                f"{topic['id']}-{history['id']}-"
                                f"alpha{format_alpha(alpha)}-seed{seed}.png"
                            ),
                        }
                    )
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--histories-file", default=str(DEFAULT_HISTORIES_FILE), help="histories JSON"
    )
    parser.add_argument(
        "--history-ids",
        nargs="+",
        default=list(DEFAULT_HISTORY_IDS),
        help="history ids to load from --histories-file",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="alpha values; 0.0 is always rendered once per topic/seed as the "
        "reference-free baseline regardless of history",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--skip-pa", type=int, nargs="+", default=list(DEFAULT_SKIP_PA))
    parser.add_argument("--use-attn-mask", action="store_true")
    parser.add_argument(
        "--sample-size",
        type=float,
        default=DEFAULT_SAMPLE_SIZE,
        help="reference profiling sample_size passed to the FAN encoder "
        "(0 = all refs; e.g. 0.1 for the official CLI's ratio)",
    )
    parser.add_argument(
        "--weight-dir",
        default=str(ROOT.parent / "fan-repro/.work/upstream/weight"),
        help="directory containing L.pth and bigG.pth",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--timeout", type=float, default=1800, help="GPU lease seconds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the job list as JSON and exit; no torch/GPU work",
    )
    return parser.parse_args(argv)


def build_job_plan(args):
    histories = load_histories(args.histories_file, args.history_ids)
    jobs = build_jobs(TOPICS, histories, args.alphas, args.seeds)
    files = [job["file"] for job in jobs]
    if len(files) != len(set(files)):
        raise ValueError("job filenames are not unique")
    return histories, jobs


def cosine_report(a, b):
    """Mean cosine similarity over tokens between two same-shaped embeddings."""
    import torch

    x, y = a.detach().float(), b.detach().float()
    cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return round(cos.mean().item(), 8)


def pixel_mae(path_a, path_b):
    import numpy as np
    from PIL import Image

    left = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    right = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    return round(float(np.abs(left - right).mean()), 4)


def load_pipeline(model, dtype, torch):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_encoder(pipe, weight_dir):
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
    return stable_diffusion_xl(large, bigG)


def encode(encoder, torch, prompt, refs, weights, alpha, *, skip, skip_pa,
           use_attn_mask, sample_size):
    with torch.no_grad():
        if refs:
            cond, pool = encoder(
                prompt,
                refs,
                weight=weights,
                alpha=alpha,
                skip=skip,
                skip_pa=skip_pa,
                use_attn_mask=use_attn_mask,
                sample_size=sample_size,
            )
        else:
            cond, pool = encoder(
                prompt,
                None,
                skip=skip,
                skip_pa=skip_pa,
                use_attn_mask=use_attn_mask,
            )
    return cond, pool


def build_contact_sheet(out_dir, jobs, topics, histories, seeds):
    """Rows = (topic, history, seed); columns = alpha values, alpha 0 first.

    The alpha-0 image doesn't depend on history, so the same baseline tile is
    reused as column 1 of every history's row for that topic/seed -- that's
    what makes each row a legible before/after comparison.
    """
    from PIL import Image, ImageDraw

    tile = 320
    caption_h = 24
    by_key = {(j["topic"], j["history"], j["seed"]): j for j in jobs}
    rows = []
    for topic in topics:
        for history in histories:
            for seed in seeds:
                plain_job = by_key.get((topic["id"], None, seed))
                row_jobs = [job for job in jobs
                            if job["topic"] == topic["id"]
                            and job["history"] == history["id"]
                            and job["seed"] == seed]
                row_jobs = sorted(row_jobs, key=lambda job: job["alpha"])
                if plain_job is not None:
                    row_jobs = [plain_job] + row_jobs
                if row_jobs:
                    rows.append((topic["id"], history["id"], seed, row_jobs))
    if not rows:
        return None
    max_cols = max(len(row[3]) for row in rows)
    sheet = Image.new(
        "RGB", (max_cols * tile, len(rows) * (tile + caption_h)), (32, 32, 32)
    )
    draw = ImageDraw.Draw(sheet)
    images_dir = Path(out_dir) / "images"
    for row_index, (topic_id, history_id, seed, row_jobs) in enumerate(rows):
        for col_index, job in enumerate(row_jobs):
            image_path = images_dir / job["file"]
            if not image_path.exists():
                continue
            thumbnail = Image.open(image_path).convert("RGB").resize((tile, tile))
            x = col_index * tile
            y = row_index * (tile + caption_h)
            sheet.paste(thumbnail, (x, y))
            caption = (
                f"{topic_id}/{history_id or 'plain'}/a{format_alpha(job['alpha'])}"
                f"/s{seed}"
            )
            draw.rectangle([x, y + tile, x + tile, y + tile + caption_h], fill=(0, 0, 0))
            draw.text((x + 4, y + tile + 4), caption, fill=(255, 255, 255))
    sheet_path = Path(out_dir) / "contact-sheet.jpg"
    sheet.save(sheet_path, format="JPEG", quality=85)
    return sheet_path


def run(args):
    import diffusers
    import torch
    import transformers

    histories, jobs = build_job_plan(args)
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16

    cancel = threading.Event()
    deadline = time.monotonic() + args.timeout

    report = {
        "experiment": "a2-paper-regime",
        "experiment_kind": "diagnostic",
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "settings": {
            "model": args.model,
            "size": args.size,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "skip": args.skip,
            "skip_pa": args.skip_pa,
            "use_attn_mask": args.use_attn_mask,
            "sample_size": args.sample_size,
            "negative_prompt": None,
        },
        "topics": TOPICS,
        "histories": histories,
        "alphas": args.alphas,
        "seeds": args.seeds,
    }

    records = []
    baselines = {}
    baseline_embeds = {}
    job_embeds = {}

    with gpu_lease(cancel, deadline):
        pipe = load_pipeline(args.model, dtype, torch)
        encoder = build_encoder(pipe, args.weight_dir)

        for job in jobs:
            started = time.monotonic()
            cond, pool = encode(
                encoder,
                torch,
                job["prompt"],
                job["refs"],
                job["weights"],
                job["alpha"],
                skip=args.skip,
                skip_pa=args.skip_pa,
                use_attn_mask=args.use_attn_mask,
                sample_size=args.sample_size,
            )
            generator = torch.Generator(device="cuda").manual_seed(job["seed"])
            image = pipe(
                prompt_embeds=cond.type(dtype),
                pooled_prompt_embeds=pool.type(dtype) if pool is not None else None,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.size,
                width=args.size,
                generator=generator,
            ).images[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = round(time.monotonic() - started, 3)
            image_path = images_dir / job["file"]
            image.save(image_path)

            key = (job["topic"], job["seed"])
            job_embeds[job["file"]] = (cond, pool)
            if job["kind"] == "plain":
                baselines[key] = image_path
                baseline_embeds[key] = (cond, pool)

            records.append(
                {
                    "file": job["file"],
                    "topic": job["topic"],
                    "history": job["history"],
                    "alpha": job["alpha"],
                    "seed": job["seed"],
                    "seconds": seconds,
                }
            )
            print(f"{job['file']} {seconds}s", flush=True)

        # Cosine similarity is a GPU tensor op, so it stays inside the lease;
        # pixel MAE only reads the saved PNGs back off disk, so it can wait.
        for record in records:
            key = (record["topic"], record["seed"])
            base_path = baselines.get(key)
            image_path = images_dir / record["file"]
            base_cond, base_pool = baseline_embeds.get(key, (None, None))
            cond, pool = job_embeds[record["file"]]
            if base_cond is None or image_path == base_path:
                record["cosine_vs_alpha0"] = {"hidden": 1.0, "pooled": 1.0}
            else:
                record["cosine_vs_alpha0"] = {
                    "hidden": cosine_report(base_cond, cond),
                    "pooled": cosine_report(base_pool, pool)
                    if base_pool is not None and pool is not None
                    else None,
                }

    for record in records:
        key = (record["topic"], record["seed"])
        base_path = baselines.get(key)
        image_path = images_dir / record["file"]
        record["pixel_mae_vs_alpha0"] = (
            (0.0 if base_path == image_path else pixel_mae(base_path, image_path))
            if base_path
            else None
        )

    report["images"] = records
    write_json(out_dir / "report.json", report)
    sheet_path = build_contact_sheet(out_dir, jobs, TOPICS, histories, args.seeds)
    print(f"report: {out_dir / 'report.json'}")
    if sheet_path:
        print(f"contact sheet: {sheet_path}")


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        _, jobs = build_job_plan(args)
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
