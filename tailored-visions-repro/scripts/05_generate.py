#!/usr/bin/env python
"""Stage 5: generate images and keep only their CLIP embeddings.

For each method on the evaluation subset, one SD v1-5 image per test sample, then
a CLIP ViT-B/32 embedding. Images are embedded and dropped as they come out --
persisting ~8k PNGs would cost several GB for no benefit, since both image
metrics only ever touch the embeddings. ``--save-images N`` keeps the first N
per method for qualitative figures.

The ``gt_proxy`` pseudo-method generates from the ground-truth *prompt*; it is
what Image-Align compares against now that the dataset's images are gone.

    python scripts/05_generate.py --methods table2

Run after ``02_rewrite.py`` has finished: SD v1-5 and the rewriter both want the
GPU, and this script assumes it has it to itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv.generate import GenConfig, SDGenerator, seed_for  # noqa: E402
from tv.metrics import CLIPScorer  # noqa: E402

GT_PROXY = "gt_proxy"


def load_rows(path: Path, user_ids: set[str] | None) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if user_ids is not None:
        rows = [r for r in rows if r["user_id"] in user_ids]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites-dir", default="results/rewrites")
    ap.add_argument("--out-dir", default="results/embeddings")
    ap.add_argument("--subset", default="results/eval_subset.json")
    ap.add_argument("--methods", default="table2", help="comma-separated names, or 'table2'/'all'")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-images", type=int, default=0)
    ap.add_argument("--images-dir", default="results/images")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rewrites_dir = Path(args.rewrites_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_ids = None
    if args.subset and Path(args.subset).exists():
        user_ids = set(json.loads(Path(args.subset).read_text(encoding="utf-8"))["user_ids"])
        print(f"[generate] restricted to {len(user_ids)} users from {args.subset}")

    if args.methods in ("table2", "all"):
        from tv.pipeline import icl_ablation_methods, table2_methods, topk_ablation_methods

        specs = table2_methods()
        if args.methods == "all":
            specs = specs + topk_ablation_methods() + icl_ablation_methods()
        names = [s.name for s in specs]
    else:
        names = [n.strip() for n in args.methods.split(",") if n.strip()]

    available = [n for n in names if (rewrites_dir / f"{n}.jsonl").exists()]
    missing = sorted(set(names) - set(available))
    if missing:
        print(f"[generate] skipping (no rewrites yet): {', '.join(missing)}")
    if not available:
        raise SystemExit("nothing to generate")

    # The ground-truth proxy is derived from any method file (all share gt_prompt).
    jobs: list[tuple[str, list[dict], list[str]]] = []
    base_rows = load_rows(rewrites_dir / f"{available[0]}.jsonl", user_ids)[: args.limit or None]
    jobs.append((GT_PROXY, base_rows, [r["gt_prompt"] for r in base_rows]))
    for name in available:
        rows = load_rows(rewrites_dir / f"{name}.jsonl", user_ids)[: args.limit or None]
        jobs.append((name, rows, [r["rewrite"] for r in rows]))

    pending = [j for j in jobs if args.overwrite or not (out_dir / f"{j[0]}.npz").exists()]
    if not pending:
        print("[generate] all embeddings present; use --overwrite to redo")
        return 0

    cfg = GenConfig(steps=args.steps, guidance_scale=args.guidance_scale, batch_size=args.batch_size)
    print("[generate] loading SD v1-5 ...")
    sd = SDGenerator(config=cfg)
    scorer = CLIPScorer()

    truncation: dict[str, float] = {}
    for name, rows, prompts in pending:
        keys = [r["key"] for r in rows]
        # Same seed per test sample across every method: differences between
        # methods must come from the prompt, not from sampling noise.
        seeds = [seed_for(k) for k in keys]
        embs = np.zeros((len(prompts), 512), dtype=np.float32)
        t0 = time.time()
        saved = 0
        for start in range(0, len(prompts), args.batch_size):
            stop = min(start + args.batch_size, len(prompts))
            images = sd.generate(prompts[start:stop], seeds[start:stop])
            # The safety checker blanks flagged images to solid black, which
            # would silently produce meaningless CLIP embeddings rather than an
            # error. It is disabled in SDGenerator; this verifies that took.
            if start == 0:
                means = [float(np.asarray(im).mean()) for im in images]
                n_black = sum(m < 1.0 for m in means)
                print(
                    f"    {name}: first batch pixel-mean "
                    f"min={min(means):.1f} max={max(means):.1f}, {n_black} near-black"
                )
                if n_black:
                    print("    WARNING: blank images -- safety checker may still be active")
            embs[start:stop] = scorer.encode_images(images)
            if saved < args.save_images:
                img_dir = Path(args.images_dir) / name
                img_dir.mkdir(parents=True, exist_ok=True)
                for key, img in zip(keys[start:stop], images):
                    if saved >= args.save_images:
                        break
                    img.save(img_dir / f"{key.replace('/', '_')}.png")
                    saved += 1
            del images
            if start % (args.batch_size * 10) == 0 or stop == len(prompts):
                rate = stop / max(time.time() - t0, 1e-9)
                eta = (len(prompts) - stop) / max(rate, 1e-9)
                print(
                    f"    {name}: {stop}/{len(prompts)} ({rate:.2f} img/s, eta {eta / 60:.1f}m)",
                    flush=True,
                )
        np.savez_compressed(out_dir / f"{name}.npz", keys=np.array(keys), embeddings=embs)
        truncation[name] = sd.truncation_rate
        print(
            f"[generate] {name}: {time.time() - t0:.1f}s  "
            f"{100 * sd.truncation_rate:.1f}% of prompts exceeded SD's 77-token limit "
            f"-> {out_dir / f'{name}.npz'}",
            flush=True,
        )
        sd._n_prompts = sd._n_truncated = 0

    (out_dir / "gen_meta.json").write_text(
        json.dumps(
            {
                "model": "stable-diffusion-v1-5",
                "scheduler": "PNDM",
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "resolution": [512, 512],
                "clip_metric_model": "openai/clip-vit-base-patch32",
                "seeding": "sha256(sample_key), shared across methods",
                "n_samples": len(base_rows),
                "prompt_truncation_rate": truncation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
