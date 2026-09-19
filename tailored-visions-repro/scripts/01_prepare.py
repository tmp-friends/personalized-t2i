#!/usr/bin/env python
"""Stage 1: deduplicate every user's history and cache CLIP text embeddings.

Both products are reused by every method, and the dedup pass is the slowest
CPU-bound step in the pipeline (BLEU against a growing reference list, ~12M
pair comparisons over the full dataset), so it is done once here.

    python scripts/01_prepare.py                    # full dataset
    python scripts/01_prepare.py --limit-users 200  # quick pass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv.data import list_user_ids, load_user, write_jsonl  # noqa: E402
from tv.retrieval import dedup_history  # noqa: E402


def _dedup_one(args):
    data_dir, user_id, threshold, max_history = args
    user = load_user(Path(data_dir) / f"{user_id}.jsonl")
    history = dedup_history(user.history_prompts, threshold=threshold, max_history=max_history)
    return {
        "user_id": user_id,
        "n_raw": len(user.history),
        "n_dedup": len(history),
        "history": history,
        "tests": [
            {"sample_id": t.sample_id, "query": t.query, "prompt": t.prompt, "nsfw": t.nsfw}
            for t in user.test_samples
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/user_data")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--limit-users", type=int, default=0, help="0 = all users")
    ap.add_argument("--bleu-threshold", type=float, default=0.5)
    ap.add_argument(
        "--max-history",
        type=int,
        default=0,
        help="cap distinct prompts entering the BLEU pass (0 = no cap, faithful)",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    user_ids = list_user_ids(args.data_dir)
    if args.limit_users:
        user_ids = user_ids[: args.limit_users]
    print(f"[prepare] {len(user_ids)} users from {args.data_dir}")

    max_history = args.max_history or None
    t0 = time.time()
    jobs = [(args.data_dir, uid, args.bleu_threshold, max_history) for uid in user_ids]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(_dedup_one, jobs, chunksize=8), 1):
            records.append(rec)
            if i % 250 == 0 or i == len(jobs):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  dedup {i}/{len(jobs)}  ({rate:.1f} users/s)", flush=True)

    hist_path = cache_dir / "history.jsonl"
    write_jsonl(records, hist_path)
    raw = sum(r["n_raw"] for r in records)
    ded = sum(r["n_dedup"] for r in records)
    print(f"[prepare] history {raw} -> {ded} prompts after dedup, wrote {hist_path}")

    if args.skip_embeddings:
        return 0

    # ---- CLIP text embeddings for EBR -------------------------------------
    from tv.retrieval import EBRRetriever

    print(f"[prepare] encoding with {args.clip_model} ...")
    retriever = EBRRetriever(model_name=args.clip_model)

    texts: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    for rec in records:
        start = len(texts)
        texts.extend(rec["history"])
        offsets[rec["user_id"]] = (start, len(texts))
    t0 = time.time()
    hist_embs = retriever.encode(texts)
    print(f"  {len(texts)} history prompts in {time.time() - t0:.1f}s")

    query_keys, query_texts = [], []
    for rec in records:
        for t in rec["tests"]:
            query_keys.append(f"{rec['user_id']}/{t['sample_id']}")
            query_texts.append(t["query"])
    query_embs = retriever.encode(query_texts)

    from tv import prompt_templates as PT

    demo_embs = retriever.encode([ex[-2] for ex in PT.EXAMPLES])

    emb_path = cache_dir / "ebr_embeddings.npz"
    np.savez_compressed(
        emb_path,
        history=hist_embs.astype(np.float16),
        queries=query_embs.astype(np.float16),
        demos=demo_embs.astype(np.float16),
    )
    meta = {
        "clip_model": args.clip_model,
        "offsets": offsets,
        "query_keys": query_keys,
        "dim": int(hist_embs.shape[1]) if hist_embs.size else 0,
    }
    (cache_dir / "ebr_index.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[prepare] wrote {emb_path} ({emb_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
