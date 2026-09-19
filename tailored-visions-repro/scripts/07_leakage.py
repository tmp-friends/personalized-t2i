#!/usr/bin/env python
"""Stage 7: measure how much of each test prompt is already in that user's history.

ROUGE-L rewards recovering the ground-truth prompt. If a test prompt is also
sitting in the user's history, retrieval can hand it to the rewriter and the
score comes for free -- so any ROUGE-L number is worth only as much as this
check. The dataset's own splitter deduplicates before splitting, so *exact*
leakage should be near zero; near-duplicates are not prevented.

A sample counts as leaked when its ground-truth prompt has token-level F1 above
``--threshold`` against some prompt in the same user's history.

    python scripts/07_leakage.py --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv.metrics import token_f1  # noqa: E402


def _score_user(rec):
    history = rec["history"]
    lowered = {h.strip().lower() for h in history}
    out = []
    for test in rec["tests"]:
        gt = test["prompt"]
        best = max((token_f1(gt, h) for h in history), default=0.0)
        out.append(
            {
                "key": f"{rec['user_id']}/{test['sample_id']}",
                "user_id": rec["user_id"],
                "exact": gt.strip().lower() in lowered,
                "best_f1": round(best, 4),
                # The dataset only abbreviates prompts longer than 6 words, so
                # short prompts arrive as their own "shortened" query and the
                # no-op baseline scores a perfect ROUGE-L on them.
                "query_is_gt": test["query"].strip().lower() == gt.strip().lower(),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out", default="results/leakage.json")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument(
        "--rewrites-dir",
        default="results/rewrites",
        help="if present, also measure whether retrieval actually surfaces the leaked prompt",
    )
    args = ap.parse_args()

    records = [
        json.loads(l) for l in open(Path(args.cache_dir) / "history.jsonl", encoding="utf-8")
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(_score_user, records, chunksize=16):
            rows.extend(chunk)

    n = len(rows)
    n_exact = sum(r["exact"] for r in rows)
    n_query_is_gt = sum(r["query_is_gt"] for r in rows)
    leaked = [r["key"] for r in rows if r["best_f1"] >= args.threshold]
    f1s = sorted(r["best_f1"] for r in rows)
    summary = {
        "n_test_samples": n,
        "threshold": args.threshold,
        "exact_matches": n_exact,
        "exact_rate": n_exact / n if n else 0.0,
        "query_equals_gt": n_query_is_gt,
        "query_equals_gt_rate": n_query_is_gt / n if n else 0.0,
        "leaked": len(leaked),
        "leak_rate": len(leaked) / n if n else 0.0,
        "best_f1_percentiles": {
            p: f1s[min(int(n * p / 100), n - 1)] for p in (50, 75, 90, 95, 99)
        },
        "leaked_keys": leaked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[leakage] {n} test samples")
    print(f"  exact history match : {n_exact} ({100 * summary['exact_rate']:.2f}%)")
    print(f"  token-F1 >= {args.threshold}    : {len(leaked)} ({100 * summary['leak_rate']:.2f}%)")
    print(
        f"  query == gt prompt  : {n_query_is_gt} "
        f"({100 * summary['query_equals_gt_rate']:.2f}%)  <- free ROUGE-L for the no-op baseline"
    )
    print(f"  best-F1 percentiles : {summary['best_f1_percentiles']}")

    # Overlap with history is only half the story. What matters causally is
    # whether the retriever puts the near-duplicate in front of the rewriter,
    # so measure that directly from the retrieved prompts already persisted by
    # stage 2. Without this the inflation claim is inferred, not shown.
    by_key = {r["key"]: r for r in rows}
    retrieval = {}
    rewrites_dir = Path(args.rewrites_dir)
    for path in sorted(rewrites_dir.glob("*.jsonl")):
        method_rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if not any(r.get("retrieved") for r in method_rows):
            continue  # baselines do not retrieve
        hits = hits_leaked = n_leaked = 0
        for r in method_rows:
            best = max((token_f1(r["gt_prompt"], p) for p in r["retrieved"]), default=0.0)
            hit = best >= args.threshold
            hits += hit
            src = by_key.get(r["key"])
            if src and src["best_f1"] >= args.threshold:
                n_leaked += 1
                hits_leaked += hit
        retrieval[path.stem] = {
            "gt_in_topk_rate": hits / len(method_rows) if method_rows else 0.0,
            "recall_of_leaked": hits_leaked / n_leaked if n_leaked else 0.0,
            "n": len(method_rows),
        }
    if retrieval:
        summary["retrieval_leakage"] = retrieval
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n  does retrieval actually surface it? (token-F1 >= threshold vs top-k)")
        for name, st in retrieval.items():
            print(
                f"    {name:28s} gt-in-top-k {100 * st['gt_in_topk_rate']:5.2f}%   "
                f"of leaked samples recovered: {100 * st['recall_of_leaked']:5.2f}%"
            )
    else:
        print("\n  (no retrieval-bearing rewrites found; run 02_rewrite.py for the causal check)")

    print(f"[leakage] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
