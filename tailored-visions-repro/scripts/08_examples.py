#!/usr/bin/env python
"""Stage 8: dump qualitative examples, the paper's Figure 8 in text form.

For each sampled test case: the shortened query, what retrieval pulled from that
user's history, and what each method produced. Sampling is deterministic so the
same cases come back on a rerun.

    python scripts/08_examples.py --n 12
    python scripts/08_examples.py --n 6 --require-distinct-retrieval
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv.metrics import token_f1  # noqa: E402

ORDER = [
    "shortened_prompt",
    "promptist",
    "general_pr",
    "personalized_pr_bm25",
    "personalized_pr_ebr",
    "personalized_pr_icl_bm25",
    "personalized_pr_icl_ebr",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites-dir", default="results/rewrites")
    ap.add_argument("--out", default="results/EXAMPLES.md")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor", default="personalized_pr_icl_ebr")
    ap.add_argument(
        "--require-distinct-retrieval",
        action="store_true",
        help="only show cases where the retrieved prompts are not near-copies of "
        "the ground truth, i.e. where personalization had to do real work",
    )
    args = ap.parse_args()

    rewrites_dir = Path(args.rewrites_dir)
    methods = {}
    for name in ORDER:
        path = rewrites_dir / f"{name}.jsonl"
        if path.exists():
            methods[name] = {
                json.loads(l)["key"]: json.loads(l) for l in open(path, encoding="utf-8")
            }
    if args.anchor not in methods:
        raise SystemExit(f"anchor method {args.anchor} not found in {rewrites_dir}")

    keys = sorted(set.intersection(*(set(m) for m in methods.values())))
    if args.require_distinct_retrieval:
        keys = [
            k
            for k in keys
            if max(
                (token_f1(methods[args.anchor][k]["gt_prompt"], p)
                 for p in methods[args.anchor][k]["retrieved"]),
                default=0.0,
            )
            < 0.5
        ]
    rng = random.Random(args.seed)
    picked = rng.sample(keys, min(args.n, len(keys)))

    lines = [
        "# Qualitative examples",
        "",
        "The paper's Figure 8 in text form: what each method does to the same query,",
        "and what the user's history offered it. Cases are sampled deterministically",
        f"(seed {args.seed}) -- not cherry-picked.",
        "",
    ]
    for i, key in enumerate(sorted(picked), 1):
        anchor = methods[args.anchor][key]
        lines += [
            f"## {i}. user `{anchor['user_id']}` — query: *{anchor['query']}*",
            "",
            f"**Ground-truth prompt** (what the user actually typed): {anchor['gt_prompt']}",
            "",
            "**Retrieved history:**",
        ]
        for j, p in enumerate(anchor["retrieved"], 1):
            lines.append(f"{j}. {p}")
        lines += ["", "**Rewrites:**", "", "| Method | Output |", "|---|---|"]
        for name in ORDER:
            row = methods.get(name, {}).get(key)
            if row:
                text = row["rewrite"].replace("|", r"\|")
                lines.append(f"| `{name}` | {text} |")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[examples] wrote {len(picked)} cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
