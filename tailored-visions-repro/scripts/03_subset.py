#!/usr/bin/env python
"""Stage 3: choose the fixed user subset the image-based metrics run on.

PMS and Image-Align need one SD v1-5 sample per method per test sample. Running
all 6,232 test samples across every method would be ~44k images (~24 GPU-hours)
to resolve gaps of 0.03-0.06 in a metric whose per-sample spread is ~0.1 -- at
1,000 samples the standard error is already ~0.003. So the image metrics run on
a fixed subset and ROUGE-L, which needs no images, runs on everything.

The subset is written to a file so that every method, and every later rerun,
scores the same users.

    python scripts/03_subset.py --n-users 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv.data import sample_user_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/user_data")
    ap.add_argument("--out", default="results/eval_subset.json")
    ap.add_argument("--n-users", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    user_ids = sample_user_ids(args.data_dir, args.n_users, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"user_ids": user_ids, "n_users": len(user_ids), "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(f"[subset] {len(user_ids)} users -> {out} (~{2 * len(user_ids)} test samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
