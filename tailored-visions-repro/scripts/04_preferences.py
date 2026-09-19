#!/usr/bin/env python
"""Stage 3: distill each user's preference P_u into 5 phrases, for the PMS metric.

Paper, Sec. 3.3: "For each user u in PIP dataset, we summarize his preference
P_u into 5 phrases from his history prompts using ChatGPT." The summaries are
not in the released dataset and the summarization prompt is not published, so
this regenerates them -- see ``docs/DEVIATIONS.md``.

    python scripts/04_preferences.py --users-from results/eval_subset.json

Note that PMS scores a generated image against a summary of the same history the
rewriter conditions on. Any method that copies history-derived style tokens into
its output gains on PMS almost by construction; the metric is reported because
the paper reports it, not because it is neutral evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv import prompt_templates as PT  # noqa: E402
from tv.rewriter import build_rewriter, clean_output  # noqa: E402


def select_history(history: list[str], max_prompts: int) -> list[str]:
    """Evenly-spaced subsample, so long histories stay inside the context window.

    Even spacing rather than head/tail so the summary reflects the whole span of
    a user's activity, not just when they started or stopped.
    """
    if len(history) <= max_prompts:
        return history
    step = len(history) / max_prompts
    return [history[int(i * step)] for i in range(max_prompts)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out", default="data/cache/preferences.jsonl")
    ap.add_argument("--users-from", default="", help="JSON file with a 'user_ids' list")
    ap.add_argument("--limit-users", type=int, default=0)
    ap.add_argument("--max-history-prompts", type=int, default=30)
    ap.add_argument("--max-prompt-chars", type=int, default=400, help="truncate each history prompt")
    ap.add_argument("--backend", default="local", choices=["local", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        print(f"[preferences] {out_path} exists; use --overwrite")
        return 0

    records = [
        json.loads(l) for l in open(Path(args.cache_dir) / "history.jsonl", encoding="utf-8")
    ]
    if args.users_from:
        wanted = set(json.loads(Path(args.users_from).read_text(encoding="utf-8"))["user_ids"])
        records = [r for r in records if r["user_id"] in wanted]
    if args.limit_users:
        records = records[: args.limit_users]
    print(f"[preferences] summarizing {len(records)} users")

    inputs = []
    for rec in records:
        picked = select_history(rec["history"], args.max_history_prompts)
        block = "\n".join(f"- {p[: args.max_prompt_chars]}" for p in picked)
        inputs.append(PT.PREFERENCE_SUMMARY_TEMPLATE.format(block))

    kwargs = dict(batch_size=args.batch_size)
    if args.backend == "local":
        kwargs.update(max_new_tokens=args.max_new_tokens, temperature=0.0)
    if args.model:
        kwargs["model_name"] = args.model
    llm = build_rewriter(args.backend, **kwargs)

    outputs: list[str] = []
    chunk = args.batch_size * 8
    t0 = time.time()
    for start in range(0, len(inputs), chunk):
        outputs.extend(llm.rewrite_batch(inputs[start : start + chunk]))
        done = min(start + chunk, len(inputs))
        rate = done / max(time.time() - t0, 1e-9)
        print(f"    {done}/{len(inputs)} ({rate:.1f}/s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_bad = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec, raw in zip(records, outputs):
            text = clean_output(raw)
            # Keep the leading 5 comma-separated phrases; models sometimes add a
            # trailing sentence of commentary.
            phrases = [p.strip(" .;") for p in text.split(",") if p.strip(" .;")][:5]
            preference = ", ".join(phrases)
            if not preference:
                n_bad += 1
            fh.write(
                json.dumps(
                    {
                        "user_id": rec["user_id"],
                        "preference": preference,
                        "n_phrases": len(phrases),
                        "raw": raw.strip(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[preferences] wrote {out_path} ({len(records)} users, {n_bad} empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
