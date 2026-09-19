#!/usr/bin/env python
"""Stage 2: retrieve + rewrite. Produces one JSONL of rewritten prompts per method.

This stage alone is enough for the paper's ROUGE-L column -- no image generation
involved -- so it runs at full test-set scale by default.

    python scripts/02_rewrite.py --methods table2
    python scripts/02_rewrite.py --methods ablation_topk,ablation_icl --limit 2000

The LLM is loaded once and reused across methods; Promptist and the no-op
baseline are handled without it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv.pipeline import (  # noqa: E402
    GENERAL_PR,
    PERSONALIZED,
    PROMPTIST,
    SHORTENED,
    MethodSpec,
    build_rewriter_input,
    demo_copy_ratio,
    icl_ablation_methods,
    table2_methods,
    topk_ablation_methods,
)
from tv.retrieval import BM25Retriever, EBRRetriever  # noqa: E402
from tv.rewriter import clean_output  # noqa: E402
from tv import prompt_templates as PT  # noqa: E402


def load_cache(cache_dir: Path):
    records = [json.loads(l) for l in open(cache_dir / "history.jsonl", encoding="utf-8")]
    index = json.loads((cache_dir / "ebr_index.json").read_text(encoding="utf-8"))
    embs = np.load(cache_dir / "ebr_embeddings.npz")
    return (
        records,
        index,
        {
            "history": embs["history"].astype(np.float32),
            "queries": embs["queries"].astype(np.float32),
            "demos": embs["demos"].astype(np.float32),
        },
    )


def resolve_methods(names: str) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    seen: set[str] = set()
    for token in [t.strip() for t in names.split(",") if t.strip()]:
        if token == "table2":
            group = table2_methods()
        elif token == "ablation_topk":
            group = topk_ablation_methods()
        elif token == "ablation_icl":
            group = icl_ablation_methods()
        elif token == "all":
            group = table2_methods() + topk_ablation_methods() + icl_ablation_methods()
        else:
            group = [m for m in
                     table2_methods() + topk_ablation_methods() + icl_ablation_methods()
                     if m.name == token]
            if not group:
                raise SystemExit(f"unknown method: {token}")
        for spec in group:
            if spec.name not in seen:
                seen.add(spec.name)
                specs.append(spec)
    return specs


def build_samples(records, index, embs, limit: int, user_limit: int,
                  sample_users: int = 0, sample_seed: int = 0,
                  users_from: str = ""):
    """Flatten users into (user_record, test, query_embedding) triples.

    ``limit``/``user_limit`` take a prefix, which is the first users by id --
    fine for smoke tests, but a different population from the full run. Use
    ``sample_users`` for anything whose numbers get reported.
    """
    key_to_row = {k: i for i, k in enumerate(index["query_keys"])}
    if users_from:
        wanted = set(json.loads(Path(users_from).read_text(encoding="utf-8"))["user_ids"])
        records = [r for r in records if r["user_id"] in wanted]
    if sample_users and sample_users < len(records):
        import random

        rng = random.Random(sample_seed)
        chosen = set(rng.sample([r["user_id"] for r in records], sample_users))
        records = [r for r in records if r["user_id"] in chosen]
    samples = []
    for rec in records[: user_limit or None]:
        for test in rec["tests"]:
            key = f"{rec['user_id']}/{test['sample_id']}"
            row = key_to_row.get(key)
            samples.append((rec, test, embs["queries"][row] if row is not None else None))
    return samples[: limit or None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="results/rewrites")
    ap.add_argument("--methods", default="table2")
    ap.add_argument("--limit", type=int, default=0, help="max test samples (0 = all)")
    ap.add_argument("--user-limit", type=int, default=0)
    ap.add_argument(
        "--users-from",
        default="",
        help="JSON file with a 'user_ids' list (e.g. results/eval_subset.json). "
        "Use for the ablations so they land on the same users the image metrics score.",
    )
    ap.add_argument(
        "--sample-users",
        type=int,
        default=0,
        help="deterministically sample N users instead of taking a prefix "
        "(use for ablations so their population matches across runs)",
    )
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--backend", default="local", choices=["local", "openai", "echo"])
    ap.add_argument("--model", default=None, help="rewriter model id")
    # 48 overflows 24 GB once the 5-shot ICL prompts get long; 16 is safe there
    # and still saturates the GPU on the short-prompt conditions.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--promptist-batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-words", type=int, default=0, help="0 = no truncation (faithful)")
    ap.add_argument(
        "--prompt-type",
        default="sentence",
        choices=["sentence", "noun_phrase", "noun"],
        help="Table 3's input scales. 'sentence' uses the released query; the other "
        "two are re-derived from the ground-truth prompt with spaCy (a reconstruction "
        "-- the paper's derivation code is not published). Use a separate --out-dir.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_words = args.max_words or None

    records, index, embs = load_cache(cache_dir)
    samples = build_samples(
        records,
        index,
        embs,
        args.limit,
        args.user_limit,
        args.sample_users,
        args.sample_seed,
        args.users_from,
    )
    if args.prompt_type != "sentence":
        from tv.shorten import Shortener

        print(f"[rewrite] deriving '{args.prompt_type}' queries from the ground-truth prompts ...")
        shortener = Shortener()
        # Derived from the full prompt, not from the released query: all three of
        # Table 3's scales are abbreviations of the same ground truth.
        short = shortener.shorten_batch([t["prompt"] for _, t, _ in samples], args.prompt_type)
        for (_, test, _), text in zip(samples, short):
            test["query"] = text
        # Query embeddings no longer match the queries; EBR must re-encode.
        from tv.retrieval import EBRRetriever

        encoder = EBRRetriever(model_name=index["clip_model"])
        new_q = encoder.encode(short)
        samples = [(rec, test, new_q[i]) for i, (rec, test, _) in enumerate(samples)]
        del encoder
        import torch

        torch.cuda.empty_cache()  # the rewriter needs this VRAM back
    specs = resolve_methods(args.methods)
    print(f"[rewrite] {len(samples)} test samples x {len(specs)} methods")
    for s in specs:
        print(f"    - {s.name:28s} {s.describe()}")

    pending = [s for s in specs if args.overwrite or not (out_dir / f"{s.name}.jsonl").exists()]
    if not pending:
        print("[rewrite] all methods already present; use --overwrite to redo")
        return 0

    hist_offsets = index["offsets"]
    bm25 = BM25Retriever()
    demo_embs = embs["demos"]

    llm = None
    promptist = None
    if any(s.uses_llm for s in pending) and args.backend != "echo":
        from tv.rewriter import build_rewriter

        kwargs = dict(batch_size=args.batch_size, temperature=args.temperature)
        if args.backend == "local":
            kwargs.update(max_new_tokens=args.max_new_tokens)
        if args.model:
            kwargs["model_name"] = args.model
        print(f"[rewrite] loading {args.backend} rewriter ...")
        llm = build_rewriter(args.backend, **kwargs)

    for spec in pending:
        t0 = time.time()
        rows = []
        inputs = []
        for rec, test, q_emb in samples:
            retrieved, scores, demos = [], [], []
            if spec.uses_retrieval:
                history = rec["history"]
                if spec.retriever == "bm25":
                    r = bm25.retrieve(history, test["query"], k=spec.num_retrieval)
                    retrieved, scores = r.prompts, r.scores
                else:
                    lo, hi = hist_offsets[rec["user_id"]]
                    h_emb = embs["history"][lo:hi]
                    if q_emb is None or h_emb.size == 0:
                        retrieved, scores = history[: spec.num_retrieval], []
                    else:
                        sims = h_emb @ q_emb
                        order = np.lexsort((np.arange(len(sims)), -sims))[: spec.num_retrieval]
                        retrieved = [history[i] for i in order]
                        scores = [float(sims[i]) for i in order]
                if spec.icl_shot and q_emb is not None:
                    demos = PT.rank_demos(q_emb, demo_embs)
            inputs.append(build_rewriter_input(spec, _as_sample(rec, test), retrieved, demos))
            rows.append(
                {
                    "user_id": rec["user_id"],
                    "sample_id": test["sample_id"],
                    "key": f"{rec['user_id']}/{test['sample_id']}",
                    "query": test["query"],
                    "gt_prompt": test["prompt"],
                    "nsfw": test.get("nsfw", ""),
                    "retrieved": retrieved,
                    "retrieval_scores": scores,
                    "demos_used": [d[-1] for d in demos[: spec.icl_shot]],
                }
            )

        if spec.kind == SHORTENED:
            outputs = [r["query"] for r in rows]
        elif spec.kind == PROMPTIST:
            from tv.baselines import PromptistRewriter

            if promptist is None:
                print("[rewrite] loading Promptist ...")
                promptist = PromptistRewriter(batch_size=args.promptist_batch_size)
            outputs = promptist.rewrite_batch(inputs)
        else:
            if llm is None:
                raise SystemExit(f"method {spec.name} needs an LLM backend")
            outputs = _run_llm(llm, inputs, spec.name)

        for row, raw in zip(rows, outputs):
            cleaned = clean_output(raw, max_words=max_words)
            if spec.kind == SHORTENED:
                cleaned = row["query"]  # never touch the baseline text
            row["raw_output"] = raw
            row["rewrite"] = cleaned
            row["n_words"] = len(cleaned.split())
            row["demo_copy"] = round(demo_copy_ratio(cleaned, row["demos_used"]), 4)

        path = out_dir / f"{spec.name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        words = np.array([r["n_words"] for r in rows])
        empty = int((words == 0).sum())
        print(
            f"[rewrite] {spec.name:28s} {time.time() - t0:7.1f}s  "
            f"words med={np.median(words):.0f} p95={np.percentile(words, 95):.0f} "
            f"empty={empty}  -> {path}",
            flush=True,
        )

    meta = {
        "backend": args.backend,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_words": max_words,
        "n_samples": len(samples),
        "prompt_type": args.prompt_type,
        "users_from": args.users_from,
        "sample_users": args.sample_users,
        "sample_seed": args.sample_seed,
        "limit": args.limit,
        "methods": [s.name for s in specs],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


class _Sample:
    __slots__ = ("user_id", "sample_id", "query", "prompt")

    def __init__(self, user_id, sample_id, query, prompt):
        self.user_id, self.sample_id, self.query, self.prompt = user_id, sample_id, query, prompt


def _as_sample(rec, test) -> _Sample:
    return _Sample(rec["user_id"], test["sample_id"], test["query"], test["prompt"])


def _run_llm(llm, inputs, label: str) -> list[str]:
    chunk = max(1, getattr(llm, "batch_size", 32)) * 8
    out: list[str] = []
    t0 = time.time()
    for start in range(0, len(inputs), chunk):
        out.extend(llm.rewrite_batch(inputs[start : start + chunk]))
        done = min(start + chunk, len(inputs))
        rate = done / max(time.time() - t0, 1e-9)
        eta = (len(inputs) - done) / max(rate, 1e-9)
        print(f"    {label}: {done}/{len(inputs)} ({rate:.1f}/s, eta {eta / 60:.1f}m)", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
