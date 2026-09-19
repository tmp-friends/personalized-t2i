#!/usr/bin/env python
"""End-to-end demo for a single user: retrieve -> rewrite -> generate.

Shows what the method actually changes, by putting the image from the raw query
next to the image from the personalized rewrite.

    python scripts/demo.py --user 87403 --prompt "a cat"
    python scripts/demo.py --prompt "a cat" --no-t2i     # rewriting only

With no ``--user``, picks the user whose history is most relevant to the prompt,
which makes the personalization visible instead of arbitrary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tailored_visions_repro import prompt_templates as PT  # noqa: E402
from tailored_visions_repro.pipeline import PERSONALIZED, MethodSpec, build_rewriter_input  # noqa: E402
from tailored_visions_repro.retrieval import BM25Retriever, EBRRetriever  # noqa: E402
from tailored_visions_repro.rewriter import build_rewriter, clean_output  # noqa: E402


class _Sample:
    def __init__(self, query):
        self.user_id = self.sample_id = "demo"
        self.query = self.prompt = query


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a cat")
    ap.add_argument("--user", default="")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--retriever", default="ebr", choices=["ebr", "bm25"])
    ap.add_argument("--num-retrieval", type=int, default=3)
    ap.add_argument("--icl-shot", type=int, default=1)
    ap.add_argument("--backend", default="local", choices=["local", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-t2i", dest="t2i", action="store_false")
    ap.add_argument("--out-dir", default="outputs/demo")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    records = [json.loads(l) for l in open(cache / "history.jsonl", encoding="utf-8")]
    by_id = {r["user_id"]: r for r in records}

    ebr = EBRRetriever()
    query_emb = ebr.encode([args.prompt])[0]

    if args.user:
        if args.user not in by_id:
            raise SystemExit(f"user {args.user} not in {cache / 'history.jsonl'}")
        rec = by_id[args.user]
    else:
        index = json.loads((cache / "ebr_index.json").read_text(encoding="utf-8"))
        embs = np.load(cache / "ebr_embeddings.npz")["history"].astype(np.float32)
        best, best_score = None, -1e9
        for uid, (lo, hi) in index["offsets"].items():
            if hi - lo < 8:
                continue
            score = float((embs[lo:hi] @ query_emb).max())
            if score > best_score:
                best, best_score = uid, score
        rec = by_id[best]
        print(f"[demo] auto-selected user {best} (max history similarity {best_score:.3f})")

    history = rec["history"]
    if args.retriever == "bm25":
        retrieved = BM25Retriever().retrieve(history, args.prompt, k=args.num_retrieval).prompts
    else:
        h = ebr.encode(history)
        order = np.argsort(-(h @ query_emb))[: args.num_retrieval]
        retrieved = [history[i] for i in order]

    demos = PT.rank_demos(query_emb, ebr.encode([ex[-2] for ex in PT.EXAMPLES]))
    spec = MethodSpec(
        "demo", kind=PERSONALIZED, retriever=args.retriever,
        num_retrieval=args.num_retrieval, icl_shot=args.icl_shot,
    )
    rewriter_input = build_rewriter_input(spec, _Sample(args.prompt), retrieved, demos)

    kwargs = dict(batch_size=1, temperature=0.0)
    if args.backend == "local":
        kwargs["max_new_tokens"] = 256
    if args.model:
        kwargs["model_name"] = args.model
    llm = build_rewriter(args.backend, **kwargs)
    new_prompt = clean_output(llm.rewrite(rewriter_input))

    print(f"\n{'=' * 72}\nUser            : {rec['user_id']} ({rec['n_dedup']} distinct prompts)")
    print(f"Original prompt : {args.prompt}")
    print(f"\nRetrieved history ({spec.retriever.upper()}, top-{args.num_retrieval}):")
    for i, p in enumerate(retrieved, 1):
        print(f"  {i}. {p[:160]}")
    print(f"\nRewritten prompt: {new_prompt}\n{'=' * 72}")

    if args.t2i:
        from tailored_visions_repro.generate import SDGenerator, seed_for

        if args.backend == "local" and hasattr(llm, "unload"):
            llm.unload()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sd = SDGenerator()
        seed = seed_for(f"demo/{rec['user_id']}/{args.prompt}")
        for label, text in (("original", args.prompt), ("personalized", new_prompt)):
            sd.generate([text], [seed])[0].save(out_dir / f"{label}.png")
            print(f"[demo] wrote {out_dir / f'{label}.png'}")
        (out_dir / "demo.json").write_text(
            json.dumps(
                {
                    "user_id": rec["user_id"],
                    "original_prompt": args.prompt,
                    "retrieved": retrieved,
                    "rewritten_prompt": new_prompt,
                    "seed": seed,
                    "rewriter_input": rewriter_input,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
