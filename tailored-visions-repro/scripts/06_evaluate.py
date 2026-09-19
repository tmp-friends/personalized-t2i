#!/usr/bin/env python
"""Stage 6: score every method and render the paper's tables.

ROUGE-L is computed over whatever ``02_rewrite.py`` produced (full test set by
default), and reported twice: over everything, and over the samples
``07_leakage.py`` did *not* flag as already present in the user's history. The
first is what compares to the paper; the second is what is actually earned.

PMS and Image-Align (proxy) are computed over the subset ``05_generate.py``
embedded, so they carry a different ``n``, printed per row rather than hidden.

    python scripts/06_evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv.metrics import CLIPScorer, image_align, pms, rouge_l, summarize  # noqa: E402
from tv.pipeline import DEMO_COPY_THRESHOLD, demo_copy_ratio  # noqa: E402

GT_PROXY = "gt_proxy"
BASELINE = "shortened_prompt"

# Paper Tables 2 / 4 / 5, for side-by-side reference only. Produced with
# gpt-3.5-turbo as the rewriter and against real user images. Read
# docs/DEVIATIONS.md before treating any gap as a reproduction failure -- in
# particular, the model-free "Shortened Prompt" row does *not* reproduce, so the
# absolute ROUGE-L scale differs between the two columns.
PAPER = {
    "shortened_prompt": (0.5567, 0.6272, 0.3268),
    "promptist": (0.5858, 0.6481, 0.2947),
    "general_pr": (0.5996, 0.5912, 0.2082),
    "personalized_pr_bm25": (0.6125, 0.6581, 0.3942),
    "personalized_pr_ebr": (0.6083, 0.6485, 0.4137),
    "personalized_pr_icl_bm25": (0.6253, 0.6456, 0.4417),
    "personalized_pr_icl_ebr": (0.6179, 0.6796, 0.4686),
    "ablation_topk1": (0.6057, 0.6751, 0.4539),
    "ablation_topk3": (0.6179, 0.6796, 0.4686),
    "ablation_topk5": (0.6204, 0.6748, 0.4474),
    "ablation_topk7": (0.6265, 0.6651, 0.4592),
    "ablation_icl1_bm25": (0.6253, 0.6456, 0.4417),
    "ablation_icl3_bm25": (0.6289, 0.6580, 0.4381),
    "ablation_icl5_bm25": (0.6236, 0.6571, 0.4226),
    "ablation_icl1_ebr": (0.6179, 0.6796, 0.4686),
    "ablation_icl3_ebr": (0.6274, 0.6708, 0.4354),
    "ablation_icl5_ebr": (0.6242, 0.6724, 0.4439),
}

DISPLAY = {
    "shortened_prompt": ("Shortened Prompt", "-"),
    "promptist": ("Promptist", "-"),
    "general_pr": ("General PR", "-"),
    "personalized_pr_bm25": ("Personalized PR", "BM25"),
    "personalized_pr_ebr": ("Personalized PR", "EBR"),
    "personalized_pr_icl_bm25": ("Personalized PR + ICL", "BM25"),
    "personalized_pr_icl_ebr": ("Personalized PR + ICL", "EBR"),
}


def load_preferences(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("preference"):
            out[rec["user_id"]] = rec["preference"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites-dir", default="results/rewrites")
    ap.add_argument("--embeddings-dir", default="results/embeddings")
    ap.add_argument("--preferences", default="data/cache/preferences.jsonl")
    ap.add_argument("--leakage", default="results/leakage.json")
    ap.add_argument("--out-json", default="results/metrics.json")
    ap.add_argument("--out-md", default="results/RESULTS.md")
    args = ap.parse_args()

    rewrites_dir = Path(args.rewrites_dir)
    emb_dir = Path(args.embeddings_dir)
    method_files = sorted(rewrites_dir.glob("*.jsonl"))
    if not method_files:
        raise SystemExit(f"no rewrites in {rewrites_dir}")

    leaked: set[str] = set()
    if Path(args.leakage).exists():
        leaked = set(json.loads(Path(args.leakage).read_text(encoding="utf-8"))["leaked_keys"])
        print(f"[evaluate] {len(leaked)} samples flagged as leaked; reporting both columns")

    preferences = load_preferences(Path(args.preferences))
    pref_embs: dict[str, np.ndarray] = {}
    gt_embs: dict[str, np.ndarray] = {}
    if (emb_dir / f"{GT_PROXY}.npz").exists():
        gt = np.load(emb_dir / f"{GT_PROXY}.npz")
        gt_embs = dict(zip(gt["keys"].tolist(), gt["embeddings"]))
    if preferences and gt_embs:
        print(f"[evaluate] embedding {len(preferences)} preference summaries ...")
        scorer = CLIPScorer()
        uids = sorted(preferences)
        pref_embs = dict(zip(uids, scorer.encode_text([preferences[u] for u in uids])))
        scorer.unload()

    results = {}
    for path in method_files:
        name = path.stem
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        clean = [r for r in rows if r["key"] not in leaked]

        # Recomputed rather than read from the file: the stored column may have
        # been written by an older, bag-of-words version of the measure.
        copy_scores = [demo_copy_ratio(r["rewrite"], r.get("demos_used", [])) for r in rows]
        entry = {
            "n_rewrites": len(rows),
            "n_non_leaked": len(clean),
            "rouge_l": summarize(
                rouge_l([r["rewrite"] for r in rows], [r["gt_prompt"] for r in rows])
            ),
            "rouge_l_non_leaked": summarize(
                rouge_l([r["rewrite"] for r in clean], [r["gt_prompt"] for r in clean])
            )
            if clean
            else None,
            "rouge_l_single_segment": summarize(
                rouge_l(
                    [r["rewrite"] for r in rows],
                    [r["gt_prompt"] for r in rows],
                    split_sentences=False,
                )
            ),
            "words": summarize([r["n_words"] for r in rows]),
            "empty_rate": float(np.mean([r["n_words"] == 0 for r in rows])) if rows else 0.0,
            "demo_copy_rate": float(np.mean([c >= DEMO_COPY_THRESHOLD for c in copy_scores])),
            "demo_copy_threshold": DEMO_COPY_THRESHOLD,
        }

        emb_path = emb_dir / f"{name}.npz"
        if emb_path.exists():
            data = np.load(emb_path)
            by_key = {r["key"]: r for r in rows}
            pms_vals, ia_vals = [], []
            for key, emb in zip(data["keys"].tolist(), data["embeddings"]):
                row = by_key.get(key)
                if row is None:
                    continue
                pref = pref_embs.get(row["user_id"])
                if pref is not None:
                    pms_vals.append(float(pms(emb[None, :], pref[None, :])[0]))
                gt = gt_embs.get(key)
                if gt is not None:
                    ia_vals.append(float(image_align(emb[None, :], gt[None, :])[0]))
            entry["pms"] = summarize(pms_vals)
            entry["image_align_proxy"] = summarize(ia_vals)
        results[name] = entry

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = render_markdown(results)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[evaluate] wrote {out_json} and {args.out_md}")
    return 0


def _fmt(stat: dict | None, digits: int = 4) -> str:
    if not stat or stat.get("n", 0) == 0 or np.isnan(stat.get("mean", float("nan"))):
        return "--"
    return f"{stat['mean']:.{digits}f}"


def _delta(stat: dict | None, base: dict | None) -> str:
    if not stat or not base or stat.get("n", 0) == 0 or base.get("n", 0) == 0:
        return "--"
    return f"{stat['mean'] - base['mean']:+.4f}"


def render_markdown(results: dict) -> list[str]:
    base = results.get(BASELINE, {})
    paper_base = PAPER[BASELINE]
    lines = [
        "# Results",
        "",
        "Generated by `scripts/06_evaluate.py`. Read `docs/DEVIATIONS.md` first --",
        "the rewriter is a different model from the paper's, the Image-Align column is a",
        "proxy, and the absolute ROUGE-L scale does not line up with the paper's (below).",
        "",
        "## Table 2 -- comparison with baselines",
        "",
        "| Method | Retriever | PMS | Image-Align (proxy) | ROUGE-L | ROUGE-L non-leaked | n (text/img) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, (label, retriever) in DISPLAY.items():
        r = results.get(name)
        if not r:
            continue
        n_img = (r.get("pms") or {}).get("n", 0) or (r.get("image_align_proxy") or {}).get("n", 0)
        lines.append(
            f"| {label} | {retriever} | {_fmt(r.get('pms'))} | {_fmt(r.get('image_align_proxy'))} | "
            f"{_fmt(r.get('rouge_l'))} | {_fmt(r.get('rouge_l_non_leaked'))} | "
            f"{r['n_rewrites']}/{n_img} |"
        )

    lines += [
        "",
        "### Against the paper, as a difference from the no-op baseline",
        "",
        "The `Shortened Prompt` row involves no model at all -- it is the untouched query,",
        "so it depends only on the dataset and the metric configuration, and it *should*",
        f"reproduce exactly. It does not ({_fmt(base.get('rouge_l'))} here vs"
        f" {paper_base[2]:.4f} in the paper), which means some preprocessing step behind the",
        "paper's ROUGE-L is not recoverable from the released artifacts. Absolute values",
        "are therefore not comparable across the two columns; the movement each method",
        "produces *relative to that shared baseline* is.",
        "",
        "| Method | ΔPMS | ΔPMS (paper) | ΔROUGE-L | ΔROUGE-L (paper) |",
        "|---|---|---|---|---|",
    ]
    for name, (label, retriever) in DISPLAY.items():
        r = results.get(name)
        if not r:
            continue
        paper = PAPER.get(name)
        tag = label if retriever == "-" else f"{label} ({retriever})"
        lines.append(
            f"| {tag} | {_delta(r.get('pms'), base.get('pms'))} | "
            f"{paper[0] - paper_base[0]:+.4f} | "
            f"{_delta(r.get('rouge_l'), base.get('rouge_l'))} | "
            f"{paper[2] - paper_base[2]:+.4f} |"
        )

    ablations = [n for n in results if n.startswith("ablation_")]
    if ablations:
        lines += [
            "",
            "## Tables 4 & 5 -- ablations",
            "",
            "Ablation rows may be scored on a different sample of users than Table 2",
            "(see `run_meta.json`); they are comparable to each other, not to the table above.",
            "",
            "| Setting | PMS | Image-Align (proxy) | ROUGE-L | ROUGE-L (paper) | n |",
            "|---|---|---|---|---|---|",
        ]
        for name in sorted(ablations):
            r = results[name]
            paper = PAPER.get(name, (None, None, None))
            lines.append(
                f"| {name} | {_fmt(r.get('pms'))} | {_fmt(r.get('image_align_proxy'))} | "
                f"{_fmt(r.get('rouge_l'))} | "
                f"{f'{paper[2]:.4f}' if paper[2] is not None else '--'} | {r['n_rewrites']} |"
            )

    lines += [
        "",
        "## Diagnostics",
        "",
        "`demo copy` is the share of rewrites that are largely a contiguous copy of the",
        f"in-context demonstration's answer (overlap >= {DEMO_COPY_THRESHOLD}) rather than a rewrite of the",
        "query -- a failure mode of small local rewriters that the paper's ChatGPT would",
        "hit less often. `words` matters because ROUGE-L uses beta=5, so longer outputs",
        "buy recall.",
        "",
        "`ROUGE-L 1-seg` scores each prompt as a single segment instead of splitting on",
        "\".\". The reference implementation splits, which quietly favours long",
        "multi-sentence rewrites; the gap between the two columns is how much of a",
        "method's ROUGE-L comes from that.",
        "",
        "| Method | ROUGE-L (±sem) | non-leaked (±sem) | ROUGE-L 1-seg | words | empty | demo copy |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(results):
        r = results[name]
        rl, nl = r["rouge_l"], r.get("rouge_l_non_leaked")
        nl_txt = f"{nl['mean']:.4f} ±{nl['sem']:.4f}" if nl else "--"
        lines.append(
            f"| {name} | {rl['mean']:.4f} ±{rl['sem']:.4f} | {nl_txt} | "
            f"{_fmt(r.get('rouge_l_single_segment'))} | "
            f"{r['words']['mean']:.1f} | {r['empty_rate']:.3f} | {r['demo_copy_rate']:.3f} |"
        )
    lines.append("")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
