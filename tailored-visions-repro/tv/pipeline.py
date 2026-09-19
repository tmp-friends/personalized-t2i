"""Method definitions and rewriter-input assembly.

A *method* is one row of the paper's tables: a retriever, a number of retrieved
history prompts, and a number of in-context demonstrations. This module turns a
:class:`MethodSpec` plus a user's history into the exact string handed to the
rewriter, and nothing else -- generation and scoring live elsewhere so the
stages can run in separate processes (the rewriter and SD v1-5 do not fit in
24 GB together).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

import numpy as np

from . import prompt_templates as PT
from .data import TestSample

# Method kinds
SHORTENED = "shortened"  # no rewriting at all
PROMPTIST = "promptist"
GENERAL_PR = "general_pr"  # LLM rewrite, no user history
PERSONALIZED = "personalized"  # the paper's method


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str = PERSONALIZED
    retriever: str | None = "ebr"
    num_retrieval: int = 3
    icl_shot: int = 0

    @property
    def uses_llm(self) -> bool:
        return self.kind in (GENERAL_PR, PERSONALIZED)

    @property
    def uses_retrieval(self) -> bool:
        return self.kind == PERSONALIZED

    def describe(self) -> str:
        if self.kind == SHORTENED:
            return "Shortened Prompt (no rewriting)"
        if self.kind == PROMPTIST:
            return "Promptist"
        if self.kind == GENERAL_PR:
            return "General PR (LLM rewrite, no user history)"
        icl = f"{self.icl_shot}-shot ICL" if self.icl_shot else "context-independent"
        return f"Personalized PR ({self.retriever.upper()}, top-{self.num_retrieval}, {icl})"


def table2_methods(retrievers: Sequence[str] = ("bm25", "ebr")) -> list[MethodSpec]:
    """The seven rows of Table 2."""
    methods = [
        MethodSpec("shortened_prompt", kind=SHORTENED, retriever=None),
        MethodSpec("promptist", kind=PROMPTIST, retriever=None),
        MethodSpec("general_pr", kind=GENERAL_PR, retriever=None),
    ]
    for r in retrievers:
        methods.append(MethodSpec(f"personalized_pr_{r}", retriever=r, num_retrieval=3, icl_shot=0))
    for r in retrievers:
        methods.append(
            MethodSpec(f"personalized_pr_icl_{r}", retriever=r, num_retrieval=3, icl_shot=1)
        )
    return methods


def topk_ablation_methods(ks: Sequence[int] = (1, 3, 5, 7)) -> list[MethodSpec]:
    """Table 4: EBR + 1-shot ICL, varying the number of retrieved prompts."""
    return [
        MethodSpec(f"ablation_topk{k}", retriever="ebr", num_retrieval=k, icl_shot=1) for k in ks
    ]


def icl_ablation_methods(
    shots: Sequence[int] = (1, 3, 5), retrievers: Sequence[str] = ("bm25", "ebr")
) -> list[MethodSpec]:
    """Table 5: top-3 retrieval, varying the number of demonstrations."""
    return [
        MethodSpec(f"ablation_icl{s}_{r}", retriever=r, num_retrieval=3, icl_shot=s)
        for r in retrievers
        for s in shots
    ]


@dataclass
class DemoRanker:
    """Orders the five hand-written demonstrations by similarity to the query.

    The paper arranges demonstrations in descending order of proximity to the
    input prompt, scoring each demonstration by its *query* field. The reference
    uses CLIP ViT-B/32 for this; we reuse whichever text encoder is passed in and
    record the choice in the run metadata.
    """

    demo_query_embs: np.ndarray
    examples: Sequence[Sequence[str]] = field(default_factory=lambda: PT.EXAMPLES)

    @classmethod
    def build(cls, encode_fn, examples: Sequence[Sequence[str]] = PT.EXAMPLES) -> "DemoRanker":
        queries = [ex[-2] for ex in examples]
        return cls(demo_query_embs=np.asarray(encode_fn(queries)), examples=examples)

    def rank(self, query_emb: np.ndarray) -> list[Sequence[str]]:
        return PT.rank_demos(query_emb, self.demo_query_embs, self.examples)


def build_rewriter_input(
    spec: MethodSpec,
    sample: TestSample,
    retrieved_prompts: Sequence[str] = (),
    demos: Sequence[Sequence[str]] = (),
) -> str:
    """Assemble the string fed to the rewriter for one test sample."""
    if spec.kind == GENERAL_PR:
        return PT.GENERAL_PR_TEMPLATE.format(sample.query)
    if spec.kind != PERSONALIZED:
        # Shortened / Promptist take the bare query.
        return sample.query
    template = PT.build_template(spec.num_retrieval, demos=demos[: spec.icl_shot])
    return PT.fill_template(template, retrieved_prompts, sample.query)


def demo_copy_ratio(rewrite: str, demo_golds: Sequence[str], min_run: int = 4) -> float:
    """How much of the rewrite is a contiguous copy of a demonstration's answer.

    A small model given in-context examples sometimes emits a demonstration's
    answer instead of rewriting the query -- the smoke test caught one for the
    query "assassins". That is a property of the method with this rewriter, not
    a bug to patch, but it has to be measured: a copied demonstration has
    nothing to do with the user.

    Bag-of-words overlap will not do here. The demonstrations are dense with
    generic prompt vocabulary ("best quality", "masterpiece", "trending in
    artstation", articles, prepositions), so an unrelated rewrite still scores
    0.25-0.35 on stopwords alone and real copying disappears into that floor.
    This uses ``difflib.SequenceMatcher`` over token sequences instead, counting
    only runs of at least ``min_run`` consecutive tokens -- shorter matches are
    still mostly "with a", "the best quality" and friends. Report the *rate*
    above a threshold rather than the mean: the underlying distribution is
    bimodal (copied / not), and a mean over it describes neither mode.
    """
    if not rewrite or not demo_golds:
        return 0.0
    tokens = rewrite.lower().split()
    if not tokens:
        return 0.0
    best = 0.0
    for gold_text in demo_golds:
        gold = gold_text.lower().split()
        if not gold:
            continue
        matcher = SequenceMatcher(None, tokens, gold, autojunk=False)
        matched = sum(b.size for b in matcher.get_matching_blocks() if b.size >= min_run)
        best = max(best, matched / len(tokens))
    return best


DEMO_COPY_THRESHOLD = 0.6
