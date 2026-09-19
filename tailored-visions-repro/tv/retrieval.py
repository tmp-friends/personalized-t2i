"""Retrievers: pick the top-k most relevant history prompts for a query.

Two variants, matching the paper:

* ``bm25``  -- sparse lexical retrieval.
* ``ebr``   -- dense embedding-based retrieval over CLIP ViT-L/14 *text*
               embeddings (the paper's "EBR").

Both operate on a user's deduplicated history. The dedup step is not incidental:
raw PIP histories contain long runs of byte-identical prompts (a user hitting
"generate" repeatedly), and without it top-3 retrieval routinely returns three
copies of one prompt, leaving the rewriter with no preference signal.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for pkg, path in (("punkt_tab", "tokenizers/punkt_tab"), ("punkt", "tokenizers/punkt")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # offline: word_tokenize falls back below
                pass
    _NLTK_READY = True


def tokenize(text: str) -> list[str]:
    """Word tokenization, matching the reference implementation's ``word_tokenize``."""
    _ensure_nltk()
    try:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text.lower())
    except Exception:
        return text.lower().split()


# --------------------------------------------------------------------------
# History deduplication (reference: language.checklines + language.no_duplicate)
# --------------------------------------------------------------------------

def _bleu4(ref_tokens: Sequence[Sequence[str]], hyp_tokens: Sequence[str]) -> float:
    """nltk ``sentence_bleu`` with default weights, over pre-tokenized references.

    Pre-tokenizing the growing reference list turns the reference implementation's
    O(n^2) re-splitting into O(n) per candidate, which matters for the long-tail
    users (up to 1316 unique prompts).
    """
    import warnings

    from nltk.translate.bleu_score import sentence_bleu

    if not hyp_tokens:
        return 0.0
    with warnings.catch_warnings():
        # "hypothesis contains 0 counts of n-gram overlaps" fires on nearly every
        # short prompt; a BLEU of 0 is the correct answer there, not a problem.
        warnings.simplefilter("ignore", UserWarning)
        return float(sentence_bleu(list(ref_tokens), list(hyp_tokens)))


def dedup_history(
    prompts: Sequence[str],
    threshold: float = 0.5,
    max_history: int | None = None,
) -> list[str]:
    """Drop prompts that are near-duplicates of ones already kept.

    A prompt is kept when ``BLEU-4(kept_so_far, prompt) < threshold``, the rule
    used by the reference ``checklines``. Exact duplicates are removed first
    (order-preserving) so the expensive BLEU pass only sees distinct strings.

    ``max_history`` caps how many distinct prompts enter the BLEU pass, keeping
    the most recent ones; ``None`` means no cap (faithful to the paper).
    """
    seen: dict[str, None] = {}
    for p in prompts:
        if isinstance(p, str) and p.strip():
            seen.setdefault(p, None)
    distinct = list(seen)
    if max_history is not None and len(distinct) > max_history:
        distinct = distinct[-max_history:]

    kept: list[str] = []
    kept_tokens: list[list[str]] = []
    for prompt in distinct:
        hyp = prompt.split()
        if not kept:
            kept.append(prompt)
            kept_tokens.append(hyp)
            continue
        if _bleu4(kept_tokens, hyp) < threshold:
            kept.append(prompt)
            kept_tokens.append(hyp)
    return kept


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

class BM25Model:
    """BM25 with the reference implementation's hyperparameters (k1=2, k2=1, b=0.5)."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 2, k2: float = 1, b: float = 0.5):
        self.documents = [list(d) for d in documents]
        self.n_docs = len(self.documents)
        self.avg_len = (
            sum(len(d) for d in self.documents) / self.n_docs if self.n_docs else 0.0
        )
        self.k1, self.k2, self.b = k1, k2, b
        self.tf: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        df: dict[str, int] = {}
        for doc in self.documents:
            counts = Counter(doc)
            self.tf.append(counts)
            for term in counts:
                df[term] = df.get(term, 0) + 1
        for term, freq in df.items():
            self.idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, index: int, query_terms: Sequence[str]) -> float:
        # NOTE: the reference uses len(self.f[index]) -- the number of *distinct*
        # terms -- as the document length. Kept as-is for faithfulness.
        doc_len = len(self.tf[index])
        qf = Counter(query_terms)
        total = 0.0
        for term in query_terms:
            freq = self.tf[index].get(term)
            if not freq:
                continue
            total += (
                self.idf[term]
                * (freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)))
                * (qf[term] * (self.k2 + 1) / (qf[term] + self.k2))
            )
        return total

    def scores(self, query_terms: Sequence[str]) -> list[float]:
        return [self.score(i, query_terms) for i in range(self.n_docs)]


@dataclass
class Retrieved:
    prompts: list[str]
    scores: list[float]


class BM25Retriever:
    """Sparse retriever.

    ``legacy_char_query=True`` reproduces a bug in the released reference code:
    ``bm25_retrieval`` tokenizes the documents with ``word_tokenize`` but passes
    the *raw query string* to the scorer, so ``for q in query`` iterates over
    characters and BM25 degenerates into single-character matching. The default
    (``False``) tokenizes the query the same way as the documents.
    """

    name = "bm25"

    def __init__(self, legacy_char_query: bool = False):
        self.legacy_char_query = legacy_char_query

    def retrieve(self, history: Sequence[str], query: str, k: int = 3) -> Retrieved:
        if not history:
            return Retrieved([], [])
        docs = [tokenize(doc) for doc in history]
        bm25 = BM25Model(docs)
        query_terms = list(query.lower()) if self.legacy_char_query else tokenize(query)
        scores = np.asarray(bm25.scores(query_terms), dtype=np.float64)
        k = min(k, len(history))
        # Stable ordering: descending score, ties broken by original position.
        order = np.lexsort((np.arange(len(history)), -scores))[:k]
        return Retrieved([history[i] for i in order], [float(scores[i]) for i in order])


# --------------------------------------------------------------------------
# EBR (dense, CLIP text embeddings)
# --------------------------------------------------------------------------

class EBRRetriever:
    """Dense retriever over CLIP ViT-L/14 text embeddings.

    The reference calls ``clip.load("ViT-L/14")`` from the OpenAI ``clip``
    package; we use the identical weights through ``transformers``
    (``openai/clip-vit-large-patch14``), which installs cleanly on modern
    Python. Prompts are truncated to CLIP's 77-token context, as in the
    reference (``clip.tokenize(..., truncate=True)``).
    """

    name = "ebr"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | None = None,
        dtype: str = "float32",
        batch_size: int = 256,
    ):
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = (
            CLIPTextModelWithProjection.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i : i + self.batch_size])
                enc = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(self.device)
                emb = self.model(**enc).text_embeds
                emb = emb / emb.norm(dim=-1, keepdim=True)
                out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)

    def retrieve(self, history: Sequence[str], query: str, k: int = 3) -> Retrieved:
        if not history:
            return Retrieved([], [])
        embs = self.encode([query, *history])
        query_emb, hist_embs = embs[0], embs[1:]
        scores = hist_embs @ query_emb
        k = min(k, len(history))
        order = np.lexsort((np.arange(len(history)), -scores))[:k]
        return Retrieved([history[i] for i in order], [float(scores[i]) for i in order])

    def retrieve_precomputed(
        self,
        history: Sequence[str],
        hist_embs: np.ndarray,
        query_emb: np.ndarray,
        k: int = 3,
    ) -> Retrieved:
        """Same ranking, but reusing embeddings computed once per user."""
        scores = hist_embs @ query_emb
        k = min(k, len(history))
        order = np.lexsort((np.arange(len(history)), -scores))[:k]
        return Retrieved([history[i] for i in order], [float(scores[i]) for i in order])


def build_retriever(name: str, **kwargs):
    name = name.lower()
    if name == "bm25":
        return BM25Retriever(**kwargs)
    if name in ("ebr", "dense", "clip"):
        return EBRRetriever(**kwargs)
    raise ValueError(f"unknown retriever: {name!r}")
