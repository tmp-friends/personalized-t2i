"""The paper's three offline metrics.

PMS (Eq. 1)
    ``2.5 * max(cos(E(I'_u), E(P_u)), 0)`` -- CLIPScore between the generated
    image and the user's preference summary ``P_u`` (5 phrases distilled from
    that user's history prompts).

Image-Align
    CLIP image-image cosine similarity between the generated image and the
    user's ground-truth image.

    **The ground-truth images no longer exist.** The PIP ``result_url`` CDN has
    been offline since 2024, so what this module computes is the proxy the
    official README suggests: similarity against an image generated from the
    ground-truth *prompt*, under the same seed. It is a different quantity from
    the paper's and is reported as ``image_align_proxy``. See
    ``docs/DEVIATIONS.md``.

ROUGE-L
    Between the rewritten prompt and the original full prompt, with beta=5 to
    emphasize recall. Google's ``rouge_score`` package hardcodes beta=1, hence
    the vendored ``rougeL``.

Both image metrics use CLIP ViT-B/32, the CLIPScore convention and what the
reference ``metrics.py`` loads -- *not* the ViT-L/14 used by the EBR retriever.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

CLIP_METRIC_MODEL = "openai/clip-vit-base-patch32"
PMS_SCALE = 2.5
ROUGE_BETA = 5.0


class CLIPScorer:
    """Embeds images and text with CLIP ViT-B/32 for PMS and Image-Align."""

    def __init__(
        self,
        model_name: str = CLIP_METRIC_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.processor(
                    text=list(texts[i : i + self.batch_size]),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(self.device)
                feats = self.model.get_text_features(**enc)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)

    def encode_images(self, images: Sequence) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                enc = self.processor(
                    images=list(images[i : i + self.batch_size]), return_tensors="pt"
                ).to(self.device)
                feats = self.model.get_image_features(**enc)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)

    def unload(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


def pms(image_embs: np.ndarray, preference_embs: np.ndarray, scale: float = PMS_SCALE) -> np.ndarray:
    """Per-sample PMS. Both inputs must be L2-normalized and row-aligned."""
    cos = np.sum(image_embs * preference_embs, axis=-1)
    return scale * np.maximum(cos, 0.0)


def image_align(gen_embs: np.ndarray, ref_embs: np.ndarray) -> np.ndarray:
    """Per-sample image-image similarity, clamped at 0 as in the reference."""
    cos = np.sum(gen_embs * ref_embs, axis=-1)
    return np.maximum(cos, 0.0)


_ROUGE = None


def _rouge(beta: float = ROUGE_BETA):
    global _ROUGE
    if _ROUGE is None or getattr(_ROUGE, "beta", None) != beta:
        from .rougeL import Rouge

        _ROUGE = Rouge(metrics=["rouge-l"], beta=beta)
    return _ROUGE


def rouge_l(
    hypotheses: Sequence[str],
    references: Sequence[str],
    beta: float = ROUGE_BETA,
    split_sentences: bool = True,
) -> list[float]:
    """Per-sample ROUGE-L F-score with the given beta.

    ``split_sentences=True`` is the reference implementation's behaviour: it
    splits both strings on "." and scores with summary-level (union-LCS)
    ROUGE-L. That is asymmetric on this data -- rewritten prompts are prose with
    real sentence breaks while the ground-truth prompts are mostly
    comma-separated fragments -- and it is method-correlated: measured here it
    is worth +0.0002 to the 4-word no-op baseline and +0.0163 to a 53-word
    multi-sentence rewrite. Kept as the default because it is what produced the
    paper's numbers; ``split_sentences=False`` scores everything as one segment
    and is reported alongside as a robustness check.

    Empty hypotheses score 0 rather than raising, which is what the reference's
    ``ignore_empty`` path effectively does but without dropping the sample --
    dropping would let a method inflate its mean by failing to produce output.
    """
    rouge = _rouge(beta)
    scores: list[float] = []
    for hyp, ref in zip(hypotheses, references):
        hyp = (hyp or "").strip()
        ref = (ref or "").strip()
        if not split_sentences:
            hyp, ref = hyp.replace(".", " "), ref.replace(".", " ")
        if not hyp or not ref:
            scores.append(0.0)
            continue
        try:
            result = rouge.get_scores([hyp], [ref])
            scores.append(float(result[0]["rouge-l"]["f"]))
        except Exception:
            scores.append(0.0)
    return scores


def token_f1(a: str, b: str) -> float:
    """Unigram-overlap F1 between two strings.

    Used to decide whether a test prompt is effectively already present in a
    user's history or in the retrieved top-k. Cheaper than ROUGE-L and
    symmetric, which is what a "are these the same prompt?" test wants.
    """
    from collections import Counter

    ta, tb = a.lower().split(), b.lower().split()
    if not ta or not tb:
        return 0.0
    overlap = sum((Counter(ta) & Counter(tb)).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r)


def summarize(values: Sequence[float]) -> dict:
    """Mean with a standard error, so table gaps can be read against noise."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sem": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }
