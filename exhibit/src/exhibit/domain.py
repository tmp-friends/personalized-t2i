"""Auditable preference context shared verbatim by rewriting and comparison."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

AXES = {
    "color": {
        "label": "色づかい",
        "values": {
            "warm": ("暖かな色", "warm accent colors"),
            "cool": ("涼しげな色", "cool accent colors"),
            "muted": ("落ち着いた色", "muted colors"),
            "vivid": ("鮮やかな色", "vivid colors"),
        },
    },
    "lighting": {
        "label": "光",
        "values": {
            "soft": ("柔らかな光", "soft lighting"),
            "dramatic": ("印象的な陰影", "dramatic contrast"),
        },
    },
    "composition": {
        "label": "構図",
        "values": {
            "spacious": ("余白を楽しむ", "spacious composition"),
            "close": ("被写体に近づく", "close framing"),
        },
    },
    "texture": {
        "label": "描画表現",
        "values": {
            "painterly": ("絵画のような表現", "painterly texture"),
            "photographic": ("写真のような表現", "photographic detail"),
        },
    },
    "mood": {
        "label": "雰囲気",
        "values": {
            "calm": ("静かで穏やか", "calm atmosphere"),
            "lively": ("活気のある雰囲気", "lively atmosphere"),
        },
    },
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def build_persona(choices, evidence):
    chosen = {
        (c["pair_id"], c["chosen_id"])
        for c in choices
        if c.get("chosen_id") is not None
    }
    fragments = [
        e
        for e in evidence
        if e.get("reviewed") and (e["pair_id"], e["chosen_id"]) in chosen
    ]
    axes = {}
    for name in AXES:
        relevant = [
            e
            for e in fragments
            if name in e["dimensions"]
            and e.get("values", {}).get(name) in AXES[name]["values"]
        ]
        votes = Counter(e["values"][name] for e in relevant)
        # Conflicting observations remain unknown, rather than a forced majority claim.
        value = next(iter(votes)) if len(votes) == 1 else None
        axes[name] = {
            "value": value,
            "count": len({e["pair_id"] for e in relevant}),
            "evidence_ids": [e["id"] for e in relevant],
            "conflict": len(votes) > 1,
        }
    return {
        "axes": axes,
        "evidence": fragments,
        "choice_ids": sorted(f"{p}:{c}" for p, c in chosen),
        "method": "reviewed-evidence-summary",
    }


def effective_context(persona, edits):
    for axis, value in edits.items():
        if axis not in AXES or (
            value is not None and value not in AXES[axis]["values"]
        ):
            raise ValueError("Unknown preference value")
    preferences = {k: edits.get(k, v["value"]) for k, v in persona["axes"].items()}
    preferences = {k: v for k, v in preferences.items() if v is not None}
    # Even an unchanged explicit override takes precedence over inference.
    removed = set(edits) | (set(AXES) - set(preferences))
    allowed_fields = {
        "id",
        "pair_id",
        "pair_version",
        "chosen_id",
        "dimensions",
        "values",
        "text",
        "source",
        "image_hashes",
        "basic_prompt_en",
        "template_version",
        "template_hash",
        "preprocessing",
        "context_source",
    }
    fragments = [
        {key: value for key, value in e.items() if key in allowed_fields}
        for e in persona["evidence"]
        if not (set(e["dimensions"]) & removed)
    ]
    overrides = {k: v for k, v in edits.items() if v is not None}
    lines = [e["text"] for e in fragments]
    lines += [
        f"User explicitly requests {AXES[k]['values'][v][1]}."
        for k, v in overrides.items()
    ]
    # A surviving inference may have lost a multi-axis fragment; do not resurrect it.
    supported = {a for e in fragments for a in e["dimensions"]} | set(overrides)
    preferences = {k: v for k, v in preferences.items() if k in supported}
    context = {
        "version": 1,
        "preferences": preferences,
        "overrides": overrides,
        "evidence": fragments,
        "choice_ids": persona["choice_ids"],
        "text": "\n".join(lines),
    }
    return {**context, "hash": digest(context)}


def cache_key(prompt, settings, seed, context=None):
    return digest(
        {"prompt": prompt, "settings": settings, "seed": seed, "context": context}
    )


def validate_prompt(prompt, topic, tokenizers):
    if not isinstance(prompt, str) or not prompt.startswith(topic["basic_prompt_en"]):
        return False
    if any(c in prompt for c in ["<", ">", "\n"]):
        return False
    return all(
        len(t(prompt, truncation=False)["input_ids"]) <= t.model_max_length
        for t in tokenizers
    )
