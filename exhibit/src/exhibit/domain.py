"""Cards, reference texts and the personalization identity shared by every stage."""

from __future__ import annotations

import hashlib
import json

from .config import CARDS_REVIEW, CONFIG, read_json

ASPECTS = ("color", "lighting", "texture", "mood")


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


def compose_prompt(basic_prompt_en, phrases, generation):
    parts = [basic_prompt_en, *phrases, generation["positive_prompt_tail"]]
    normalized = [part.strip().rstrip(" ,.\n") for part in parts if part.strip()]
    return ", ".join(normalized) + "."


def build_cards(config=CONFIG):
    """4 subjects x 4 expression profiles. The subject never enters a reference."""
    cards = []
    for subject in config["card_subjects"]:
        for profile in config["card_profiles"]:
            aspects = {key: profile["aspects"][key] for key in ASPECTS}
            card_id = f"{subject['id']}-{profile['id']}"
            cards.append(
                {
                    "id": card_id,
                    "subject_id": subject["id"],
                    "profile_id": profile["id"],
                    "subject_label": subject["label"],
                    "profile_label": profile["label"],
                    "label": f"{subject['label']} · {profile['label']}",
                    "aspects": aspects,
                    "aspects_ja": {key: profile["aspects_ja"][key] for key in ASPECTS},
                    "ref_en": ", ".join(aspects.values()),
                    "prompt": compose_prompt(
                        subject["basic_prompt_en"],
                        list(aspects.values()),
                        config["generation"],
                    ),
                    "seed": subject["seed"],
                    "path": f"cards/{card_id}.png",
                }
            )
    return cards


CARDS = {card["id"]: card for card in build_cards()}


def reviewed_ids(path=CARDS_REVIEW):
    review = read_json(path, {}) or {}
    return {
        card_id
        for card_id in CARDS
        if isinstance(review.get(card_id), dict) and review[card_id].get("reviewed")
    }


def ref_text(card, aspects_off=()):
    """The visitor may drop aspects; the dropped phrases leave the reference."""
    off = set(aspects_off or ())
    if off - set(ASPECTS):
        raise ValueError("Unknown aspect")
    text = ", ".join(
        phrase for key, phrase in card["aspects"].items() if key not in off
    )
    if not text:
        raise ValueError("説明文がすべて外されています。1つ以上残してください。")
    return text


def normalize_selection(entries, config=CONFIG):
    """`[{card_id, aspects_off}]`, order preserved, validated against the catalog."""
    entries = list(entries or [])
    limits = config["selection"]
    if not limits["min"] <= len(entries) <= limits["max"]:
        raise ValueError(f"{limits['min']}〜{limits['max']}枚を選んでください。")
    selection = []
    for entry in entries:
        card_id = entry.get("card_id")
        if card_id not in CARDS:
            raise ValueError("Unknown card")
        if any(x["card_id"] == card_id for x in selection):
            raise ValueError("Duplicate card")
        aspects_off = [a for a in ASPECTS if a in set(entry.get("aspects_off") or ())]
        if set(entry.get("aspects_off") or ()) - set(ASPECTS):
            raise ValueError("Unknown aspect")
        ref_text(CARDS[card_id], aspects_off)
        selection.append({"card_id": card_id, "aspects_off": aspects_off})
    return selection


# Everything that changes the encoding and therefore the identity of a result.
FAN_SETTINGS = ("skip", "sample_size", "skip_pa", "use_attn_mask")


def fan_settings(config=CONFIG):
    return {key: config["fan"][key] for key in FAN_SETTINGS if key in config["fan"]}


def personalization_hash(refs, alpha, *, commit, generation, seeds, fan=None):
    return digest(
        {
            "refs": refs,
            "alpha": alpha,
            "commit": commit,
            "fan": fan or {},
            "generation": generation,
            "seeds": seeds,
        }
    )


def build_personalization(selection, weights, alpha_key, config=CONFIG):
    """One reference per distinct aspect phrase; duplicates merge their weights.

    A single bundled sentence per card was measured to swing composition, while
    short aspect phrases keep the target framing.
    """
    if alpha_key not in config["alphas"]:
        raise ValueError("Unknown alpha")
    alpha = config["alphas"][alpha_key]
    weights = weights or {}
    if set(weights) - {entry["card_id"] for entry in selection}:
        raise ValueError("Unknown card weight")
    merged = {}
    for entry in selection:
        key = weights.get(entry["card_id"], "normal")
        if key not in config["weights"]:
            raise ValueError("Unknown weight")
        weight = float(config["weights"][key])
        if not weight:
            continue
        card = CARDS[entry["card_id"]]
        off = set(entry["aspects_off"])
        for aspect in ASPECTS:
            if aspect in off:
                continue
            phrase = card["aspects"][aspect]
            if phrase in merged:
                merged[phrase]["weight"] += weight
                merged[phrase]["card_ids"].append(card["id"])
            else:
                merged[phrase] = {
                    "text": phrase,
                    "weight": weight,
                    "aspect": aspect,
                    "card_ids": [card["id"]],
                }
    refs = list(merged.values())
    if not refs:
        raise ValueError("参照を1つ以上残してください。")
    return {
        "refs": refs,
        "alpha": alpha,
        "sample_size": config["fan"]["sample_size"],
        "hash": personalization_hash(
            refs,
            alpha,
            commit=config["fan"]["commit"],
            generation=config["generation"],
            seeds=config["seeds"],
            fan=fan_settings(config),
        ),
    }


def target_prompt(topic, config=CONFIG):
    """The same fixed sentence for plain and personalized images."""
    return compose_prompt(topic["basic_prompt_en"], [], config["generation"])


def variant_cache_key(topic_id, personalization, config=CONFIG):
    return digest(
        {
            "topic": topic_id,
            "personalization": personalization["hash"],
            "generation": config["generation"],
            "seeds": config["seeds"],
        }
    )
