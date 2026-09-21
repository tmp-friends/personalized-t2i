"""Cards, reference texts and the personalization identity shared by every stage."""

from __future__ import annotations

import hashlib
import json
import math

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


# The v1 flow kept its own limits; the multi-round config never widens them.
LEGACY_SELECTION = {"min": 3, "max": 5}


def normalize_selection(entries, config=CONFIG, *, limits=None):
    """`[{card_id, aspects_off}]`, order preserved, validated against the catalog."""
    entries = list(entries or [])
    limits = limits or LEGACY_SELECTION
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
    """The stable legacy cache identity used by current assets."""
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


def legacy_snapshot_from_selection(selection, config=CONFIG):
    """Explicit v1 list conversion; the new public builder never accepts lists."""
    from .catalog import load_catalog

    return {
        "revision": 0,
        "catalog_id": "catalog-v1",
        "catalog_hash": load_catalog("catalog-v1", reviewed_only=False)["catalog_hash"],
        "selection": [
            {
                "card_id": entry["card_id"],
                "strength": 1,
                "aspects": [a for a in ASPECTS if a not in entry["aspects_off"]],
            }
            for entry in normalize_selection(selection, config)
        ],
        "aspect_gains": {aspect: 1 for aspect in ASPECTS},
    }


def build_legacy_personalization(selection, config=CONFIG):
    """The byte-compatible builder retained until the service is migrated."""
    alpha, merged = config["alpha"], {}
    for entry in selection:
        card = CARDS[entry["card_id"]]
        for aspect in ASPECTS:
            if aspect in set(entry["aspects_off"]):
                continue
            phrase = card["aspects"][aspect]
            item = merged.setdefault(
                phrase,
                {"text": phrase, "weight": 0.0, "aspect": aspect, "card_ids": []},
            )
            item["weight"] += 1.0
            item["card_ids"].append(card["id"])
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


def _normal_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Reference text must be non-empty")
    return " ".join(value.split())


def _ref_id(value):
    return hashlib.sha256(_normal_text(value).encode()).hexdigest()


def _finite_json(value, label):
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite values")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _finite_json(item, label)
        return
    raise ValueError(f"{label} must be JSON-shaped")


def valid_strength(value):
    """The two explicit strengths; `True` is not the integer 1 here."""
    return type(value) is int and value in (1, 2)


def valid_gain(value):
    """The three explicit gain steps; bool and non-finite values are not gains."""
    return type(value) in (int, float) and value in (0.5, 1, 2)


def canonical_aspects(aspects, *, allow_empty=False):
    """Order-independent liked aspects; unanswered never becomes every aspect."""
    if (
        not isinstance(aspects, list)
        or (not aspects and not allow_empty)
        or any(not isinstance(aspect, str) for aspect in aspects)
        or set(aspects) - set(ASPECTS)
        or len(aspects) != len(set(aspects))
    ):
        raise ValueError("Invalid aspects")
    return sorted(aspects)


def _snapshot(snapshot, catalog=None):
    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must be a PreferenceSnapshot object")
    needed = {"revision", "catalog_id", "catalog_hash", "selection", "aspect_gains"}
    if set(snapshot) != needed:
        raise ValueError("Snapshot has unknown or missing fields")
    if (
        type(snapshot["revision"]) is not int
        or snapshot["revision"] < 0
        or not all(
            isinstance(snapshot[key], str) and snapshot[key]
            for key in ("catalog_id", "catalog_hash")
        )
    ):
        raise ValueError("Invalid snapshot identity")
    if catalog is None:
        from .catalog import load_catalog

        catalog = load_catalog(snapshot["catalog_id"], reviewed_only=True)
    if (
        snapshot["catalog_id"] != catalog["catalog_id"]
        or snapshot["catalog_hash"] != catalog["catalog_hash"]
    ):
        raise ValueError("Stale catalog hash")
    card_map = {card["id"]: card for card in catalog["cards"]}
    gains = snapshot["aspect_gains"]
    if not isinstance(gains, dict) or set(gains) != set(ASPECTS):
        raise ValueError("aspect_gains must name every aspect")
    if any(not valid_gain(gain) for gain in gains.values()):
        raise ValueError("Invalid aspect_gains")
    if not isinstance(snapshot["selection"], list) or not snapshot["selection"]:
        raise ValueError("Snapshot selection is required")
    seen, selection = set(), []
    for entry in snapshot["selection"]:
        if not isinstance(entry, dict) or set(entry) != {
            "card_id",
            "strength",
            "aspects",
        }:
            raise ValueError("Invalid selection entry")
        card_id = entry["card_id"]
        if not isinstance(card_id, str) or card_id not in card_map or card_id in seen:
            raise ValueError("Unknown or duplicate card")
        if not valid_strength(entry["strength"]):
            raise ValueError("strength must be 1 or 2")
        seen.add(card_id)
        selection.append(
            {
                "card_id": card_id,
                "strength": entry["strength"],
                "aspects": canonical_aspects(entry["aspects"]),
            }
        )
    return {
        "catalog_id": snapshot["catalog_id"],
        "catalog_hash": catalog["catalog_hash"],
        "selection": sorted(selection, key=lambda item: item["card_id"]),
        "aspect_gains": {key: float(gains[key]) for key in ASPECTS},
    }


def _provenance(provenance):
    required = {
        "fan_pin",
        "adapter_hash",
        "decoder_hash",
        "tokenizer_hash",
        "generation",
        "seeds",
    }
    if not isinstance(provenance, dict) or not required <= set(provenance):
        raise ValueError(
            "provenance must include source, decoder, tokenizer, generation, and seeds"
        )
    if any(
        not isinstance(provenance[key], str) or not provenance[key]
        for key in ("fan_pin", "adapter_hash", "decoder_hash", "tokenizer_hash")
    ):
        raise ValueError("provenance source identities are required")
    if (
        not isinstance(provenance["generation"], dict)
        or not provenance["generation"]
        or not isinstance(provenance["seeds"], list)
        or not provenance["seeds"]
    ):
        raise ValueError("provenance generation and seeds are required")
    _finite_json(provenance, "provenance")
    return json.loads(json.dumps(provenance, sort_keys=True))


def _refs(snapshot, policy, catalog=None):
    if catalog is None:
        from .catalog import load_catalog

        catalog = load_catalog(snapshot["catalog_id"], reviewed_only=True)
    cards = {card["id"]: card for card in catalog["cards"]}
    gains, unit, merged = snapshot["aspect_gains"], policy["reference_unit"], {}
    if unit == "card_description" and any(gain != 1 for gain in gains.values()):
        raise ValueError("card_description requires all aspect_gains to equal 1")
    for entry in snapshot["selection"]:
        card = cards[entry["card_id"]]
        aspects = [aspect for aspect in ASPECTS if aspect in entry["aspects"]]
        if unit == "aspect_phrase":
            candidates = [
                (card["aspects"][aspect], aspect, entry["strength"] * gains[aspect])
                for aspect in aspects
            ]
        else:
            candidates = [
                (
                    ", ".join(card["aspects"][aspect] for aspect in aspects),
                    None,
                    entry["strength"],
                )
            ]
        for text, aspect, weight in candidates:
            text = _normal_text(text)
            item = merged.setdefault(
                text, {"text": text, "weight": 0.0, "card_ids": [], "aspects": []}
            )
            item["weight"] += weight
            item["card_ids"].append(card["id"])
            if aspect is not None:
                item["aspects"].append(aspect)
    refs = [
        {
            "ref_id": _ref_id(text),
            "text": text,
            "weight": item["weight"],
            "card_ids": sorted(item["card_ids"]),
            "aspects": sorted(set(item["aspects"])),
        }
        for text, item in merged.items()
        if item["weight"] > 0
    ]
    if not refs:
        raise ValueError("参照を1つ以上残してください。")
    return sorted(refs, key=lambda item: (item["text"], item["ref_id"]))


def build_personalization(snapshot, *, prompt, policy, provenance, catalog=None):
    """Build a policy-bound, content-addressed snapshot; implicit lists are invalid."""
    from .fan_adapter import freeze_policy, profiling_argument, thaw_policy

    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must be a PreferenceSnapshot object")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    needed = {"revision", "catalog_id", "catalog_hash", "selection", "aspect_gains"}
    if set(snapshot) != needed:
        raise ValueError("Snapshot has unknown or missing fields")
    from .catalog import load_catalog

    # The caller may pass the server-owned catalog it already validated.
    resolved_catalog = (
        load_catalog(snapshot["catalog_id"], reviewed_only=True)
        if catalog is None
        else catalog
    )
    snapshot, effective, source = (
        _snapshot(snapshot, resolved_catalog),
        freeze_policy(policy),
        _provenance(provenance),
    )
    policy_value = thaw_policy(effective)
    refs = _refs(snapshot, policy_value, resolved_catalog)
    identity = {
        "snapshot": snapshot,
        "refs": refs,
        "prompt": prompt,
        "effective_policy": policy_value,
        "provenance": source,
    }
    content_hash = digest(identity)
    return {
        "refs": refs,
        "effective_policy": policy_value,
        "policy_hash": digest(policy_value),
        "provenance": source,
        "sample_size": profiling_argument(effective),
        "hash": content_hash,
        "personalization_hash": content_hash,
    }


def target_prompt(topic, config=CONFIG):
    """The same fixed sentence for plain and personalized images."""
    return compose_prompt(topic["basic_prompt_en"], [], config["generation"])


def run_cache_key(topic_id, personalization, config=CONFIG):
    return digest(
        {
            "topic": topic_id,
            "personalization": personalization["hash"],
            "generation": config["generation"],
            "seeds": config["seeds"],
        }
    )
