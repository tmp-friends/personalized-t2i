"""Server-owned, content-addressed card catalogs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import ASSETS, CARDS_REVIEW, CONFIG, ROOT
from .domain import ASPECTS, build_cards, compose_prompt, digest, file_hash

V2 = ROOT / "configs/catalog-v2.json"
V2_REVIEW = ROOT / "configs/cards-v2-review.json"
TOKENIZER_FILES = tuple(
    f"{tokenizer}/{filename}"
    for tokenizer in ("tokenizer", "tokenizer_2")
    for filename in (
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
)
_TOKEN_ROW_FIELDS = (
    "prompt_hash",
    "tokenizer",
    "token_ids",
    "tokens",
    "limit",
    "overflow",
    "special_tokens",
)
LEGACY_OVERFLOW_IDS = (
    "girl-warm_soft",
    "student-warm_soft",
    "barista-warm_soft",
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def _copy(value):
    return json.loads(json.dumps(value))


def description_hash(card):
    """Bind a human review to every displayed and encoded description field."""
    return digest(
        {
            "aspects": card["aspects"],
            "aspects_ja": card["aspects_ja"],
            "ref_en": card["ref_en"],
            "label": card["label"],
            "profile_label": card["profile_label"],
        }
    )


def prompt_set_hash(cards):
    return digest([{"id": card["id"], "prompt": card["prompt"]} for card in cards])


def _axis_map(value, label):
    if not isinstance(value, dict) or set(value) != set(ASPECTS):
        raise ValueError(f"{label} must name the four catalog axes")
    for axis in ASPECTS:
        levels = value[axis]
        if (
            not isinstance(levels, list)
            or len(levels) != 4
            or any(not isinstance(item, str) or not item.strip() for item in levels)
            or len(set(levels)) != 4
        ):
            raise ValueError(f"{label}.{axis} requires four distinct non-empty levels")
    return value


def _forbidden_pairs(value):
    """Level pairs the phrase pilot found contradictory; each carries its reason."""
    if not isinstance(value, list) or not value:
        raise ValueError("forbidden_level_pairs requires at least one entry")
    pairs = {}
    for item in value:
        axes = item.get("axes") if isinstance(item, dict) else None
        levels = item.get("levels") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or set(item) != {"axes", "levels", "reason"}
            or not isinstance(axes, list)
            or len(axes) != 2
            or any(axis not in ASPECTS for axis in axes)
            or axes[0] == axes[1]
            or not isinstance(levels, list)
            or len(levels) != 2
            or any(type(level) is not int or not 0 <= level <= 3 for level in levels)
            or not isinstance(item["reason"], str)
            or not item["reason"].strip()
        ):
            raise ValueError("Invalid forbidden level pair")
        pairs.setdefault((axes[0], axes[1]), set()).add((levels[0], levels[1]))
    return pairs


def _profiles(definition):
    """The 16 profiles are listed explicitly; nothing derives them from an axis."""
    forbidden = _forbidden_pairs(definition["forbidden_level_pairs"])
    profiles = definition["profiles"]
    if not isinstance(profiles, list) or len(profiles) != 16:
        raise ValueError("Catalog requires sixteen profiles")
    levels = []
    for profile in profiles:
        if (
            not isinstance(profile, dict)
            or set(profile) != set(ASPECTS)
            or any(
                type(profile[axis]) is not int or not 0 <= profile[axis] <= 3
                for axis in ASPECTS
            )
        ):
            raise ValueError("Invalid profile")
        for (left, right), banned in forbidden.items():
            if (profile[left], profile[right]) in banned:
                raise ValueError(f"Profile uses a forbidden {left}/{right} pair")
        levels.append({axis: profile[axis] for axis in ASPECTS})
    if len({tuple(item.values()) for item in levels}) != 16:
        raise ValueError("Catalog profiles must be unique")
    for axis in ASPECTS:
        counts = [
            sum(1 for item in levels if item[axis] == level) for level in range(4)
        ]
        if counts != [4, 4, 4, 4]:
            raise ValueError(f"Catalog profiles are unbalanced on {axis}")
    return levels


def card_negative_prompt(definition):
    """The card-only negative prompt; the exhibit's own negative never changes."""
    value = (
        definition.get("card_negative_prompt") if isinstance(definition, dict) else None
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Catalog definition requires a card negative prompt")
    return value


def card_settings(catalog_id, definition=None):
    """Exhibit generation settings, with v2's single card-only negative override."""
    if catalog_id != "catalog-v2":
        return _copy(CONFIG["generation"])
    if definition is None:
        definition = _read_json_value(V2, "catalog-v2 definition")
    return {
        **_copy(CONFIG["generation"]),
        "negative_prompt": card_negative_prompt(definition),
    }


def build_catalog(definition):
    """Build the deterministic 4-subject x 16-profile catalog from the listed profiles."""
    required = {
        "catalog_id",
        "subjects",
        "axes",
        "axes_ja",
        "forbidden_level_pairs",
        "profiles",
        "card_negative_prompt",
        "generation",
    }
    if not isinstance(definition, dict) or set(definition) != required:
        raise ValueError("Invalid catalog definition")
    if definition["catalog_id"] != "catalog-v2":
        raise ValueError("Catalog definition must identify catalog-v2")
    if definition["generation"] != CONFIG["generation"]:
        raise ValueError("Catalog generation settings are not canonical")
    card_negative_prompt(definition)

    axes = _axis_map(definition["axes"], "axes")
    axes_ja = _axis_map(definition["axes_ja"], "axes_ja")
    subjects = definition["subjects"]
    if not isinstance(subjects, list) or len(subjects) != 4:
        raise ValueError("Catalog requires four subjects")
    subject_fields = {"id", "label", "basic_prompt_en", "seed"}
    if any(
        not isinstance(subject, dict) or set(subject) != subject_fields
        for subject in subjects
    ):
        raise ValueError("Invalid subject")
    for subject in subjects:
        if (
            any(
                not isinstance(subject[key], str) or not subject[key].strip()
                for key in ("id", "label", "basic_prompt_en")
            )
            or type(subject["seed"]) is not int
            or not re.fullmatch(r"[a-z0-9_-]+", subject["id"])
        ):
            raise ValueError("Invalid subject")
    ids = [subject["id"] for subject in subjects]
    if len(set(ids)) != 4:
        raise ValueError("Catalog requires four unique subjects")

    cards = []
    profiles = _profiles(definition)
    for subject in subjects:
        for levels in profiles:
            profile_id = "c{color}-l{lighting}-t{texture}-m{mood}".format(**levels)
            aspects = {axis: axes[axis][levels[axis]] for axis in ASPECTS}
            aspects_ja = {axis: axes_ja[axis][levels[axis]] for axis in ASPECTS}
            card_id = f"{subject['id']}-{profile_id}"
            profile_label = f"{aspects_ja['color']}・{aspects_ja['texture']}"
            cards.append(
                {
                    "id": card_id,
                    "subject_id": subject["id"],
                    "profile_id": profile_id,
                    "subject_label": subject["label"],
                    "profile_label": profile_label,
                    "label": f"{subject['label']} · {profile_label}",
                    "axis_levels": dict(levels),
                    "aspects": aspects,
                    "aspects_ja": aspects_ja,
                    "ref_en": ", ".join(aspects.values()),
                    "prompt": compose_prompt(
                        subject["basic_prompt_en"],
                        list(aspects.values()),
                        definition["generation"],
                    ),
                    "seed": subject["seed"],
                    "path": f"cards-v2/{card_id}.png",
                }
            )
    if len({card["id"] for card in cards}) != 64:
        raise ValueError("Catalog card IDs must be unique")
    return cards


def _v1_cards():
    cards = []
    base = build_cards()
    levels = {
        axis: {
            phrase: index
            for index, phrase in enumerate(
                dict.fromkeys(card["aspects"][axis] for card in base)
            )
        }
        for axis in ASPECTS
    }
    for card in base:
        copied = _copy(card)
        copied["axis_levels"] = {
            axis: levels[axis][card["aspects"][axis]] for axis in ASPECTS
        }
        cards.append(copied)
    return cards


def _read_json_value(path, label, *, missing=None):
    path = Path(path)
    if not path.exists():
        return missing
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid {label}") from exc


def _image_bytes_match(image, assets):
    try:
        path = (Path(assets) / image["path"]).resolve()
        return (
            path.is_relative_to(Path(assets).resolve())
            and isinstance(image["sha256"], str)
            and bool(_HEX_SHA256.fullmatch(image["sha256"]))
            and file_hash(path) == image["sha256"]
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _valid_v1(card, image, review, assets):
    if (
        not isinstance(image, dict)
        or not isinstance(review, dict)
        or review.get("reviewed") is not True
    ):
        return False
    return (
        _image_bytes_match(image, assets)
        and all(
            image.get(key) == card[key]
            for key in ("path", "seed", "prompt", "ref_en", "aspects")
        )
        and image.get("settings") == CONFIG["generation"]
    )


def _valid_v2(card, image, review, assets, settings):
    if not isinstance(image, dict) or not isinstance(review, dict):
        return False
    fields = (
        "path",
        "seed",
        "prompt",
        "ref_en",
        "aspects",
        "aspects_ja",
        "label",
        "profile_label",
    )
    verdicts = review.get("aspects")
    return (
        _image_bytes_match(image, assets)
        and all(image.get(key) == card[key] for key in fields)
        and image.get("settings") == settings
        and review.get("reviewed") is True
        and review.get("image_sha256") == image.get("sha256")
        and review.get("description_hash") == description_hash(card)
        and isinstance(verdicts, dict)
        and set(verdicts) == set(ASPECTS)
        and all(verdicts[axis] is True for axis in ASPECTS)
    )


def _manifest_images(manifest, *, catalog_id, settings=None):
    if manifest is None:
        return {}
    if not isinstance(manifest, dict):
        raise TypeError(f"Invalid {catalog_id} manifest")
    if catalog_id == "catalog-v2":
        required = {
            "version",
            "catalog_id",
            "generation",
            "token_validation",
            "images",
        }
        if (
            set(manifest) != required
            or manifest["version"] != 2
            or manifest["catalog_id"] != catalog_id
            or manifest["generation"] != settings
            or not isinstance(manifest["token_validation"], dict)
            or not isinstance(manifest["images"], dict)
        ):
            raise ValueError("Invalid catalog-v2 manifest")
    elif not isinstance(manifest.get("images"), dict):
        raise ValueError("Invalid catalog-v1 manifest")
    return manifest["images"]


def load_catalog(catalog_id, *, reviewed_only=True, assets=ASSETS, review_path=None):
    """Load a known catalog and filter to image-bound human reviews by default."""
    assets = Path(assets)
    if catalog_id == "catalog-v1":
        all_cards = _v1_cards()
        manifest = _read_json_value(
            assets / "manifest.json", "catalog-v1 manifest", missing={"images": {}}
        )
        images = _manifest_images(manifest, catalog_id=catalog_id)
        review = _read_json_value(
            review_path or CARDS_REVIEW, "catalog-v1 review", missing={}
        )
        if not isinstance(review, dict):
            raise ValueError("Invalid catalog-v1 review")
        eligible = [
            card
            for card in all_cards
            if _valid_v1(card, images.get(card["id"]), review.get(card["id"]), assets)
        ]
    elif catalog_id == "catalog-v2":
        definition = _read_json_value(V2, "catalog-v2 definition")
        all_cards = build_catalog(definition)
        settings = card_settings(catalog_id, definition)
        manifest = _read_json_value(
            assets / "catalog-v2.json", "catalog-v2 manifest", missing=None
        )
        images = _manifest_images(manifest, catalog_id=catalog_id, settings=settings)
        review = _read_json_value(
            review_path or V2_REVIEW, "catalog-v2 review", missing={}
        )
        if not isinstance(review, dict):
            raise ValueError("Invalid catalog-v2 review")
        eligible = [
            card
            for card in all_cards
            if _valid_v2(
                card, images.get(card["id"]), review.get(card["id"]), assets, settings
            )
        ]
    else:
        raise ValueError("Unknown catalog")

    image_hashes = {
        card["id"]: (
            images.get(card["id"], {}).get("sha256")
            if isinstance(images.get(card["id"]), dict)
            else None
        )
        for card in all_cards
    }
    identity = {
        "catalog_id": catalog_id,
        "all_cards": all_cards,
        "image_hashes": image_hashes,
    }
    return {
        "catalog_id": catalog_id,
        "catalog_hash": digest(identity),
        "all_cards": _copy(all_cards),
        "cards": _copy(eligible if reviewed_only else all_cards),
    }


def _token_row(text, name, tokenizer):
    encoded = tokenizer(text, add_special_tokens=True)
    ids = list(encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids)
    return {
        "prompt_hash": digest(text),
        "tokenizer": name,
        "token_ids": ids,
        "tokens": len(ids),
        "limit": 77,
        "overflow": len(ids) > 77,
        "special_tokens": True,
    }


def validate_card_tokens(cards, tokenizers):
    return [
        {"card_id": card["id"], **_token_row(card["prompt"], name, tokenizer)}
        for card in cards
        for name, tokenizer in tokenizers.items()
    ]


def validate_negative_tokens(negative_prompt, tokenizers):
    """FAN truncates the negative prompt at 77 just as silently as the positive."""
    rows = [
        _token_row(negative_prompt, name, tokenizer)
        for name, tokenizer in tokenizers.items()
    ]
    return {
        "prompt_hash": digest(negative_prompt),
        "results": rows,
        "max_tokens": max((row["tokens"] for row in rows), default=0),
    }


def _validate_negative_report(negative, negative_prompt):
    if (
        not isinstance(negative, dict)
        or set(negative) != {"prompt_hash", "results", "max_tokens"}
        or negative["prompt_hash"] != digest(negative_prompt)
        or not isinstance(negative["results"], list)
    ):
        raise ValueError("Negative prompt token validation does not match the catalog")
    names = set()
    for row in negative["results"]:
        ids = row.get("token_ids") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or set(row) != set(_TOKEN_ROW_FIELDS)
            or row["tokenizer"] not in ("tokenizer", "tokenizer_2")
            or row["tokenizer"] in names
            or row["prompt_hash"] != digest(negative_prompt)
            or row["special_tokens"] is not True
            or not isinstance(ids, list)
            or any(type(item) is not int for item in ids)
            or row["tokens"] != len(ids)
            or row["limit"] != 77
            or row["overflow"] is not (len(ids) > 77)
        ):
            raise TypeError("Invalid negative token validation row")
        names.add(row["tokenizer"])
    if names != {"tokenizer", "tokenizer_2"}:
        raise ValueError("Negative prompt token validation rows are incomplete")
    maximum = max(row["tokens"] for row in negative["results"])
    if negative["max_tokens"] != maximum or maximum > 77:
        raise ValueError("Card negative prompt token validation failed")


def validate_token_report(report, cards, *, catalog_id="catalog-v2", generation=None):
    """Reject reports not bound to the exact prompts and pinned tokenizer files."""
    if not isinstance(report, dict):
        raise TypeError("Invalid token validation report")
    generation = generation if generation is not None else card_settings(catalog_id)
    required = {
        "schema_version",
        "catalog_id",
        "generation",
        "prompt_set_hash",
        "tokenizers",
        "results",
        "max_tokens",
        "negative_validation",
        "legacy_overflow_evidence",
    }
    if (
        set(report) != required
        or report["schema_version"] != 2
        or report["catalog_id"] != catalog_id
        or report["generation"] != generation
        or report["prompt_set_hash"] != prompt_set_hash(cards)
    ):
        raise ValueError("Token validation report does not match the catalog")
    _validate_negative_report(
        report["negative_validation"], generation["negative_prompt"]
    )
    pipeline = CONFIG["generation"]["pipeline_config"]
    provenance = report["tokenizers"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"repo_id", "revision", "files"}
        or provenance["repo_id"] != pipeline["model"]
        or provenance["revision"] != pipeline["revision"]
        or not isinstance(provenance["files"], dict)
        or set(provenance["files"]) != set(TOKENIZER_FILES)
        or any(
            not isinstance(value, str) or not _HEX_SHA256.fullmatch(value)
            for value in provenance["files"].values()
        )
    ):
        raise ValueError("Token validation provenance is incomplete")

    rows = report["results"]
    cards_by_id = {card["id"]: card for card in cards}
    expected_pairs = {
        (card["id"], tokenizer)
        for card in cards
        for tokenizer in ("tokenizer", "tokenizer_2")
    }
    if not isinstance(rows, list) or len(rows) != len(expected_pairs):
        raise ValueError("Token validation rows are incomplete")
    pairs = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Invalid token validation row")
        card = cards_by_id.get(row.get("card_id"))
        pair = (row.get("card_id"), row.get("tokenizer"))
        ids = row.get("token_ids")
        if (
            card is None
            or pair not in expected_pairs
            or pair in pairs
            or row.get("prompt_hash") != digest(card["prompt"])
            or row.get("special_tokens") is not True
            or not isinstance(ids, list)
            or any(type(item) is not int for item in ids)
            or row.get("tokens") != len(ids)
            or row.get("limit") != 77
            or row.get("overflow") is not (len(ids) > 77)
        ):
            raise TypeError("Invalid token validation row")
        pairs.add(pair)
    if pairs != expected_pairs:
        raise ValueError("Token validation rows are incomplete")
    maximum = max((len(row["token_ids"]) for row in rows), default=0)
    if report["max_tokens"] != maximum or maximum > 77:
        raise ValueError("Card token validation failed")

    legacy = report["legacy_overflow_evidence"]
    if (
        not isinstance(legacy, dict)
        or set(legacy) != {"path", "sha256", "over_limit_ids"}
        or legacy["over_limit_ids"] != list(LEGACY_OVERFLOW_IDS)
        or not isinstance(legacy["sha256"], str)
        or not _HEX_SHA256.fullmatch(legacy["sha256"])
    ):
        raise ValueError("Legacy overflow evidence is missing")
    return _copy(report)
