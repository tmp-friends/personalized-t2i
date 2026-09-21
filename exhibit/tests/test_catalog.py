import copy
import json
from collections import Counter
from itertools import combinations

import pytest
from exhibit.catalog import (
    build_catalog,
    card_settings,
    load_catalog,
    validate_card_tokens,
    validate_negative_tokens,
    validate_token_report,
)
from exhibit.config import CONFIG, ROOT, read_json
from exhibit.domain import ASPECTS, digest, file_hash

from exhibit import catalog as catalog_module

# The pairs the phrase pilot found contradictory; catalog-v2.json states why.
FORBIDDEN_PAIRS = (
    ("color", "lighting", 0, 1),
    ("color", "lighting", 3, 3),
    ("color", "texture", 1, 0),
    ("color", "texture", 3, 3),
    ("lighting", "texture", 2, 0),
    ("lighting", "texture", 2, 3),
    ("lighting", "texture", 3, 0),
    ("lighting", "texture", 3, 2),
)


def definition():
    return copy.deepcopy(read_json(ROOT / "configs/catalog-v2.json"))


def v2_settings():
    return card_settings("catalog-v2", definition())


def description_hash(card):
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


def token_validation(cards, *, overflow=False):
    file_hashes = {
        f"{name}/{filename}": "1" * 64
        for name in ("tokenizer", "tokenizer_2")
        for filename in (
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    }
    rows = []
    for card in cards:
        for tokenizer in ("tokenizer", "tokenizer_2"):
            rows.append(
                {
                    "card_id": card["id"],
                    "prompt_hash": digest(card["prompt"]),
                    "tokenizer": tokenizer,
                    "token_ids": [49406, 100, 49407],
                    "tokens": 78 if overflow and card is cards[0] else 3,
                    "limit": 77,
                    "overflow": overflow and card is cards[0],
                    "special_tokens": True,
                }
            )
    settings = v2_settings()
    negative = [
        {
            "prompt_hash": digest(settings["negative_prompt"]),
            "tokenizer": tokenizer,
            "token_ids": [49406, 100, 49407],
            "tokens": 3,
            "limit": 77,
            "overflow": False,
            "special_tokens": True,
        }
        for tokenizer in ("tokenizer", "tokenizer_2")
    ]
    return {
        "schema_version": 2,
        "catalog_id": "catalog-v2",
        "generation": settings,
        "prompt_set_hash": prompt_set_hash(cards),
        "tokenizers": {
            "repo_id": CONFIG["generation"]["pipeline_config"]["model"],
            "revision": CONFIG["generation"]["pipeline_config"]["revision"],
            "files": file_hashes,
        },
        "results": rows,
        "max_tokens": max(row["tokens"] for row in rows),
        "negative_validation": {
            "prompt_hash": digest(settings["negative_prompt"]),
            "results": negative,
            "max_tokens": 3,
        },
        "legacy_overflow_evidence": {
            "path": "configs/legacy-card-token-overflow.json",
            "sha256": "2" * 64,
            "over_limit_ids": [
                "girl-warm_soft",
                "student-warm_soft",
                "barista-warm_soft",
            ],
        },
    }


def write_v2_bundle(root, review_path, *, reviewed=True):
    cards = build_catalog(definition())
    settings = v2_settings()
    images = {}
    reviews = {}
    for card in cards:
        path = root / card["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("image:" + card["id"]).encode())
        images[card["id"]] = {
            "path": card["path"],
            "sha256": file_hash(path),
            "seed": card["seed"],
            "prompt": card["prompt"],
            "ref_en": card["ref_en"],
            "aspects": card["aspects"],
            "aspects_ja": card["aspects_ja"],
            "label": card["label"],
            "profile_label": card["profile_label"],
            "settings": settings,
        }
        reviews[card["id"]] = {
            # `reviewed` is a flag for every card, or the set that passed.
            "reviewed": reviewed
            if isinstance(reviewed, bool)
            else card["id"] in reviewed,
            "image_sha256": images[card["id"]]["sha256"],
            "description_hash": description_hash(card),
            "aspects": {aspect: True for aspect in ASPECTS},
            "note": "human note",
        }
    (root / "catalog-v2.json").write_text(
        json.dumps(
            {
                "version": 2,
                "catalog_id": "catalog-v2",
                "generation": settings,
                "token_validation": token_validation(cards),
                "images": images,
            }
        )
    )
    review_path.write_text(json.dumps(reviews))
    return cards, images, reviews


@pytest.fixture
def reviewed_v2(tmp_path):
    root = tmp_path / "assets"
    review = tmp_path / "review.json"
    cards, images, reviews = write_v2_bundle(root, review)
    return {
        "root": root,
        "review": review,
        "cards": cards,
        "images": images,
        "reviews": reviews,
    }


def test_v2_build_has_a_balanced_explicit_16_profile_design():
    """The 16 profiles are listed, not derived; the list is balanced and legal."""
    cards = build_catalog(definition())
    assert len(cards) == 64
    assert cards[0]["id"] == "girl-c0-l0-t0-m2"
    assert cards[-1]["id"] == "barista-c3-l2-t1-m3"
    assert len({card["id"] for card in cards}) == 64
    assert len({card["profile_id"] for card in cards}) == 16
    for subject in ("girl", "student", "traveler", "barista"):
        rows = [card for card in cards if card["subject_id"] == subject]
        assert len(rows) == 16
        assert len({card["seed"] for card in rows}) == 1
        for axis in ASPECTS:
            assert Counter(card["axis_levels"][axis] for card in rows) == Counter(
                dict.fromkeys(range(4), 4)
            )
        for left, right, first, second in FORBIDDEN_PAIRS:
            assert all(
                (card["axis_levels"][left], card["axis_levels"][right])
                != (first, second)
                for card in rows
            ), (left, right, first, second)
        # 85 of the 96 level pairs occur; eight of the eleven gaps are forbidden.
        pairs = Counter(
            (left, right, card["axis_levels"][left], card["axis_levels"][right])
            for left, right in combinations(ASPECTS, 2)
            for card in rows
        )
        assert len(pairs) == 85
        assert max(pairs.values()) == 2
        assert all("c" not in card["profile_label"] for card in rows)
        assert all(card["label"].startswith(card["subject_label"]) for card in rows)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(catalog_id="wrong"),
        lambda value: value.update(extra=True),
        lambda value: value["axes"].pop("mood"),
        lambda value: value["axes"]["mood"].pop(),
        lambda value: value["axes_ja"]["mood"].pop(),
        lambda value: value["axes_ja"]["mood"].__setitem__(0, ""),
        lambda value: value.update(subjects=[None] * 4),
        lambda value: value["subjects"].__setitem__(
            1, copy.deepcopy(value["subjects"][0])
        ),
        lambda value: value["subjects"][0].update(id=[]),
        lambda value: value.update(generation={**value["generation"], "steps": 1}),
        lambda value: value.pop("card_negative_prompt"),
        lambda value: value.update(card_negative_prompt="  "),
        lambda value: value["profiles"].pop(),
        lambda value: value["profiles"].__setitem__(
            1, copy.deepcopy(value["profiles"][0])
        ),
        lambda value: value["profiles"][0].update(color=4),
        lambda value: value["profiles"][0].update(mood=3),
        lambda value: value["profiles"][0].update(lighting=1),
        lambda value: value.update(forbidden_level_pairs=[]),
        lambda value: value["forbidden_level_pairs"][0].pop("reason"),
        lambda value: value["forbidden_level_pairs"][0].update(axes=["color", "color"]),
        lambda value: value.update(seed_overrides=[]),
        lambda value: value.update(seed_overrides={"girl-nope": 11}),
        lambda value: value.update(seed_overrides={"girl-c0-l0-t0-m2": "11"}),
        lambda value: value.update(seed_overrides={"girl-c0-l0-t0-m2": 601}),
    ],
)
def test_build_catalog_rejects_malformed_or_noncanonical_definitions(mutate):
    value = definition()
    mutate(value)
    with pytest.raises(ValueError):
        build_catalog(value)


def test_seed_overrides_reroll_one_card_and_invalidate_only_its_review(reviewed_v2):
    """A per-card seed changes that card's contract; the other 63 stay reviewed."""
    value = definition()
    assert value["seed_overrides"] == {}
    target = "girl-c0-l0-t0-m2"
    value["seed_overrides"] = {target: 9601}
    cards = {card["id"]: card for card in build_catalog(value)}
    assert cards[target]["seed"] == 9601
    assert all(
        card["seed"] == 601
        for card_id, card in cards.items()
        if card_id.startswith("girl-") and card_id != target
    )
    assert cards["student-c0-l0-t0-m2"]["seed"] == 602


def test_a_seed_override_invalidates_only_that_card(reviewed_v2, tmp_path, monkeypatch):
    value = definition()
    target = "girl-c0-l0-t0-m2"
    value["seed_overrides"] = {target: 9601}
    changed = tmp_path / "catalog-definition.json"
    changed.write_text(json.dumps(value))
    monkeypatch.setattr(catalog_module, "V2", changed)
    loaded = load_catalog(
        "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
    )
    assert {item["id"] for item in loaded["cards"]} == {
        card["id"] for card in reviewed_v2["cards"]
    } - {target}


def test_v2_cards_are_generated_with_the_catalog_only_negative_prompt():
    value = definition()
    settings = card_settings("catalog-v2", value)
    assert settings["negative_prompt"] == value["card_negative_prompt"]
    assert settings["negative_prompt"] != CONFIG["generation"]["negative_prompt"]
    assert "from behind" in settings["negative_prompt"]
    # The override is the only difference; the exhibit's own settings are untouched.
    assert {key: settings[key] for key in settings if key != "negative_prompt"} == {
        key: item
        for key, item in CONFIG["generation"].items()
        if key != "negative_prompt"
    }
    assert card_settings("catalog-v1") == CONFIG["generation"]


def test_v2_review_is_invalidated_when_cards_used_the_exhibit_negative_prompt(
    reviewed_v2,
):
    card = reviewed_v2["cards"][0]
    manifest_path = reviewed_v2["root"] / "catalog-v2.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"][card["id"]]["settings"] = CONFIG["generation"]
    manifest_path.write_text(json.dumps(manifest))
    loaded = load_catalog(
        "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
    )
    assert card["id"] not in {item["id"] for item in loaded["cards"]}


def test_v2_loader_accepts_a_fully_reviewed_synthetic_catalog(reviewed_v2):
    loaded = load_catalog(
        "catalog-v2",
        assets=reviewed_v2["root"],
        review_path=reviewed_v2["review"],
    )
    assert len(loaded["cards"]) == len(loaded["all_cards"]) == 64
    assert loaded["cards"][0]["id"] == "girl-c0-l0-t0-m2"


@pytest.mark.parametrize("tamper", ["bytes", "sha"])
def test_v2_review_is_invalidated_by_image_bytes_or_sha(reviewed_v2, tamper):
    card = reviewed_v2["cards"][0]
    if tamper == "bytes":
        (reviewed_v2["root"] / card["path"]).write_bytes(b"tampered")
    else:
        manifest = json.loads((reviewed_v2["root"] / "catalog-v2.json").read_text())
        manifest["images"][card["id"]]["sha256"] = "0" * 64
        (reviewed_v2["root"] / "catalog-v2.json").write_text(json.dumps(manifest))
    loaded = load_catalog(
        "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
    )
    assert card["id"] not in {item["id"] for item in loaded["cards"]}


@pytest.mark.parametrize("language", ["english", "japanese"])
def test_v2_review_is_invalidated_by_description_changes(
    reviewed_v2, tmp_path, monkeypatch, language
):
    changed = definition()
    axis = "color"
    if language == "english":
        changed["axes"][axis][0] += ", edited"
    else:
        changed["axes_ja"][axis][0] += "（編集）"
    changed_path = tmp_path / "catalog-definition.json"
    changed_path.write_text(json.dumps(changed))
    monkeypatch.setattr(catalog_module, "V2", changed_path)
    loaded = load_catalog(
        "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
    )
    assert "girl-c0-l0-t0-m2" not in {item["id"] for item in loaded["cards"]}


@pytest.mark.parametrize(
    "verdicts",
    [
        {"color": True, "lighting": True, "texture": True},
        {"color": True, "lighting": True, "texture": True, "mood": False},
        {"color": True, "lighting": True, "texture": True, "mood": 1},
        {"color": True, "lighting": True, "texture": True, "mood": True, "other": True},
    ],
)
def test_v2_requires_exactly_four_strict_axis_verdicts(reviewed_v2, verdicts):
    card = reviewed_v2["cards"][0]
    reviews = reviewed_v2["reviews"]
    reviews[card["id"]]["aspects"] = verdicts
    reviewed_v2["review"].write_text(json.dumps(reviews))
    loaded = load_catalog(
        "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
    )
    assert card["id"] not in {item["id"] for item in loaded["cards"]}


def test_catalog_hash_ignores_review_notes_and_filter_but_tracks_content(reviewed_v2):
    first = load_catalog(
        "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
    )
    reviews = reviewed_v2["reviews"]
    card = reviewed_v2["cards"][0]
    reviews[card["id"]]["note"] = "a different note"
    reviews[card["id"]]["reviewed"] = False
    reviewed_v2["review"].write_text(json.dumps(reviews))
    filtered = load_catalog(
        "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
    )
    unfiltered = load_catalog(
        "catalog-v2",
        reviewed_only=False,
        assets=reviewed_v2["root"],
        review_path=reviewed_v2["review"],
    )
    assert (
        filtered["catalog_hash"] == first["catalog_hash"] == unfiltered["catalog_hash"]
    )

    manifest = json.loads((reviewed_v2["root"] / "catalog-v2.json").read_text())
    manifest["images"][card["id"]]["sha256"] = "0" * 64
    (reviewed_v2["root"] / "catalog-v2.json").write_text(json.dumps(manifest))
    changed = load_catalog(
        "catalog-v2",
        reviewed_only=False,
        assets=reviewed_v2["root"],
        review_path=reviewed_v2["review"],
    )
    assert changed["catalog_hash"] != first["catalog_hash"]


@pytest.mark.parametrize(
    "manifest_value",
    [[], {"version": 99}, {"version": 2, "catalog_id": "wrong", "images": {}}],
)
def test_v2_loader_rejects_malformed_generated_manifests(tmp_path, manifest_value):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "catalog-v2.json").write_text(json.dumps(manifest_value))
    with pytest.raises((TypeError, ValueError)):
        load_catalog("catalog-v2", reviewed_only=False, assets=root)


def test_v2_loader_rejects_malformed_review_manifest(reviewed_v2):
    reviewed_v2["review"].write_text("[]")
    with pytest.raises(ValueError):
        load_catalog(
            "catalog-v2", assets=reviewed_v2["root"], review_path=reviewed_v2["review"]
        )


def test_v1_loader_preserves_all_frozen_cards_and_legacy_contract(asset_tree):
    loaded = load_catalog(
        "catalog-v1", assets=asset_tree["root"], review_path=asset_tree["review"]
    )
    assert len(loaded["cards"]) == len(loaded["all_cards"]) == 16
    assert loaded["all_cards"][0]["id"] == "girl-warm_soft"
    assert loaded["all_cards"][-1]["id"] == "barista-vivid_lively"
    warm = next(card for card in loaded["all_cards"] if card["id"] == "girl-warm_soft")
    cool = next(card for card in loaded["all_cards"] if card["id"] == "girl-cool_clean")
    assert warm["axis_levels"]["mood"] == cool["axis_levels"]["mood"]


class FakeTokenizer:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def __call__(self, prompt, *, add_special_tokens):
        self.calls.append((prompt, add_special_tokens))
        return {"input_ids": self.ids}


def test_validate_card_tokens_records_special_token_ids_and_overflow():
    cards = [
        {"id": "short", "prompt": "short prompt"},
        {"id": "long", "prompt": "long prompt"},
    ]
    short = FakeTokenizer([49406, 10, 49407])
    long = FakeTokenizer(list(range(78)))
    rows = validate_card_tokens(cards, {"tokenizer": short, "tokenizer_2": long})
    assert rows[0] == {
        "card_id": "short",
        "prompt_hash": digest("short prompt"),
        "tokenizer": "tokenizer",
        "token_ids": [49406, 10, 49407],
        "tokens": 3,
        "limit": 77,
        "overflow": False,
        "special_tokens": True,
    }
    assert rows[-1]["overflow"] is True
    assert short.calls == [("short prompt", True), ("long prompt", True)]
    assert long.calls == [("short prompt", True), ("long prompt", True)]


def test_validate_negative_tokens_checks_the_card_negative_against_the_same_limit():
    short = FakeTokenizer([49406, 10, 49407])
    long = FakeTokenizer(list(range(78)))
    result = validate_negative_tokens(
        "worst quality", {"tokenizer": short, "tokenizer_2": long}
    )
    assert result["prompt_hash"] == digest("worst quality")
    assert result["max_tokens"] == 78
    assert [row["tokenizer"] for row in result["results"]] == [
        "tokenizer",
        "tokenizer_2",
    ]
    assert [row["overflow"] for row in result["results"]] == [False, True]
    assert short.calls == [("worst quality", True)]


def test_token_report_binds_the_card_negative_prompt_and_rejects_its_overflow():
    cards = build_catalog(definition())[:2]
    report = token_validation(cards)
    assert validate_token_report(report, cards, generation=v2_settings())

    overflowing = copy.deepcopy(report)
    row = overflowing["negative_validation"]["results"][0]
    row.update(token_ids=list(range(78)), tokens=78, overflow=True)
    overflowing["negative_validation"]["max_tokens"] = 78
    with pytest.raises(ValueError, match="negative prompt token validation failed"):
        validate_token_report(overflowing, cards, generation=v2_settings())

    exhibit_negative = copy.deepcopy(report)
    exhibit_negative["negative_validation"]["prompt_hash"] = digest(
        CONFIG["generation"]["negative_prompt"]
    )
    with pytest.raises(ValueError, match="Negative prompt token validation"):
        validate_token_report(exhibit_negative, cards, generation=v2_settings())

    without = copy.deepcopy(report)
    without.pop("negative_validation")
    with pytest.raises(ValueError, match="does not match the catalog"):
        validate_token_report(without, cards, generation=v2_settings())
