import copy
import json
from itertools import combinations

import pytest
from exhibit.catalog import build_catalog, load_catalog, validate_card_tokens
from exhibit.config import CONFIG, ROOT, read_json
from exhibit.domain import ASPECTS, digest, file_hash

from exhibit import catalog as catalog_module


def definition():
    return copy.deepcopy(read_json(ROOT / "configs/catalog-v2.json"))


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
    return {
        "schema_version": 1,
        "catalog_id": "catalog-v2",
        "generation": CONFIG["generation"],
        "prompt_set_hash": prompt_set_hash(cards),
        "tokenizers": {
            "repo_id": CONFIG["generation"]["pipeline_config"]["model"],
            "revision": CONFIG["generation"]["pipeline_config"]["revision"],
            "files": file_hashes,
        },
        "results": rows,
        "max_tokens": max(row["tokens"] for row in rows),
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
            "settings": CONFIG["generation"],
        }
        reviews[card["id"]] = {
            "reviewed": reviewed,
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
                "generation": CONFIG["generation"],
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


def test_v2_build_has_stable_orthogonal_64_card_layout():
    cards = build_catalog(definition())
    assert len(cards) == 64
    assert cards[0]["id"] == "girl-c0-l0-t0-m0"
    assert cards[-1]["id"] == "barista-c3-l3-t0-m2"
    assert len({card["id"] for card in cards}) == 64
    assert len({card["profile_id"] for card in cards}) == 16
    for subject in ("girl", "student", "traveler", "barista"):
        rows = [card for card in cards if card["subject_id"] == subject]
        assert len(rows) == 16
        assert len({card["seed"] for card in rows}) == 1
        for left, right in combinations(ASPECTS, 2):
            assert {
                (card["axis_levels"][left], card["axis_levels"][right]) for card in rows
            } == {(a, b) for a in range(4) for b in range(4)}
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
    ],
)
def test_build_catalog_rejects_malformed_or_noncanonical_definitions(mutate):
    value = definition()
    mutate(value)
    with pytest.raises(ValueError):
        build_catalog(value)


def test_v2_loader_accepts_a_fully_reviewed_synthetic_catalog(reviewed_v2):
    loaded = load_catalog(
        "catalog-v2",
        assets=reviewed_v2["root"],
        review_path=reviewed_v2["review"],
    )
    assert len(loaded["cards"]) == len(loaded["all_cards"]) == 64
    assert loaded["cards"][0]["id"] == "girl-c0-l0-t0-m0"


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
    assert "girl-c0-l0-t0-m0" not in {item["id"] for item in loaded["cards"]}


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
