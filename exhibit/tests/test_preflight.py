import json

import pytest
from exhibit.config import CONFIG
from exhibit.domain import CARDS
from exhibit.preflight import check_assets

from exhibit import preflight


def result_for(tree, **kwargs):
    review = kwargs.pop("review", tree["review"])
    return check_assets(tree["root"], review=review, **kwargs)


def test_missing_assets_block_live_start_without_crashing(tmp_path):
    result = check_assets(tmp_path, review=tmp_path / "absent.json")
    assert not result["ready"]
    assert any("manifest.json" in e for e in result["errors"])
    assert any("Unreviewed card" in e for e in result["errors"])


def test_a_complete_bundle_is_ready(asset_tree):
    result = result_for(asset_tree)
    assert result["ready"], result["errors"]
    assert result["reviewed_cards"] == 16
    assert result["samples"] == 6
    assert result["mode"] == "fan-live"


def test_unreviewed_cards_fail_preflight(asset_tree):
    review = json.loads(asset_tree["review"].read_text())
    stale = next(iter(review))
    review[stale]["reviewed"] = False
    asset_tree["review"].write_text(json.dumps(review))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert f"Unreviewed card: {stale}" in result["errors"]


def test_corrupt_and_stale_images_are_detected(asset_tree):
    manifest = json.loads((asset_tree["root"] / "manifest.json").read_text())
    card = next(iter(CARDS))
    manifest["images"][card]["sha256"] = "incorrect"
    manifest["images"]["cat-0"]["seed"] = -1
    manifest["images"]["cat-1"]["prompt"] = "a different sentence"
    (asset_tree["root"] / "manifest.json").write_text(json.dumps(manifest))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert f"Missing or corrupt image: {card}" in result["errors"]
    assert "Generic contract mismatch: cat-0" in result["errors"]
    assert "Generic contract mismatch: cat-1" in result["errors"]


def test_changed_generation_settings_invalidate_the_bundle(asset_tree):
    manifest = json.loads((asset_tree["root"] / "manifest.json").read_text())
    manifest["generation"] = {**CONFIG["generation"], "steps": 1}
    (asset_tree["root"] / "manifest.json").write_text(json.dumps(manifest))
    result = result_for(asset_tree)
    assert "Generation settings mismatch" in result["errors"]


def test_samples_must_match_the_personalization_they_claim(asset_tree):
    samples = json.loads((asset_tree["root"] / "samples.json").read_text())
    samples[0]["personalization"]["hash"] = "rewritten"
    samples[1]["images"][0]["seed"] = -5
    samples[2]["selection"] = [{"card_id": "unknown-card", "aspects_off": []}]
    (asset_tree["root"] / "samples.json").write_text(json.dumps(samples))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert any("personalization mismatch" in e for e in result["errors"])
    assert any("seed order" in e for e in result["errors"])
    assert any("selection is unusable" in e for e in result["errors"])


def test_write_preflight_persists_the_fresh_model_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "check_assets",
        lambda: {"ready": True, "errors": [], "fixed_images": 40},
    )
    monkeypatch.setattr(
        preflight,
        "check_models",
        lambda: {
            "ready": True,
            "errors": [],
            "models": [{"model": "OnomaAIResearch/Illustrious-XL-v2.0"}],
            "fan": {"ready": True, "errors": []},
        },
    )
    output = tmp_path / "preflight.json"

    result = preflight.write_preflight(output, include_models=True)

    assert result["ready"]
    assert json.loads(output.read_text()) == result


def _description_hash(card):
    from exhibit.domain import digest

    return digest(
        {
            "aspects": card["aspects"],
            "aspects_ja": card["aspects_ja"],
            "ref_en": card["ref_en"],
            "label": card["label"],
            "profile_label": card["profile_label"],
        }
    )


def _install_reviewed_v2(asset_tree):
    from exhibit.catalog import build_catalog, card_settings
    from exhibit.config import ROOT, read_json
    from exhibit.domain import ASPECTS, digest, file_hash

    root = asset_tree["root"]
    definition = read_json(ROOT / "configs/catalog-v2.json")
    cards = build_catalog(definition)
    settings = card_settings("catalog-v2", definition)
    rows = []
    images = {}
    reviews = {}
    for card in cards:
        path = root / card["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("v2:" + card["id"]).encode())
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
            "reviewed": True,
            "image_sha256": images[card["id"]]["sha256"],
            "description_hash": _description_hash(card),
            "aspects": {aspect: True for aspect in ASPECTS},
            "note": "",
        }
        for tokenizer in ("tokenizer", "tokenizer_2"):
            rows.append(
                {
                    "card_id": card["id"],
                    "prompt_hash": digest(card["prompt"]),
                    "tokenizer": tokenizer,
                    "token_ids": [49406, 42, 49407],
                    "tokens": 3,
                    "limit": 77,
                    "overflow": False,
                    "special_tokens": True,
                }
            )
    files = {
        f"{tokenizer}/{filename}": "1" * 64
        for tokenizer in ("tokenizer", "tokenizer_2")
        for filename in (
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    }
    token_validation = {
        "schema_version": 2,
        "catalog_id": "catalog-v2",
        "generation": settings,
        "prompt_set_hash": digest(
            [{"id": card["id"], "prompt": card["prompt"]} for card in cards]
        ),
        "tokenizers": {
            "repo_id": CONFIG["generation"]["pipeline_config"]["model"],
            "revision": CONFIG["generation"]["pipeline_config"]["revision"],
            "files": files,
        },
        "results": rows,
        "max_tokens": 3,
        "negative_validation": {
            "prompt_hash": digest(settings["negative_prompt"]),
            "results": [
                {
                    "prompt_hash": digest(settings["negative_prompt"]),
                    "tokenizer": tokenizer,
                    "token_ids": [49406, 42, 49407],
                    "tokens": 3,
                    "limit": 77,
                    "overflow": False,
                    "special_tokens": True,
                }
                for tokenizer in ("tokenizer", "tokenizer_2")
            ],
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
    (root / "catalog-v2.json").write_text(
        json.dumps(
            {
                "version": 2,
                "catalog_id": "catalog-v2",
                "generation": settings,
                "token_validation": token_validation,
                "images": images,
            }
        )
    )
    review = asset_tree["root"].parent / "cards-v2-review.json"
    review.write_text(json.dumps(reviews))
    return cards, review


def test_v2_preflight_uses_v2_cards_and_v1_generic_assets_separately(asset_tree):
    cards, review = _install_reviewed_v2(asset_tree)
    result = result_for(asset_tree, catalog_id="catalog-v2", review=review)
    assert result["ready"], result["errors"]
    assert result["catalog_id"] == "catalog-v2"
    assert result["cards"] == result["reviewed_cards"] == len(cards) == 64
    assert result["fixed_images"] == 64
    assert result["generic_images"] == len(CONFIG["topics"]) * len(CONFIG["seeds"])


def test_v2_preflight_rejects_tampered_card_contract(asset_tree):
    cards, review = _install_reviewed_v2(asset_tree)
    manifest_path = asset_tree["root"] / "catalog-v2.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"][cards[0]["id"]]["aspects_ja"]["color"] = "改ざん"
    manifest_path.write_text(json.dumps(manifest))
    result = result_for(asset_tree, catalog_id="catalog-v2", review=review)
    assert not result["ready"]
    assert f"Card contract mismatch: {cards[0]['id']}" in result["errors"]


def test_preflight_defaults_to_the_active_config_catalog(asset_tree, monkeypatch):
    _, review = _install_reviewed_v2(asset_tree)
    monkeypatch.setattr(preflight, "CONFIG", {**CONFIG, "catalog_id": "catalog-v2"})
    result = check_assets(asset_tree["root"], review=review)
    assert result["ready"], result["errors"]
    assert result["catalog_id"] == "catalog-v2"


def test_sample_manifest_declares_required_ids_and_rejects_deletion(
    asset_tree, monkeypatch
):
    samples_path = asset_tree["root"] / "samples.json"
    samples = json.loads(samples_path.read_text())
    required = [sample["id"] for sample in samples]
    monkeypatch.setattr(
        preflight,
        "CONFIG",
        {
            **CONFIG,
            "catalog_id": "catalog-v1",
            "sample_manifest": {
                "catalog_id": "catalog-v1",
                "required_ids": required,
            },
        },
    )
    samples_path.write_text(json.dumps(samples[:-1]))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert "Sample manifest IDs mismatch" in result["errors"]


def test_write_preflight_accepts_an_explicit_catalog(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        preflight,
        "check_assets",
        lambda *, catalog_id: (
            calls.append(catalog_id) or {"ready": False, "errors": ["not prepared"]}
        ),
    )
    result = preflight.write_preflight(
        tmp_path / "preflight.json", catalog_id="catalog-v2"
    )
    assert calls == ["catalog-v2"]
    assert result["errors"] == ["not prepared"]


@pytest.mark.parametrize("value", [[], "manifest", 3])
def test_preflight_reports_wrong_shaped_v1_manifest_without_crashing(tmp_path, value):
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    result = check_assets(
        tmp_path,
        require_samples=False,
        review=tmp_path / "missing-review.json",
        catalog_id="catalog-v1",
    )
    assert not result["ready"]
    assert "Invalid manifest.json structure" in result["errors"]


def test_preflight_reports_wrong_shaped_v2_manifest_without_crashing(asset_tree):
    (asset_tree["root"] / "catalog-v2.json").write_text("[]")
    result = result_for(asset_tree, catalog_id="catalog-v2")
    assert not result["ready"]
    assert "Invalid catalog-v2.json structure" in result["errors"]


def test_preflight_reports_wrong_shaped_generic_prompts_without_crashing(asset_tree):
    (asset_tree["root"] / "generic-prompts.json").write_text("[]")
    result = result_for(asset_tree)
    assert not result["ready"]
    assert "Invalid generic-prompts.json structure" in result["errors"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda sample: sample.update(id=[]), "invalid id"),
        (lambda sample: sample.update(images=None), "invalid images"),
        (lambda sample: sample.update(images=[None]), "invalid image record"),
        (lambda sample: sample.update(selection={}), "selection is unusable"),
        (lambda sample: sample.update(personalization=[]), "invalid personalization"),
    ],
)
def test_preflight_reports_malformed_nested_sample_records_without_crashing(
    asset_tree, mutate, message
):
    path = asset_tree["root"] / "samples.json"
    samples = json.loads(path.read_text())
    mutate(samples[0])
    path.write_text(json.dumps(samples))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert any(message in error for error in result["errors"])
