import json

import pytest
from exhibit.config import CONFIG
from exhibit.preflight import check_assets

from exhibit import preflight


def result_for(tree, **kwargs):
    review = kwargs.pop("review", tree["review"])
    return check_assets(tree["root"], review=review, **kwargs)


def test_missing_assets_block_live_start_without_crashing(tmp_path):
    result = check_assets(tmp_path, review=tmp_path / "absent.json")
    assert not result["ready"]
    assert any("manifest.json" in e for e in result["errors"])
    assert "No reviewed cards" in result["errors"]


def test_a_complete_bundle_is_ready(asset_tree):
    result = result_for(asset_tree)
    assert result["ready"], result["errors"]
    assert result["catalog_id"] == "catalog-v2"
    assert result["cards"] == result["reviewed_cards"] == result["fixed_images"] == 64
    assert result["generic_images"] == len(CONFIG["topics"]) * len(CONFIG["seeds"])
    assert result["samples"] == 6
    assert result["mode"] == "fan-live"


def test_unreviewed_cards_are_reported_but_do_not_block(asset_tree):
    review = json.loads(asset_tree["review"].read_text())
    samples = (asset_tree["root"] / "samples.json").read_text()
    stale = next(card for card in review if card not in samples)
    review[stale]["reviewed"] = False
    asset_tree["review"].write_text(json.dumps(review))
    result = result_for(asset_tree)
    assert result["ready"], result["errors"]
    assert result["warnings"] == [f"Unreviewed card: {stale}"]
    assert result["reviewed_cards"] == result["cards"] - 1


def test_no_reviewed_cards_fail_preflight(asset_tree):
    review = json.loads(asset_tree["review"].read_text())
    for entry in review.values():
        entry["reviewed"] = False
    asset_tree["review"].write_text(json.dumps(review))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert "No reviewed cards" in result["errors"]


def test_corrupt_and_stale_images_are_detected(asset_tree):
    card = asset_tree["cards"][0]["id"]
    card_path = asset_tree["root"] / "catalog-v2.json"
    cards = json.loads(card_path.read_text())
    cards["images"][card]["sha256"] = "incorrect"
    card_path.write_text(json.dumps(cards))
    manifest_path = asset_tree["root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"]["cat-0"]["seed"] = -1
    manifest["images"]["cat-1"]["prompt"] = "a different sentence"
    manifest_path.write_text(json.dumps(manifest))
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


def test_preflight_rejects_a_tampered_card_contract(asset_tree):
    card = asset_tree["cards"][0]["id"]
    path = asset_tree["root"] / "catalog-v2.json"
    manifest = json.loads(path.read_text())
    manifest["images"][card]["aspects_ja"]["color"] = "改ざん"
    path.write_text(json.dumps(manifest))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert f"Card contract mismatch: {card}" in result["errors"]


def test_samples_must_match_the_personalization_they_claim(asset_tree):
    samples = json.loads((asset_tree["root"] / "samples.json").read_text())
    samples[0]["personalization"]["hash"] = "rewritten"
    samples[1]["images"][0]["seed"] = -5
    samples[2]["preference"]["selection"] = [
        {"card_id": "unknown-card", "strength": 1, "aspects": ["color"]}
    ]
    samples[3]["personalization"]["provenance"]["fan_pin"] = "another-commit"
    (asset_tree["root"] / "samples.json").write_text(json.dumps(samples))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert any("personalization mismatch" in e for e in result["errors"])
    assert any("seed order" in e for e in result["errors"])
    assert any("preference is unusable" in e for e in result["errors"])
    assert any("provenance is not the current contract" in e for e in result["errors"])


def test_a_sample_of_an_unreviewed_card_is_refused(asset_tree):
    samples = json.loads((asset_tree["root"] / "samples.json").read_text())
    dropped = samples[0]["preference"]["selection"][0]["card_id"]
    review = json.loads(asset_tree["review"].read_text())
    review[dropped]["reviewed"] = False
    asset_tree["review"].write_text(json.dumps(review))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert any("preference is unusable" in e for e in result["errors"])


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


def test_sample_manifest_declares_required_ids_and_rejects_deletion(
    asset_tree, monkeypatch
):
    samples_path = asset_tree["root"] / "samples.json"
    samples = json.loads(samples_path.read_text())
    monkeypatch.setattr(
        preflight,
        "CONFIG",
        {**CONFIG, "sample_ids": [sample["id"] for sample in samples]},
    )
    samples_path.write_text(json.dumps(samples[:-1]))
    result = result_for(asset_tree)
    assert not result["ready"]
    assert "Sample manifest IDs mismatch" in result["errors"]


@pytest.mark.parametrize("value", [[], "manifest", 3])
def test_preflight_reports_wrong_shaped_manifest_without_crashing(tmp_path, value):
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    result = check_assets(
        tmp_path, require_samples=False, review=tmp_path / "missing-review.json"
    )
    assert not result["ready"]
    assert "Invalid manifest.json structure" in result["errors"]


def test_preflight_reports_wrong_shaped_catalog_manifest_without_crashing(asset_tree):
    (asset_tree["root"] / "catalog-v2.json").write_text("[]")
    result = result_for(asset_tree)
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
        (lambda sample: sample.update(preference={}), "preference is unusable"),
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
