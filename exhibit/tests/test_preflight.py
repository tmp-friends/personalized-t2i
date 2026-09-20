import json

from exhibit.config import CONFIG
from exhibit.domain import CARDS
from exhibit.preflight import check_assets

from exhibit import preflight


def result_for(tree, **kwargs):
    return check_assets(tree["root"], review=tree["review"], **kwargs)


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
