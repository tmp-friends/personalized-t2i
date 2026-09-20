import json

from exhibit import preflight
from exhibit.preflight import check_assets


def test_missing_assets_block_live_start_without_crashing(tmp_path):
    result = check_assets(tmp_path)
    assert not result["ready"]
    assert result["errors"]


def test_corrupt_assets_are_detected(tmp_path):
    from exhibit.config import CONFIG

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "generation": CONFIG["generation"],
                "images": {"p1-a": {"path": "bad.png", "sha256": "incorrect"}},
            }
        )
    )
    (tmp_path / "bad.png").write_bytes(b"not png")
    result = check_assets(tmp_path)
    assert not result["ready"]
    assert any("p1-a" in e for e in result["errors"])


def test_write_preflight_persists_the_fresh_model_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "check_assets",
        lambda: {"ready": True, "errors": [], "fixed_images": 34},
    )
    monkeypatch.setattr(
        preflight,
        "check_models",
        lambda: {
            "ready": True,
            "errors": [],
            "models": [{"model": "OnomaAIResearch/Illustrious-XL-v2.0"}],
        },
    )
    output = tmp_path / "preflight.json"

    result = preflight.write_preflight(output, include_models=True)

    assert result["ready"]
    assert json.loads(output.read_text()) == result
