from exhibit.preflight import check_assets


def test_missing_assets_block_live_start_without_crashing(tmp_path):
    result = check_assets(tmp_path)
    assert not result["ready"]
    assert result["errors"]


def test_corrupt_assets_are_detected(tmp_path):
    import json

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
