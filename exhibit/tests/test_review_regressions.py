import json

from exhibit.config import ASSETS
from exhibit.preflight import check_assets
from pigreward_repro.adapter import parse_judgment


def test_parser_rejects_ignored_malformed_rows_and_contradictory_decisions():
    good = "Composition: Balanced. (Score 1: 8, Score 2: 6)\nScore 1: 8\nScore 2: 6\nImage 1 is better"
    for extra in [
        "Lighting: Unclear. (Score 1: ??, Score 2: 4)\n",
        "Winner: Image 2\n",
    ]:
        assert (
            parse_judgment(extra + good, ["a", "b"], finished=True)["winner_id"] is None
        )


def test_preflight_rejects_sample_seed_settings_and_rewriter_changes(monkeypatch):
    from pathlib import Path

    original = Path.read_text
    for mutate in [
        lambda s: s["images"][0].update(seed=-123),
        lambda s: s["images"][0]["settings"].update(steps=1),
        lambda s: s["rewrite"]["model"].update(revision="wrong-revision"),
    ]:
        samples = json.loads(original(ASSETS / "samples.json"))
        mutate(samples[0])

        def read(path, *args, samples=samples, **kwargs):
            return (
                json.dumps(samples)
                if path == ASSETS / "samples.json"
                else original(path, *args, **kwargs)
            )

        with monkeypatch.context() as m:
            m.setattr(Path, "read_text", read)
            result = check_assets()
            assert not result["ready"]
            assert any("Sample" in e for e in result["errors"])


def test_sample_endpoint_rejects_wrong_seed_before_showing_images(
    tmp_path, monkeypatch
):
    import exhibit.service as module

    original = module.read_json
    samples = original(ASSETS / "samples.json")
    samples[0]["images"][0]["seed"] = -999

    def read(path, default=None):
        return samples if path == ASSETS / "samples.json" else original(path, default)

    monkeypatch.setattr(module, "read_json", read)
    service = module.Service(tmp_path)
    session = service.create_session()
    import pytest

    with pytest.raises(ValueError, match="inconsistent"):
        service.sample(session["id"], samples[0]["id"])
    assert service.snapshot(session["id"])["run"] is None
