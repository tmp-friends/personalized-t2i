import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from exhibit.catalog import build_catalog
from exhibit.config import CONFIG, ROOT, read_json
from exhibit.domain import digest, file_hash


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v2_cards():
    return build_catalog(read_json(ROOT / "configs/catalog-v2.json"))


def fake_provenance():
    return {
        "repo_id": CONFIG["generation"]["pipeline_config"]["model"],
        "revision": CONFIG["generation"]["pipeline_config"]["revision"],
        "files": {
            f"{tokenizer}/{filename}": "a" * 64
            for tokenizer in ("tokenizer", "tokenizer_2")
            for filename in (
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "special_tokens_map.json",
            )
        },
    }


class FakeTokenizer:
    def __init__(self, count=3):
        self.ids = [49406, *range(max(0, count - 2)), 49407]

    def __call__(self, prompt, *, add_special_tokens):
        assert add_special_tokens is True
        return {"input_ids": self.ids}


def report_for(cards, *, overflow=False):
    tokenizers = {
        "tokenizer": FakeTokenizer(78 if overflow else 3),
        "tokenizer_2": FakeTokenizer(3),
    }
    script = load_script("check_card_tokens")
    return script.build_token_report(
        "catalog-v2",
        cards,
        tokenizers,
        fake_provenance(),
        ROOT / "configs/legacy-card-token-overflow.json",
    )


def test_runtime_tokenizer_provenance_hashes_all_pinned_tokenizer_files(tmp_path):
    from exhibit.evaluation import runtime_tokenizer_provenance

    requested = []

    def download(repo_id, *, filename, revision, local_files_only):
        requested.append((repo_id, filename, revision, local_files_only))
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())
        return str(path)

    result = runtime_tokenizer_provenance(CONFIG["generation"], download=download)
    expected_files = {
        f"{tokenizer}/{filename}"
        for tokenizer in ("tokenizer", "tokenizer_2")
        for filename in (
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    }
    assert set(result["files"]) == expected_files
    assert result["repo_id"] == CONFIG["generation"]["pipeline_config"]["model"]
    assert result["revision"] == CONFIG["generation"]["pipeline_config"]["revision"]
    assert {item[1] for item in requested} == expected_files
    assert all(item[3] is True for item in requested)
    assert all(len(value) == 64 for value in result["files"].values())


def test_token_report_binds_prompts_generation_special_ids_files_and_legacy_evidence():
    cards = v2_cards()[:2]
    report = report_for(cards)
    assert report["schema_version"] == 1
    assert report["catalog_id"] == "catalog-v2"
    assert report["generation"] == CONFIG["generation"]
    assert report["prompt_set_hash"] == digest(
        [{"id": card["id"], "prompt": card["prompt"]} for card in cards]
    )
    assert report["tokenizers"] == fake_provenance()
    assert len(report["results"]) == 4
    assert all(row["special_tokens"] is True for row in report["results"])
    assert report["results"][0]["token_ids"][0] == 49406
    assert report["results"][0]["token_ids"][-1] == 49407
    assert report["results"][0]["prompt_hash"] == digest(cards[0]["prompt"])
    evidence = report["legacy_overflow_evidence"]
    assert evidence["over_limit_ids"] == [
        "girl-warm_soft",
        "student-warm_soft",
        "barista-warm_soft",
    ]
    assert evidence["sha256"] == file_hash(
        ROOT / "configs/legacy-card-token-overflow.json"
    )


def test_legacy_three_card_overflow_evidence_remains_frozen():
    rows = read_json(ROOT / "configs/legacy-card-token-overflow.json")
    for tokenizer in rows:
        assert [item["id"] for item in tokenizer["over_limit"]] == [
            "girl-warm_soft",
            "student-warm_soft",
            "barista-warm_soft",
        ]


def _configure_prepare(module, tmp_path, monkeypatch, *, overflow):
    assets = tmp_path / "assets"
    outputs = tmp_path / "outputs"
    assets.mkdir()
    legacy = assets / "manifest.json"
    legacy.write_bytes(b"frozen v1 manifest")
    monkeypatch.setattr(module, "ASSETS", assets)
    monkeypatch.setattr(module, "OUTPUTS", outputs)
    cards = v2_cards()

    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report_for(cards, overflow=overflow)))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    return assets, outputs, legacy, cards


def test_prepare_v2_blocks_generation_when_report_contains_overflow(
    tmp_path, monkeypatch
):
    module = load_script("prepare")
    assets, _, legacy, _ = _configure_prepare(
        module, tmp_path, monkeypatch, overflow=True
    )
    calls = []
    monkeypatch.setattr(
        module, "stage", lambda *args, **kwargs: calls.append(args) or []
    )
    with pytest.raises(RuntimeError, match="token validation"):
        module.prepare_cards("v2")
    assert calls == []
    assert legacy.read_bytes() == b"frozen v1 manifest"
    assert not (assets / "catalog-v2.json").exists()


def test_prepare_v2_writes_only_v2_manifest_with_bound_token_report(
    tmp_path, monkeypatch
):
    module = load_script("prepare")
    assets, _, legacy, cards = _configure_prepare(
        module, tmp_path, monkeypatch, overflow=False
    )

    def stage(request, name):
        events = []
        for item in request["items"]:
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["id"].encode())
            events.append(
                {
                    "type": "image",
                    "id": item["id"],
                    "path": str(path),
                    "sha256": file_hash(path),
                    "seed": item["seed"],
                    "prompt": item["prompt"],
                    "settings": CONFIG["generation"],
                }
            )
        return events

    monkeypatch.setattr(module, "stage", stage)
    module.prepare_cards("v2")
    manifest = json.loads((assets / "catalog-v2.json").read_text())
    assert legacy.read_bytes() == b"frozen v1 manifest"
    assert manifest["version"] == 2
    assert manifest["catalog_id"] == "catalog-v2"
    assert len(manifest["images"]) == len(cards) == 64
    assert manifest["token_validation"]["prompt_set_hash"] == digest(
        [{"id": card["id"], "prompt": card["prompt"]} for card in cards]
    )
    first = manifest["images"][cards[0]["id"]]
    assert first["aspects_ja"] == cards[0]["aspects_ja"]
    assert first["label"] == cards[0]["label"]
