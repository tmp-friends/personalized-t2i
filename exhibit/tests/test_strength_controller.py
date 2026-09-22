"""CPU-only tests for the `strength` subcommand of scripts/evaluate_fan.py."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

from exhibit.config import ROOT

STRENGTH_CONFIG_PATH = Path(__file__).parents[1] / "configs/fan-strength.json"


def _load_module(name):
    script = ROOT / "scripts/evaluate_fan.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_covers_every_declared_strength_experiment(capsys):
    module = _load_module("test_strength_list_evaluate_fan")

    module.main(["strength", "--config", str(STRENGTH_CONFIG_PATH), "--list"])

    lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    # New experiments keep being declared, so the listing is compared against
    # the config itself; the first four must never disappear from it.
    declared = set(json.loads(STRENGTH_CONFIG_PATH.read_text())["experiments"])
    assert {item["experiment_id"] for item in lines} == declared
    assert declared >= {
        "e1-settings",
        "e2-adapter",
        "e3-references",
        "e5-official-sampler",
    }
    assert all(item["description"] for item in lines)


def test_run_strength_wires_diagnostics_and_execution(monkeypatch):
    module = _load_module("test_strength_run_evaluate_fan")

    captured = {}

    def fake_validate_evaluator_preparation(path, *, expected=None):
        return {"evaluator_hash": "test-evaluator-hash"}

    def fake_ensure_diagnostics(config_path, config, preparation, cancel, deadline, on_event):
        captured["config_path"] = config_path
        captured["config"] = config
        return {"report": "stub"}

    def fake_experiment_provenance(report, preparation):
        return {"provenance": "stub"}

    def fake_numerical_evidence(report, policies):
        return {
            item["policy_hash"]: {"passed": True, "evidence_hash": "stub"}
            for item in policies
        }

    def fake_execute_experiment(config, provenance, *, resume, cancel, deadline, on_event):
        captured["execute_config"] = config
        captured["execute_provenance"] = provenance
        return {
            "experiment_hash": "stub-hash",
            "directory": "stub-directory",
            "decision": {"status": "keep_legacy"},
        }

    monkeypatch.setattr(
        module, "validate_evaluator_preparation", fake_validate_evaluator_preparation
    )
    monkeypatch.setattr(module, "ensure_diagnostics", fake_ensure_diagnostics)
    monkeypatch.setattr(module, "experiment_provenance", fake_experiment_provenance)
    monkeypatch.setattr(module, "numerical_evidence", fake_numerical_evidence)
    monkeypatch.setattr(module, "execute_experiment", fake_execute_experiment)

    result = module.run_strength(
        str(STRENGTH_CONFIG_PATH),
        "e1-settings",
        resume=False,
        cancel=threading.Event(),
        deadline=time.monotonic() + 10,
        on_event=lambda event: None,
    )

    assert captured["config_path"].endswith("fan-evaluation.json")
    assert captured["config"]["phase"] == "strength"
    assert captured["config"]["experiment_id"] == "e1-settings"
    assert captured["config"]["policies"]

    execute_config = captured["execute_config"]
    assert execute_config is captured["config"]
    policy_hashes = {item["policy_hash"] for item in execute_config["policies"]}
    assert policy_hashes
    assert set(execute_config["numerical"]) == policy_hashes
    assert captured["execute_provenance"] == {"provenance": "stub"}
    assert result == {
        "experiment_hash": "stub-hash",
        "directory": "stub-directory",
        "decision": {"status": "keep_legacy"},
    }
