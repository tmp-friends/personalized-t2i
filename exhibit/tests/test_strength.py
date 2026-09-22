"""Strength experiments: declared matrices, history pool, and their gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from exhibit.domain import digest
from exhibit.evaluation import (
    build_experiment,
    list_strength_experiments,
    load_evaluation_config,
    load_strength_config,
    register_experiment,
    select_candidates,
    summarize_records,
)

CONFIGS = Path(__file__).parents[1] / "configs"
EVALUATION_PATH = CONFIGS / "fan-evaluation.json"
STRENGTH_PATH = CONFIGS / "fan-strength.json"

# topics x histories x seeds x (1 plain-per-seed + legacy + one per policy)
EXPECTED = {
    "e1-settings": {
        "policies": 11,
        "topics": 3,
        "histories": 4,
        "seeds": 2,
        "jobs": 294,
    },
    "e2-adapter": {"policies": 6, "topics": 3, "histories": 4, "seeds": 2, "jobs": 174},
    "e3-references": {
        "policies": 2,
        "topics": 3,
        "histories": 9,
        "seeds": 2,
        "jobs": 168,
    },
    "e5-official-sampler": {
        "policies": 2,
        "topics": 3,
        "histories": 4,
        "seeds": 2,
        "jobs": 78,
    },
}


@pytest.fixture
def provenance():
    return {
        "fan_pin": "9d0b76843f6437718195accac9cf3f050a25d26b",
        "fan_source": {"model.py": "fan-model", "wrapper.py": "fan-wrapper"},
        "patches": {},
        "adapter_hash": "adapter",
        "worker_hash": "worker",
        "evaluation_hash": "evaluation",
        "controller_hash": "controller",
        "decoder_hash": {"L.pth": "l", "bigG.pth": "g"},
        "tokenizer_hash": {"clip_l": "tl", "clip_g": "tg"},
        "model_hash": "model",
        "evaluator_hash": "evaluator",
        "environment": {"python": "3.10", "torch": "test"},
    }


def _strength(experiment_id):
    return load_strength_config(STRENGTH_PATH, experiment_id)


def _roles(experiment):
    roles = {}
    for job in experiment["jobs"]:
        roles.setdefault(job["role"], []).append(job)
    return roles


def test_every_declared_experiment_lists_its_description():
    declared = set(json.loads(STRENGTH_PATH.read_text())["experiments"])
    listed = list_strength_experiments(STRENGTH_PATH)

    # The set grows with every new experiment; the matrices below pin the four
    # that must always be there.
    assert set(listed) == declared
    assert declared >= set(EXPECTED)
    assert all(text for text in listed.values())


@pytest.mark.parametrize("experiment_id", sorted(EXPECTED))
def test_declared_experiments_build_a_fixed_plain_legacy_candidate_matrix(
    experiment_id, provenance
):
    expected = EXPECTED[experiment_id]
    config = _strength(experiment_id)

    assert config["phase"] == "strength"
    assert config["experiment_id"] == experiment_id
    assert len(config["policies"]) == expected["policies"]
    assert len(config["topics"]) == expected["topics"]
    assert len(config["histories"]) == expected["histories"]
    assert len(config["seeds"]) == expected["seeds"]

    experiment = build_experiment(config, provenance)
    roles = _roles(experiment)

    assert len(roles["plain"]) == expected["topics"] * expected["seeds"]
    assert len(roles["legacy"]) == (
        expected["topics"] * expected["histories"] * expected["seeds"]
    )
    assert len(roles["candidate"]) == (
        expected["policies"]
        * expected["topics"]
        * expected["histories"]
        * expected["seeds"]
    )
    assert len(experiment["jobs"]) == expected["jobs"]
    assert len({job["job_id"] for job in experiment["jobs"]}) == expected["jobs"]
    assert experiment["experiment_hash"] == digest(experiment["identity"])
    # Every candidate is comparable to the legacy and plain image of its cell.
    for job in roles["candidate"]:
        assert job["plain_job_id"] and job["legacy_job_id"]


def test_a_generation_override_makes_the_experiment_diagnostic(provenance):
    demo = load_evaluation_config(EVALUATION_PATH, "screen")["generation"]
    settings = _strength("e1-settings")
    sampler = _strength("e5-official-sampler")

    assert settings["experiment_kind"] == "policy"
    assert settings["rules"]["binding"] is True
    assert settings["generation"] == demo
    assert settings["generation_override"] == {}

    assert sampler["experiment_kind"] == "generation"
    assert sampler["rules"]["binding"] is False
    assert sampler["generation_override"] == {
        "negative_prompt": "",
        "steps": 50,
        "scheduler_kwargs": {
            "algorithm_type": "dpmsolver++",
            "use_karras_sigmas": True,
        },
    }
    assert sampler["generation"]["steps"] == 50
    assert sampler["generation"]["negative_prompt"] == ""
    assert sampler["generation"]["model"] == demo["model"]

    experiment = build_experiment(sampler, provenance)
    assert experiment["identity"]["experiment_kind"] == "generation"
    assert experiment["display"] == {
        "experiment_id": "e5-official-sampler",
        "experiment_kind": "generation",
        "description": sampler["description"],
        "plan_items": ["A6"],
        "generation_override": sampler["generation_override"],
    }
    # The override reaches the identity, so it addresses its own directory.
    assert experiment["identity"]["generation"]["steps"] == 50
    assert (
        experiment["experiment_hash"]
        != build_experiment(_strength("e1-settings"), provenance)["experiment_hash"]
    )


def _write_strength(tmp_path, spec, **top):
    document = {
        "schema_version": 1,
        "evaluation_config": str(EVALUATION_PATH),
        "output_root": "out",
        "experiments": {"one": spec},
        **top,
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "strength.json"
    path.write_text(json.dumps(document))
    return path


def _spec(**overrides):
    return {
        "topics": ["cat"],
        "histories": ["mixed"],
        "seeds": [1],
        "policies": {"strong_v1": "strong_v1"},
        **overrides,
    }


def test_unknown_experiments_histories_and_repeated_policies_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown strength experiment"):
        load_strength_config(STRENGTH_PATH, "e4-missing")

    unknown_history = _write_strength(tmp_path / "a", _spec(histories=["nope"]))
    with pytest.raises(ValueError, match="unknown history"):
        load_strength_config(unknown_history, "one")

    legacy = _write_strength(
        tmp_path / "b", _spec(policies={"legacy_exhibit": "legacy_exhibit"})
    )
    with pytest.raises(ValueError, match="legacy_exhibit is compared implicitly"):
        load_strength_config(legacy, "one")

    registry = json.loads((CONFIGS / "fan-policies.json").read_text())
    duplicate = _write_strength(
        tmp_path / "c",
        _spec(
            policies={
                "strong_v1": "strong_v1",
                "same-by-value": copy.deepcopy(registry["policies"]["strong_v1"]),
            }
        ),
    )
    with pytest.raises(ValueError, match="repeats an effective policy"):
        load_strength_config(duplicate, "one")


def test_heldout_histories_are_reachable_beside_the_screen_ids_that_repeat_them(
    tmp_path,
):
    path = _write_strength(tmp_path, _spec(histories=["mixed", "heldout:mixed"]))
    config = load_strength_config(path, "one")

    screen, heldout = config["histories"]
    assert screen["id"] == "mixed"
    assert heldout["id"] == "heldout:mixed"

    fixtures = json.loads(EVALUATION_PATH.read_text())["fixtures"]
    expected = {
        name: next(
            item["refs"]
            for item in json.loads((CONFIGS / fixtures[name]).read_text())["histories"]
            if item["id"] == "mixed"
        )
        for name in ("screen_histories", "heldout_histories")
    }
    assert screen["refs"] == expected["screen_histories"]
    assert heldout["refs"] == expected["heldout_histories"]
    assert screen["refs"] != heldout["refs"]


def _candidate_rows(policy_hash, history_delta, target_delta, seconds=1.0, count=2):
    return [
        {
            "job_id": f"{policy_hash}-{index}",
            "role": "candidate",
            "policy_hash": policy_hash,
            "status": "measured",
            "topic_id": "cat",
            "history_id": f"h{index}",
            "seed": index,
            "target_score": 0.5,
            "history_score": 0.5,
            "delta_vs_legacy": {
                "history_score": history_delta,
                "target_score": target_delta,
            },
            "delta_vs_plain": {
                "history_score": history_delta,
                "target_score": target_delta,
            },
            "seconds": seconds,
        }
        for index in range(count)
    ]


def _rules(*policy_hashes):
    return {
        "stage": "strength",
        "expected_cases": 2,
        "target_non_degradation": -0.01,
        "history_improvement": 0.005,
        "numerical": {
            key: {"passed": True, "evidence_hash": "encoding"} for key in policy_hashes
        },
    }


def test_strength_selection_requires_both_gates_and_keeps_every_survivor():
    names = ("winner", "runner-up", "third", "weak-history", "target-loss")
    rows = [
        *_candidate_rows("winner", 0.04, 0.0),
        *_candidate_rows("runner-up", 0.03, 0.0),
        *_candidate_rows("third", 0.02, 0.0),
        *_candidate_rows("weak-history", 0.004, 0.0),
        *_candidate_rows("target-loss", 0.20, -0.02),
    ]

    decision = select_candidates(rows, _rules(*names))

    by_hash = {item["policy_hash"]: item for item in decision["candidates"]}
    assert by_hash["weak-history"]["status"] == "fail"
    assert by_hash["weak-history"]["reasons"] == ["history_improvement"]
    assert by_hash["target-loss"]["status"] == "fail"
    assert by_hash["target-loss"]["reasons"] == ["target_non_degradation"]
    # Screening keeps two; a strength sweep reports every policy that passed.
    assert [item["policy_hash"] for item in decision["selected"]] == [
        "winner",
        "runner-up",
        "third",
    ]
    assert decision["status"] == "selected"
    assert decision["decision_hash"] == digest(
        {key: value for key, value in decision.items() if key != "decision_hash"}
    )


def test_strength_selection_keeps_legacy_when_no_policy_clears_both_gates():
    decision = select_candidates(_candidate_rows("weak", 0.004, 0.0), _rules("weak"))

    assert decision["status"] == "keep_legacy"
    assert decision["selected"] == []


def test_strength_summaries_carry_history_clustered_bootstrap_intervals():
    rows = [
        *_candidate_rows("winner", 0.04, 0.0),
        *_candidate_rows("runner-up", 0.01, 0.0),
    ]

    summary = summarize_records(rows, "strength", {"bootstrap_draws": 200, "seed": 0})

    intervals = summary["policies"]["winner"]["bootstrap_95"]
    assert set(intervals) == {"target_delta_vs_legacy", "history_delta_vs_legacy"}
    assert intervals["history_delta_vs_legacy"]["mean"] == pytest.approx(0.04)
    assert summary["stage"] == "strength"


def test_registering_a_strength_experiment_records_its_display_and_resumes(
    tmp_path, provenance
):
    config = _strength("e5-official-sampler")
    experiment = build_experiment(config, provenance)

    directory, checkpoint = register_experiment(tmp_path, experiment, resume=False)
    manifest = json.loads((directory / "manifest.json").read_text())

    assert directory == tmp_path / experiment["experiment_hash"]
    assert manifest["display"] == experiment["display"]
    assert manifest["display"]["experiment_kind"] == "generation"
    assert digest(manifest["identity"]) == experiment["experiment_hash"]
    assert len(checkpoint["jobs"]) == len(experiment["jobs"])

    again, resumed = register_experiment(tmp_path, experiment, resume=True)
    assert again == directory
    assert set(resumed["jobs"]) == set(checkpoint["jobs"])

    # A different experiment cannot adopt this directory's results.
    other = build_experiment(_strength("e3-references"), provenance)
    with pytest.raises(ValueError, match="resume manifest hash or identity mismatch"):
        register_experiment(directory, other, resume=True)


def test_a_policy_only_experiment_carries_no_generation_override_in_its_display(
    provenance,
):
    experiment = build_experiment(_strength("e2-adapter"), provenance)

    assert experiment["display"]["experiment_kind"] == "policy"
    assert experiment["display"]["generation_override"] == {}
    assert experiment["display"]["plan_items"] == ["C1", "C2"]
    assert experiment["identity"]["experiment_kind"] == "policy"
    gains = [
        item["effective_policy"].get("embed_gain")
        for item in experiment["identity"]["policies"]
    ]
    assert sorted(value for value in gains if value is not None) == [1.5, 1.5, 2.0, 2.5]
