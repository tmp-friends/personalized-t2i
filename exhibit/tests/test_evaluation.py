"""Reproducible FAN evaluation matrices, checkpoints, scoring, and selection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from exhibit.domain import digest, file_hash
from exhibit.evaluation import (
    EVALUATOR_FILES,
    attach_parent_pairs,
    build_experiment,
    cluster_bootstrap_interval,
    load_evaluation_config,
    prepare_evaluator,
    register_experiment,
    score_records,
    select_candidates,
    summarize_records,
    validate_evaluator_preparation,
    validate_heldout_catalog,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs/fan-evaluation.json"


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


def _phase(name, parents=None):
    return load_evaluation_config(CONFIG_PATH, name, parents=parents or {})


def test_config_wires_eight_named_prompts_and_five_named_histories():
    encoding = _phase("encoding")

    assert len(encoding["prompts"]) == 8
    assert len({item["id"] for item in encoding["prompts"]}) == 8
    assert len(encoding["histories"]) == 5
    assert len({item["id"] for item in encoding["histories"]}) == 5
    assert all(item["refs"] for item in encoding["histories"])
    assert len(encoding["policies"]) == 9  # legacy plus the eight screen policies


@pytest.mark.parametrize(
    ("phase", "parents", "expected"),
    [
        ("screen", {}, 222),
        (
            "refine",
            {
                "screen": {
                    "manifest_hash": "screen-manifest",
                    "decision_hash": "screen-decision",
                    "selected_policies": [],
                }
            },
            96,
        ),
        (
            "heldout",
            {
                "refine": {
                    "manifest_hash": "refine-manifest",
                    "decision_hash": "refine-decision",
                    "selected_policies": [],
                }
            },
            156,
        ),
    ],
)
def test_job_counts_are_fixed_before_generation(phase, parents, expected, provenance):
    selected = copy.deepcopy(_phase("screen")["policies"][:2])
    if phase == "refine":
        parents["screen"]["selected_policies"] = selected
    elif phase == "heldout":
        parents["refine"]["selected_policies"] = selected
    config = _phase(phase, parents)
    experiment = build_experiment(config, provenance)

    assert len(experiment["jobs"]) == expected
    assert len({job["job_id"] for job in experiment["jobs"]}) == expected
    assert len({job["path"] for job in experiment["jobs"]}) == expected
    assert experiment["experiment_hash"] == digest(experiment["identity"])


def test_more_than_512_jobs_are_rejected_before_generation(provenance):
    config = _phase("screen")
    config["seeds"] = list(range(10))

    with pytest.raises(ValueError, match="512"):
        build_experiment(config, provenance)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_images", 513),
        ("max_images", True),
        ("max_attempts", 3),
        ("max_attempts", False),
    ],
)
def test_experiment_rejects_limits_above_fixed_budgets(provenance, name, value):
    config = _phase("screen")
    config["limits"][name] = value

    with pytest.raises(ValueError, match=name):
        build_experiment(config, provenance)


def test_resume_on_an_unregistered_experiment_starts_it_and_stays_strict(
    tmp_path, provenance
):
    experiment = build_experiment(_phase("screen"), provenance)

    directory, checkpoint = register_experiment(tmp_path, experiment, resume=True)

    assert directory == tmp_path / experiment["experiment_hash"]
    assert {state["status"] for state in checkpoint["jobs"].values()} == {"not_run"}
    # The documented command is always `--resume`; a second call resumes it.
    again, _ = register_experiment(tmp_path, experiment, resume=True)
    assert again == directory
    # Half a registration is corruption, never a fresh start.
    (directory / "checkpoint.json").unlink()
    with pytest.raises(ValueError, match="registered manifest and checkpoint"):
        register_experiment(tmp_path, experiment, resume=True)


def test_effective_policy_change_changes_identity_and_refuses_resume(
    tmp_path, provenance
):
    first = build_experiment(_phase("screen"), provenance)
    register_experiment(tmp_path, first, resume=False)
    changed_config = _phase("screen")
    changed_config["policies"][0]["effective_policy"]["alpha"] = 0.41
    del changed_config["policies"][0]["policy_hash"]
    changed = build_experiment(changed_config, provenance)

    assert changed["experiment_hash"] != first["experiment_hash"]
    with pytest.raises(ValueError, match="manifest hash"):
        register_experiment(tmp_path / first["experiment_hash"], changed, resume=True)


def test_label_only_alias_does_not_change_effective_identity(provenance):
    first_config = _phase("screen")
    second_config = copy.deepcopy(first_config)
    second_config["policies"][0]["policy_id"] = "renamed-display-alias"

    first = build_experiment(first_config, provenance)
    second = build_experiment(second_config, provenance)

    assert first["experiment_hash"] == second["experiment_hash"]
    assert first["identity"] == second["identity"]


def test_resume_reuses_only_byte_verified_completed_jobs(tmp_path, provenance):
    experiment = build_experiment(_phase("screen"), provenance)
    directory, checkpoint = register_experiment(tmp_path, experiment, resume=False)
    job = experiment["jobs"][0]
    image = directory / job["path"]
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"first image")
    checkpoint["jobs"][job["job_id"]].update(
        {
            "status": "done",
            "attempts": 1,
            "sha256": file_hash(image),
            "contract_hash": job["contract_hash"],
            "event": {
                "id": job["job_id"],
                "path": str(image),
                "sha256": file_hash(image),
                "seed": job["seed"],
                "prompt": job["prompt"],
                "settings": experiment["identity"]["generation"],
                "effective_policy": job["effective_policy"],
                "policy_hash": job["policy_hash"],
                "personalization_hash": None,
            },
        }
    )
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))

    _, resumed = register_experiment(directory, experiment, resume=True)
    assert resumed["jobs"][job["job_id"]]["status"] == "done"

    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        register_experiment(directory, experiment, resume=True)


def test_resume_rejects_missing_or_mismatched_worker_event_contract(
    tmp_path, provenance
):
    experiment = build_experiment(_phase("screen"), provenance)
    directory, checkpoint = register_experiment(tmp_path, experiment, resume=False)
    job = experiment["jobs"][0]
    image = directory / job["path"]
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"validated image")
    state = checkpoint["jobs"][job["job_id"]]
    state.update(
        {
            "status": "done",
            "attempts": 1,
            "sha256": file_hash(image),
            "contract_hash": job["contract_hash"],
        }
    )
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="event contract"):
        register_experiment(directory, experiment, resume=True)

    state["event"] = {
        "id": job["job_id"],
        "path": str(image),
        "sha256": file_hash(image),
        "seed": job["seed"] + 1,
        "prompt": job["prompt"],
        "settings": experiment["identity"]["generation"],
        "effective_policy": job["effective_policy"],
        "policy_hash": job["policy_hash"],
        "personalization_hash": None,
    }
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="event contract"):
        register_experiment(directory, experiment, resume=True)


def test_resume_keeps_failures_visible_and_caps_attempts_at_two(tmp_path, provenance):
    experiment = build_experiment(_phase("screen"), provenance)
    directory, checkpoint = register_experiment(tmp_path, experiment, resume=False)
    job_id = experiment["jobs"][0]["job_id"]
    checkpoint["jobs"][job_id].update(
        {"status": "failed", "attempts": 2, "error": "worker_exit_1"}
    )
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))

    _, resumed = register_experiment(directory, experiment, resume=True)

    assert resumed["jobs"][job_id] == {
        "status": "failed",
        "attempts": 2,
        "error": "worker_exit_1",
    }


def _scoring_manifest():
    refs = [
        {"ref_id": "warm", "text": "warm", "weight": 2.0},
        {"ref_id": "cool", "text": "cool", "weight": 1.0},
    ]
    plain = {
        "job_id": "plain",
        "role": "plain",
        "topic_id": "cat",
        "history_id": None,
        "seed": 1,
        "target_text_id": "target:cat",
        "policy_hash": "legacy",
        "path": "images/plain.png",
        "contract_hash": "plain-contract",
    }
    legacy = {
        "job_id": "legacy",
        "role": "legacy",
        "topic_id": "cat",
        "history_id": "mixed",
        "seed": 1,
        "target_text_id": "target:cat",
        "refs": refs,
        "policy_hash": "legacy",
        "plain_job_id": "plain",
        "legacy_job_id": "legacy",
        "path": "images/legacy.png",
        "contract_hash": "legacy-contract",
    }
    candidate = {
        **legacy,
        "job_id": "candidate",
        "role": "candidate",
        "policy_hash": "candidate-policy",
        "legacy_job_id": "legacy",
        "path": "images/candidate.png",
        "contract_hash": "candidate-contract",
    }
    other = {
        **candidate,
        "job_id": "other",
        "policy_hash": "other-policy",
        "path": "images/other.png",
        "contract_hash": "other-contract",
    }
    return {"identity": {"jobs": [plain, legacy, candidate, other]}}


def _embeddings():
    return {
        "images": {
            "plain": {"embedding": [0.0, 1.0], "valid": True, "sha256": "p"},
            "legacy": {"embedding": [0.0, 1.0], "valid": True, "sha256": "l"},
            "candidate": {
                "embedding": [1.0, 0.0],
                "valid": True,
                "sha256": "c",
            },
            "other": {"embedding": [0.0, 1.0], "valid": True, "sha256": "o"},
        },
        "texts": {
            "target:cat": [1.0, 0.0],
            "ref:warm": [1.0, 0.0],
            "ref:cool": [-1.0, 0.0],
        },
        "conditioning": {
            "candidate": {
                "hidden": {"cosine": 0.9, "valid": True},
                "pooled": {"cosine": 0.8, "valid": True},
            }
        },
    }


def test_history_score_uses_all_preprofiling_references_and_reverses_selected_only():
    embeddings = _embeddings()
    embeddings["images"]["candidate"]["selected_ref_ids"] = ["cool"]

    records = score_records(_scoring_manifest(), embeddings)
    candidate = next(item for item in records if item["job_id"] == "candidate")
    other = next(item for item in records if item["job_id"] == "other")

    assert candidate["history_score"] == pytest.approx(1 / 3)
    assert other["history_score"] == 0
    assert candidate["history_score"] > other["history_score"]
    assert candidate["history_components"] == [
        {"ref_id": "warm", "weight": 2.0, "cosine": 1.0},
        {"ref_id": "cool", "weight": 1.0, "cosine": -1.0},
    ]


def test_scores_include_deltas_against_plain_and_legacy_without_copying_plain():
    records = score_records(_scoring_manifest(), _embeddings())
    candidate = next(item for item in records if item["job_id"] == "candidate")

    assert len([item for item in records if item["role"] == "plain"]) == 1
    assert candidate["delta_vs_plain"] == {
        "target_score": 1.0,
        "history_score": pytest.approx(1 / 3),
    }
    assert candidate["delta_vs_legacy"] == {
        "target_score": 1.0,
        "history_score": pytest.approx(1 / 3),
    }
    assert candidate["conditioning_target_align"] == {
        "hidden": 0.9,
        "pooled": 0.8,
    }


def test_missing_embeddings_stay_visible_as_unmeasured_records():
    embeddings = _embeddings()
    del embeddings["images"]["candidate"]

    records = score_records(_scoring_manifest(), embeddings)
    candidate = next(item for item in records if item["job_id"] == "candidate")

    assert candidate["status"] == "unmeasured"
    assert candidate["reasons"] == ["missing_image_embedding"]
    assert candidate["target_score"] is None
    assert candidate["history_score"] is None


def test_invalid_generated_image_fails_while_missing_image_is_unmeasured():
    embeddings = _embeddings()
    embeddings["images"]["candidate"] = {
        "valid": False,
        "invalid_reason": "all_zero_rgb",
        "sha256": "black",
    }
    del embeddings["images"]["other"]

    records = score_records(_scoring_manifest(), embeddings)
    candidates = [item for item in records if item["role"] == "candidate"]
    by_job = {item["job_id"]: item for item in candidates}
    assert by_job["candidate"]["status"] == "failed"
    assert by_job["candidate"]["reasons"] == ["all_zero_rgb"]
    assert by_job["other"]["status"] == "unmeasured"
    assert by_job["other"]["reasons"] == ["missing_image_embedding"]

    decision = select_candidates(
        candidates,
        {
            "stage": "screen",
            "expected_cases": 1,
            "target_non_degradation": -0.01,
            "numerical": {
                "candidate-policy": {"passed": True, "evidence_hash": "e"},
                "other-policy": {"passed": True, "evidence_hash": "e"},
            },
        },
    )
    by_policy = {item["policy_hash"]: item for item in decision["candidates"]}
    assert by_policy["candidate-policy"]["status"] == "fail"
    assert by_policy["candidate-policy"]["reasons"] == [
        "failed_case",
        "all_zero_rgb",
    ]
    assert by_policy["other-policy"]["status"] == "unmeasured"
    assert by_policy["other-policy"]["reasons"] == ["missing_image_embedding"]


def test_duplicate_artifact_reuse_for_distinct_contracts_is_rejected():
    manifest = _scoring_manifest()
    jobs = manifest["identity"]["jobs"]
    jobs[-1]["path"] = jobs[-2]["path"]

    with pytest.raises(ValueError, match="artifact"):
        score_records(manifest, _embeddings())


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
            "delta_vs_legacy": {
                "history_score": history_delta,
                "target_score": target_delta,
            },
            "seconds": seconds,
        }
        for index in range(count)
    ]


def test_selection_distinguishes_unmeasured_failed_and_ranked_candidates():
    rows = [
        *_candidate_rows("best", 0.02, -0.005, seconds=2.0),
        *_candidate_rows("tie-slower", 0.02, -0.005, seconds=3.0),
        *_candidate_rows("target-fail", 0.50, -0.02),
        {
            **_candidate_rows("missing", 0.5, 0.1, count=1)[0],
            "status": "unmeasured",
            "reasons": ["missing_image_embedding"],
        },
    ]
    rules = {
        "stage": "screen",
        "expected_cases": 2,
        "target_non_degradation": -0.01,
        "numerical": {
            key: {"passed": True, "evidence_hash": "encoding"}
            for key in ("best", "tie-slower", "target-fail", "missing")
        },
    }

    decision = select_candidates(rows, rules)

    assert decision["status"] == "selected"
    assert [item["policy_hash"] for item in decision["selected"]] == [
        "best",
        "tie-slower",
    ]
    by_hash = {item["policy_hash"]: item for item in decision["candidates"]}
    assert by_hash["target-fail"]["status"] == "fail"
    assert "target_non_degradation" in by_hash["target-fail"]["reasons"]
    assert by_hash["missing"]["status"] == "unmeasured"


def test_heldout_requires_history_gain_and_never_changes_default_policy():
    rows = _candidate_rows("candidate", 0.004, 0.0)
    registry = {"default_policy_id": "legacy_exhibit"}
    before = copy.deepcopy(registry)
    decision = select_candidates(
        rows,
        {
            "stage": "heldout",
            "expected_cases": 2,
            "target_non_degradation": -0.01,
            "history_improvement": 0.005,
            "numerical": {"candidate": {"passed": True, "evidence_hash": "encoding"}},
            "policy_registry": registry,
        },
    )

    assert decision["status"] == "keep_legacy"
    assert decision["candidates"][0]["status"] == "fail"
    assert "history_improvement" in decision["candidates"][0]["reasons"]
    assert registry == before


def test_numerical_evidence_is_required_for_the_exact_policy_hash():
    decision = select_candidates(
        _candidate_rows("effective-hash", 0.5, 0.1),
        {
            "stage": "screen",
            "expected_cases": 2,
            "target_non_degradation": -0.01,
            "numerical": {
                "generic-official": {"passed": True, "evidence_hash": "encoding"}
            },
        },
    )

    assert decision["status"] == "insufficient_candidates"
    assert decision["candidates"][0]["status"] == "unmeasured"
    assert decision["candidates"][0]["reasons"] == [
        "missing_exact_policy_numerical_evidence"
    ]


def test_heldout_bootstrap_resamples_four_history_clusters_deterministically():
    values = {
        "warm-cel": [0.01, 0.02],
        "cool-watercolor": [0.02, 0.03],
        "calm-flat": [-0.01, 0.00],
        "mixed": [0.03, 0.04],
    }

    first = cluster_bootstrap_interval(values, draws=2000, seed=0)
    second = cluster_bootstrap_interval(values, draws=2000, seed=0)

    assert first == second
    assert first["unit"] == "synthetic_history"
    assert first["draws"] == 2000
    assert first["confidence"] == 0.95
    assert first["lower"] <= first["mean"] <= first["upper"]


def test_preparation_hashes_every_consumed_file_and_pins_resolved_commit(tmp_path):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    names = {
        "config.json": b"config",
        "model.safetensors": b"weights",
        "preprocessor_config.json": b"processor",
        "tokenizer_config.json": b"tokenizer config",
        "tokenizer.json": b"tokenizer",
        "vocab.json": b"vocab",
        "merges.txt": b"merges",
        "special_tokens_map.json": b"special",
    }
    for name, payload in names.items():
        (snapshot / name).write_bytes(payload)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    output = tmp_path / "preparation.json"
    result = prepare_evaluator(
        {
            "repo_id": "openai/clip-vit-large-patch14",
            "revision": "main",
            "weight_format": "safetensors",
        },
        output,
        snapshot_download=snapshot_download,
        versions={"transformers": "test", "huggingface_hub": "test"},
    )

    assert result["resolved_revision"] == "a" * 40
    assert set(result["files"]) == set(names)
    assert all(
        result["files"][name]["sha256"] == file_hash(snapshot / name) for name in names
    )
    assert calls[0]["allow_patterns"] == sorted(names)
    assert calls[0]["ignore_patterns"] == ["pytorch_model.bin"]
    assert json.loads(output.read_text()) == result


def test_runtime_rejects_missing_incomplete_tampered_or_changed_preparation(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="not prepared"):
        validate_evaluator_preparation(missing)

    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    for name in EVALUATOR_FILES:
        (snapshot / name).write_bytes(name.encode())
    preparation = tmp_path / "preparation.json"
    result = prepare_evaluator(
        {
            "repo_id": "openai/clip-vit-large-patch14",
            "revision": "main",
            "expected_revision": "a" * 40,
            "weight_format": "safetensors",
        },
        preparation,
        snapshot_download=lambda **kwargs: str(snapshot),
        versions={"transformers": "test", "huggingface_hub": "test"},
    )
    assert (
        validate_evaluator_preparation(
            preparation,
            expected={
                "repo_id": "openai/clip-vit-large-patch14",
                "resolved_revision": "a" * 40,
                "weight_format": "safetensors",
            },
        )
        == result
    )

    incomplete = copy.deepcopy(result)
    incomplete["files"].pop("model.safetensors")
    preparation.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="required file"):
        validate_evaluator_preparation(preparation)

    preparation.write_text(json.dumps({**result, "evaluator_hash": "tampered"}))
    with pytest.raises(ValueError, match="identity hash"):
        validate_evaluator_preparation(preparation)

    preparation.write_text(json.dumps(result))
    (snapshot / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file hash"):
        validate_evaluator_preparation(preparation)


def test_encoding_worker_expands_every_named_prompt_history_pair():
    from exhibit.evaluation_worker import encoding_cases

    config = _phase("encoding")
    cases = encoding_cases(config)

    assert len(cases) == 40
    assert len({(item["prompt_id"], item["history_id"]) for item in cases}) == 40
    assert all(item["refs"] for item in cases)


def test_refine_records_pair_against_frozen_screen_baselines():
    parent = [
        {
            "job_id": "plain",
            "role": "plain",
            "topic_id": "cat",
            "history_id": None,
            "seed": 1,
            "status": "measured",
            "target_score": 0.4,
            "history_score": None,
        },
        {
            "job_id": "legacy",
            "role": "legacy",
            "topic_id": "cat",
            "history_id": "warm",
            "seed": 1,
            "status": "measured",
            "target_score": 0.5,
            "history_score": 0.2,
            "plain_context_history_score": 0.1,
        },
    ]
    refine = [
        {
            "job_id": "refine",
            "role": "candidate",
            "policy_hash": "alpha-03",
            "topic_id": "cat",
            "history_id": "warm",
            "seed": 1,
            "status": "measured",
            "target_score": 0.55,
            "history_score": 0.3,
        }
    ]

    paired = attach_parent_pairs(refine, parent)

    assert paired[0]["plain_job_id"] == "plain"
    assert paired[0]["legacy_job_id"] == "legacy"
    assert paired[0]["delta_vs_plain"] == {
        "target_score": pytest.approx(0.15),
        "history_score": pytest.approx(0.2),
    }
    assert paired[0]["delta_vs_legacy"] == {
        "target_score": pytest.approx(0.05),
        "history_score": pytest.approx(0.1),
    }


def test_refine_diagnostics_use_the_exact_new_alpha_policy_hashes():
    selected = copy.deepcopy(_phase("screen")["policies"][:2])
    refine = _phase(
        "refine",
        parents={
            "screen": {
                "manifest_hash": "screen",
                "decision_hash": "decision",
                "selected_policies": selected,
            }
        },
    )

    assert len(refine["policies"]) == 4
    assert {item["effective_policy"]["alpha"] for item in refine["policies"]} == {
        0.3,
        0.5,
    }
    assert not {item["policy_hash"] for item in refine["policies"]} & {
        item["policy_hash"] for item in selected
    }


def test_controller_validates_events_checkpoints_and_resumes_without_regeneration(
    tmp_path, provenance
):
    import importlib.util

    from exhibit.config import ROOT
    from PIL import Image

    script = ROOT / "scripts/evaluate_fan.py"
    spec = importlib.util.spec_from_file_location("task3_evaluate_fan", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = _phase("screen")
    config["output_root"] = str(tmp_path)
    config["topics"] = config["topics"][:1]
    config["histories"] = config["histories"][:1]
    config["seeds"] = config["seeds"][:1]
    config["policies"] = config["policies"][:1]
    config["expected_jobs"] = 3
    policy_hash = config["policies"][0]["policy_hash"]
    config["numerical"] = {
        policy_hash: {"passed": True, "evidence_hash": "encoding-report"}
    }
    calls = []

    def runner(request, directory, cancel, deadline, on_event):
        calls.append(request)
        images = {}
        texts = {}
        conditioning = {}
        for item in request["items"]:
            on_event({"type": "image_started", "id": item["id"]})
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(1, 2, 3)).save(path)
            event = {
                "type": "image",
                "id": item["id"],
                "path": str(path),
                "sha256": file_hash(path),
                "seed": item["seed"],
                "prompt": item["prompt"],
                "settings": request["settings"],
                "effective_policy": item["effective_policy"],
                "policy_hash": item["policy_hash"],
                "personalization_hash": (
                    item["personalization"]["personalization_hash"]
                    if item["personalization"]
                    else None
                ),
                "seconds": 0.1,
            }
            on_event(event)
            images[item["id"]] = {
                "embedding": [1.0, 0.0],
                "valid": True,
                "sha256": event["sha256"],
                "seconds": 0.1,
            }
            conditioning[item["id"]] = {
                "hidden": {"valid": True, "cosine": 1.0},
                "pooled": {"valid": True, "cosine": 1.0},
            }
            texts[item["target_text_id"]] = [1.0, 0.0]
            for ref in item.get("refs", []):
                texts["ref:" + ref["ref_id"]] = [1.0, 0.0]
        embeddings_path = Path(request["embeddings_output"])
        embeddings_path.write_text(
            json.dumps(
                {
                    "images": images,
                    "texts": texts,
                    "conditioning": conditioning,
                }
            )
        )
        on_event({"type": "evaluation_embeddings", "path": str(embeddings_path)})
        return {"returncode": 0, "wall_seconds": 0.1}

    controller_events = []
    first = module.execute_experiment(
        config,
        provenance,
        resume=False,
        cancel=module.threading.Event(),
        deadline=module.time.monotonic() + 10,
        on_event=controller_events.append,
        runner=runner,
    )
    second = module.execute_experiment(
        config,
        provenance,
        resume=True,
        cancel=module.threading.Event(),
        deadline=module.time.monotonic() + 10,
        on_event=lambda event: None,
        runner=lambda *args, **kwargs: pytest.fail("resume regenerated done images"),
    )

    assert len(calls) == 1
    assert len(calls[0]["items"]) == 3
    assert controller_events[0] == {
        "type": "experiment_plan",
        "phase": "screen",
        "experiment_hash": first["experiment_hash"],
        "planned_images": 3,
        "total_images": 3,
        "evaluation_only": False,
    }
    assert first["decision"]["status"] == "insufficient_candidates"
    assert second["decision"] == first["decision"]
    checkpoint = json.loads((Path(first["directory"]) / "checkpoint.json").read_text())
    assert {item["status"] for item in checkpoint["jobs"].values()} == {"done"}
    assert checkpoint["created_at"]
    assert checkpoint["runs"][0]["started_at"]
    assert checkpoint["runs"][0]["finished_at"]
    assert checkpoint["runs"][0]["status"] == "finished"
    assert all(
        item["attempt_history"][0]["finished_at"]
        for item in checkpoint["jobs"].values()
    )


def test_controller_rejects_an_image_event_for_a_different_effective_policy(
    tmp_path, provenance
):
    import importlib.util

    from exhibit.config import ROOT

    script = ROOT / "scripts/evaluate_fan.py"
    spec = importlib.util.spec_from_file_location("task3_evaluate_fan_mismatch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = _phase("screen")
    config["output_root"] = str(tmp_path)
    config["topics"] = config["topics"][:1]
    config["histories"] = config["histories"][:1]
    config["seeds"] = config["seeds"][:1]
    config["policies"] = config["policies"][:1]
    config["expected_jobs"] = 3

    def runner(request, directory, cancel, deadline, on_event):
        item = request["items"][0]
        on_event({"type": "image_started", "id": item["id"]})
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        on_event(
            {
                "type": "image",
                "id": item["id"],
                "path": str(path),
                "sha256": file_hash(path),
                "seed": item["seed"],
                "prompt": item["prompt"],
                "settings": request["settings"],
                "effective_policy": item["effective_policy"],
                "policy_hash": "wrong-policy",
                "personalization_hash": None,
            }
        )

    with pytest.raises(ValueError, match="worker event contract"):
        module.execute_experiment(
            config,
            provenance,
            resume=False,
            cancel=module.threading.Event(),
            deadline=module.time.monotonic() + 10,
            on_event=lambda event: None,
            runner=runner,
        )


def test_selection_rejects_duplicate_job_ids_and_case_keys():
    duplicate_id = _candidate_rows("candidate", 0.1, 0.0)
    duplicate_id[1]["job_id"] = duplicate_id[0]["job_id"]
    rules = {
        "stage": "screen",
        "expected_cases": 2,
        "target_non_degradation": -0.01,
        "numerical": {"candidate": {"passed": True, "evidence_hash": "e"}},
    }
    with pytest.raises(ValueError, match="duplicate record"):
        select_candidates(duplicate_id, rules)

    duplicate_case = _candidate_rows("candidate", 0.1, 0.0)
    duplicate_case[1].update(
        {
            "topic_id": duplicate_case[0]["topic_id"],
            "history_id": duplicate_case[0]["history_id"],
            "seed": duplicate_case[0]["seed"],
        }
    )
    with pytest.raises(ValueError, match="duplicate case"):
        select_candidates(duplicate_case, rules)


def test_screen_with_only_one_eligible_policy_is_explicitly_insufficient():
    rows = [
        *_candidate_rows("only", 0.1, 0.0),
        *_candidate_rows("failed", 0.5, -0.02),
    ]
    decision = select_candidates(
        rows,
        {
            "stage": "screen",
            "expected_cases": 2,
            "target_non_degradation": -0.01,
            "numerical": {
                "only": {"passed": True, "evidence_hash": "e"},
                "failed": {"passed": True, "evidence_hash": "e"},
            },
        },
    )

    assert decision["status"] == "insufficient_candidates"
    assert [item["policy_hash"] for item in decision["selected"]] == ["only"]


def test_matrix_image_validation_rejects_black_and_wrong_size(tmp_path):
    from exhibit.evaluation_worker import validate_image_file
    from PIL import Image

    valid = tmp_path / "valid.png"
    black = tmp_path / "black.png"
    wrong = tmp_path / "wrong.png"
    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(valid)
    Image.new("RGB", (8, 6), color=(0, 0, 0)).save(black)
    Image.new("RGB", (7, 6), color=(1, 2, 3)).save(wrong)

    assert validate_image_file(valid, 8, 6)["valid"] is True
    assert validate_image_file(black, 8, 6)["invalid_reason"] == "all_zero_rgb"
    assert validate_image_file(wrong, 8, 6)["invalid_reason"] == "wrong_dimensions"


def test_matrix_worker_reuses_generation_and_persists_embeddings(tmp_path, monkeypatch):
    from exhibit import evaluation_worker

    item = {
        "id": "job",
        "prompt": "prompt",
        "target_text_id": "target:cat",
        "target_text": "a cat",
        "refs": [{"ref_id": "warm", "text": "warm", "weight": 1.0}],
        "seed": 1,
        "path": str(tmp_path / "job.png"),
        "personalization": None,
        "effective_policy": _phase("screen")["legacy_policy"]["effective_policy"],
        "policy_hash": _phase("screen")["legacy_policy"]["policy_hash"],
    }
    output = tmp_path / "embeddings.json"
    events = []
    monkeypatch.setattr(
        evaluation_worker, "emit", lambda kind, **data: events.append((kind, data))
    )
    calls = []

    cosines = {"hidden": 0.9, "pooled": 0.8}
    monkeypatch.setattr(
        evaluation_worker,
        "tensor_metrics",
        lambda candidate, base: {
            "finite": True,
            "candidate_zero_vectors": 0,
            "base_zero_vectors": 0,
            "mean_cosine": cosines[candidate],
            "shape": [1],
        },
    )

    def generator(request, *, event_sink, conditioning_sink):
        calls.append(request)
        Path(item["path"]).write_bytes(b"image")
        # The same call `workers.generate` makes: item id, candidate, plain.
        aligned = conditioning_sink(
            "job",
            {"hidden": "hidden", "pooled": "pooled"},
            {"hidden": "plain-hidden", "pooled": "plain-pooled"},
        )
        assert {name: aligned[name]["cosine"] for name in aligned} == cosines
        event_sink(
            "image",
            id="job",
            path=item["path"],
            sha256=file_hash(Path(item["path"])),
            conditioning_target_align=aligned,
            seconds=0.1,
        )

    def embedder(items, preparation, settings, seconds):
        assert items == [item]
        assert seconds == {"job": 0.1}
        return {
            "images": {
                "job": {
                    "valid": True,
                    "embedding": [1.0, 0.0],
                    "sha256": file_hash(Path(item["path"])),
                    "seconds": 0.1,
                }
            },
            "texts": {"target:cat": [1.0, 0.0], "ref:warm": [1.0, 0.0]},
        }

    result = evaluation_worker.run_matrix(
        {
            "settings": {"width": 8, "height": 8},
            "items": [item],
            "all_items": [item],
            "evaluator_preparation": {"snapshot_path": "/snapshot"},
            "embeddings_output": str(output),
        },
        generator=generator,
        embedder=embedder,
    )

    assert len(calls) == 1
    assert result["conditioning"]["job"]["hidden"]["cosine"] == 0.9
    assert json.loads(output.read_text()) == result
    assert events[-1] == (
        "evaluation_embeddings",
        {"path": str(output.resolve()), "image_count": 1},
    )


def test_worker_failure_charges_only_the_image_that_started(tmp_path, provenance):
    import importlib.util

    from exhibit.config import ROOT

    script = ROOT / "scripts/evaluate_fan.py"
    spec = importlib.util.spec_from_file_location("task3_attempts", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = _phase("screen")
    config["output_root"] = str(tmp_path)
    config["topics"] = config["topics"][:1]
    config["histories"] = config["histories"][:1]
    config["seeds"] = config["seeds"][:1]
    config["policies"] = config["policies"][:1]
    config["expected_jobs"] = 3

    def failing_runner(request, directory, cancel, deadline, on_event):
        on_event({"type": "image_started", "id": request["items"][0]["id"]})
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        module.execute_experiment(
            config,
            provenance,
            resume=False,
            cancel=module.threading.Event(),
            deadline=module.time.monotonic() + 10,
            on_event=lambda event: None,
            runner=failing_runner,
        )

    experiment = build_experiment(config, provenance)
    checkpoint = json.loads(
        (tmp_path / experiment["experiment_hash"] / "checkpoint.json").read_text()
    )
    states = list(checkpoint["jobs"].values())
    assert [item["attempts"] for item in states] == [1, 0, 0]
    assert [item["status"] for item in states] == ["failed", "not_run", "not_run"]


def test_resume_runs_evaluation_only_when_images_are_done_but_embeddings_missing(
    tmp_path, provenance
):
    import importlib.util

    from exhibit.config import ROOT
    from PIL import Image

    script = ROOT / "scripts/evaluate_fan.py"
    spec = importlib.util.spec_from_file_location("task3_evaluation_resume", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = _phase("screen")
    config["output_root"] = str(tmp_path)
    config["topics"] = config["topics"][:1]
    config["histories"] = config["histories"][:1]
    config["seeds"] = config["seeds"][:1]
    config["policies"] = config["policies"][:1]
    config["expected_jobs"] = 3
    experiment = build_experiment(config, provenance)
    directory, checkpoint = register_experiment(tmp_path, experiment, resume=False)
    for job in experiment["jobs"]:
        image = directory / job["path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image)
        personalization = job.get("personalization")
        event = {
            "type": "image",
            "id": job["job_id"],
            "path": str(image.resolve()),
            "sha256": file_hash(image),
            "seed": job["seed"],
            "prompt": job["prompt"],
            "settings": config["generation"],
            "effective_policy": job["effective_policy"],
            "policy_hash": job["policy_hash"],
            "personalization_hash": (
                personalization["personalization_hash"] if personalization else None
            ),
        }
        checkpoint["jobs"][job["job_id"]] = {
            "status": "done",
            "attempts": 1,
            "sha256": event["sha256"],
            "contract_hash": job["contract_hash"],
            "event": event,
        }
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))
    calls = []

    def evaluation_only(request, worker_directory, cancel, deadline, on_event):
        calls.append(request)
        assert request["items"] == []
        images = {}
        texts = {}
        for item in request["all_items"]:
            images[item["id"]] = {
                "embedding": [1.0, 0.0],
                "valid": True,
                "sha256": file_hash(Path(item["path"])),
            }
            texts[item["target_text_id"]] = [1.0, 0.0]
            for ref in item.get("refs", []):
                texts["ref:" + ref["ref_id"]] = [1.0, 0.0]
        output = Path(request["embeddings_output"])
        output.write_text(
            json.dumps({"images": images, "texts": texts, "conditioning": {}})
        )
        on_event({"type": "evaluation_embeddings", "path": str(output)})
        return {"returncode": 0, "wall_seconds": 0.1}

    module.execute_experiment(
        config,
        provenance,
        resume=True,
        cancel=module.threading.Event(),
        deadline=module.time.monotonic() + 10,
        on_event=lambda event: None,
        runner=evaluation_only,
    )

    assert len(calls) == 1
    assert calls[0]["items"] == []


def test_evaluation_only_worker_restores_timing_and_conditioning_from_checkpoint(
    tmp_path, monkeypatch
):
    from exhibit import evaluation_worker

    output = tmp_path / "embeddings.json"
    item = {
        "id": "done-job",
        "path": str(tmp_path / "done.png"),
        "target_text_id": "target:cat",
        "target_text": "cat",
        "refs": [],
        "prior_event": {
            "seconds": 2.5,
            "conditioning_target_align": {
                "hidden": {"valid": True, "cosine": 0.7},
                "pooled": {"valid": True, "cosine": 0.6},
            },
        },
    }
    monkeypatch.setattr(evaluation_worker, "emit", lambda *args, **kwargs: None)

    def embedder(items, preparation, settings, seconds):
        assert seconds == {"done-job": 2.5}
        return {
            "images": {
                "done-job": {
                    "valid": True,
                    "embedding": [1.0],
                    "seconds": seconds["done-job"],
                }
            },
            "texts": {"target:cat": [1.0]},
        }

    result = evaluation_worker.run_matrix(
        {
            "settings": {"width": 8, "height": 8},
            "items": [],
            "all_items": [item],
            "evaluator_preparation": {"snapshot_path": "/snapshot"},
            "embeddings_output": str(output),
        },
        generator=lambda *args, **kwargs: pytest.fail("regenerated image"),
        embedder=embedder,
    )

    assert (
        result["conditioning"]["done-job"]
        == item["prior_event"]["conditioning_target_align"]
    )
    assert result["images"]["done-job"]["seconds"] == 2.5


def test_resume_normalizes_persisted_running_state_without_refunding_attempt(
    tmp_path, provenance
):
    experiment = build_experiment(_phase("screen"), provenance)
    directory, checkpoint = register_experiment(tmp_path, experiment, resume=False)
    job_id = experiment["jobs"][0]["job_id"]
    checkpoint["jobs"][job_id] = {
        "status": "running",
        "attempts": 1,
        "started_at": "2026-09-21T00:00:00Z",
        "attempt_history": [
            {
                "attempt": 1,
                "started_at": "2026-09-21T00:00:00Z",
                "finished_at": None,
                "status": "running",
            }
        ],
    }
    checkpoint["runs"] = [
        {
            "started_at": "2026-09-21T00:00:00Z",
            "finished_at": None,
            "status": "running",
        }
    ]
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))

    _, resumed = register_experiment(directory, experiment, resume=True)

    assert resumed["jobs"][job_id]["attempts"] == 1
    assert resumed["jobs"][job_id]["status"] == "failed"
    assert resumed["jobs"][job_id]["error"] == "interrupted_before_resume"
    assert resumed["jobs"][job_id]["attempt_history"][0]["status"] == "interrupted"
    assert resumed["jobs"][job_id]["attempt_history"][0]["finished_at"]
    assert resumed["runs"][0]["status"] == "interrupted"


def test_heldout_summary_includes_topic_history_and_clustered_intervals():
    rows = []
    for history_index, history in enumerate(
        ("warm-cel", "cool-watercolor", "calm-flat", "mixed")
    ):
        for topic in ("cat", "tokyo"):
            rows.append(
                {
                    "job_id": f"{history}-{topic}",
                    "role": "candidate",
                    "policy_hash": "candidate",
                    "status": "measured",
                    "topic_id": topic,
                    "history_id": history,
                    "seed": history_index,
                    "target_score": 0.5,
                    "history_score": 0.6,
                    "delta_vs_plain": {
                        "target_score": 0.01,
                        "history_score": 0.02,
                    },
                    "delta_vs_legacy": {
                        "target_score": 0.001 * history_index,
                        "history_score": 0.01 + 0.001 * history_index,
                    },
                }
            )

    summary = summarize_records(
        rows,
        "heldout",
        {"bootstrap_draws": 2000, "bootstrap_seed": 0},
    )

    candidate = summary["policies"]["candidate"]
    assert set(candidate["by_topic"]) == {"cat", "tokyo"}
    assert set(candidate["by_history"]) == {
        "warm-cel",
        "cool-watercolor",
        "calm-flat",
        "mixed",
    }
    assert (
        candidate["bootstrap_95"]["history_delta_vs_legacy"]["unit"]
        == "synthetic_history"
    )
    assert candidate["bootstrap_95"]["history_delta_vs_legacy"]["draws"] == 2000


def test_heldout_requires_a_prepared_reviewed_v2_catalog(tmp_path):
    heldout = _phase(
        "heldout",
        parents={
            "refine": {
                "manifest_hash": "refine",
                "decision_hash": "decision",
                "selected_policies": copy.deepcopy(_phase("screen")["policies"][:2]),
            }
        },
    )

    def unavailable(*args, **kwargs):
        raise ValueError("catalog-v2 assets are missing")

    with pytest.raises(ValueError, match="heldout catalog is not prepared"):
        validate_heldout_catalog(heldout, catalog_loader=unavailable)


def _heldout_phase():
    return _phase(
        "heldout",
        parents={
            "refine": {
                "manifest_hash": "refine",
                "decision_hash": "decision",
                "selected_policies": copy.deepcopy(_phase("screen")["policies"][:2]),
            }
        },
    )


def _v2_cards():
    from exhibit.catalog import build_catalog
    from exhibit.config import ROOT as EXHIBIT_ROOT
    from exhibit.config import read_json

    return build_catalog(read_json(EXHIBIT_ROOT / "configs/catalog-v2.json"))


def _loader_for(reviewed_ids):
    cards = _v2_cards()

    def loader(catalog_id, *, reviewed_only):
        assert (catalog_id, reviewed_only) == ("catalog-v2", True)
        return {
            "catalog_id": catalog_id,
            "catalog_hash": "catalog-content-hash",
            "cards": [copy.deepcopy(c) for c in cards if c["id"] in reviewed_ids],
            "all_cards": copy.deepcopy(cards),
        }

    return loader


def _referenced(heldout):
    return {card for history in heldout["histories"] for card in history["cards"]}


def test_heldout_catalog_binds_the_frozen_refs_to_the_real_v2_catalog():
    """The fixture's nine cards must resolve, aspect for aspect, against v2."""
    heldout = _heldout_phase()
    every = {card["id"] for card in _v2_cards()}
    identity = validate_heldout_catalog(heldout, catalog_loader=_loader_for(every))

    assert identity["catalog_hash"] == "catalog-content-hash"
    assert identity["reviewed_card_count"] == 64
    assert identity["reviewed_card_ids_hash"] == digest(sorted(every))
    assert identity["min_reviewed_per_level"] == 2
    assert identity["min_reviewed_cards"] == 32
    assert sorted(identity["card_ids"]) == sorted(_referenced(heldout))
    assert len(identity["card_ids"]) == 9


def test_heldout_runs_on_a_partly_reviewed_catalog_and_records_that_set():
    """Design §6.1: the run is tied to the reviewed set it actually ran on."""
    heldout = _heldout_phase()
    reviewed = {
        card["id"] for card in _v2_cards() if card["subject_id"] in ("girl", "student")
    } | _referenced(heldout)
    identity = validate_heldout_catalog(heldout, catalog_loader=_loader_for(reviewed))

    assert identity["reviewed_card_count"] == len(reviewed) == 35
    assert identity["reviewed_card_ids_hash"] == digest(sorted(reviewed))


def test_heldout_names_the_gate_condition_that_failed():
    heldout = _heldout_phase()
    cards = _v2_cards()
    referenced = _referenced(heldout)

    missing_card = {card["id"] for card in cards} - {min(referenced)}
    with pytest.raises(ValueError, match="heldout card is not reviewed"):
        validate_heldout_catalog(heldout, catalog_loader=_loader_for(missing_card))

    # One sketch card left reviewed: the texture axis can no longer be compared.
    sketch = [card["id"] for card in cards if card["axis_levels"]["texture"] == 0]
    thin_level = {card["id"] for card in cards} - set(sketch[1:])
    with pytest.raises(ValueError, match="texture level 0 has 1 reviewed cards"):
        validate_heldout_catalog(heldout, catalog_loader=_loader_for(thin_level))

    reviewed = {
        card["id"] for card in cards if card["subject_id"] in ("girl", "student")
    } | referenced
    raised = copy.deepcopy(heldout)
    raised["catalog_gate"]["min_reviewed_cards"] = 40
    with pytest.raises(ValueError, match="too few reviewed cards: 35 of 64"):
        validate_heldout_catalog(raised, catalog_loader=_loader_for(reviewed))


def test_runtime_model_provenance_hashes_only_consumed_pinned_files(tmp_path):
    from exhibit.evaluation_worker import runtime_file_provenance

    files = {}

    def download(repo_id, *, filename, revision, local_files_only):
        path = tmp_path / repo_id.replace("/", "_") / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{repo_id}:{revision}:{filename}".encode())
        files[(repo_id, revision, filename)] = path
        return str(path)

    generation = copy.deepcopy(_phase("screen")["generation"])
    result = runtime_file_provenance(generation, download=download)

    assert result["checkpoint"]["revision"] == generation["revision"]
    assert result["checkpoint"]["files"] == {
        generation["checkpoint"]: file_hash(
            files[
                (generation["model"], generation["revision"], generation["checkpoint"])
            ]
        )
    }
    pipeline_files = result["pipeline_config"]["files"]
    assert "text_encoder/config.json" in pipeline_files
    assert "text_encoder_2/config.json" in pipeline_files
    assert "tokenizer/merges.txt" in pipeline_files
    assert "tokenizer_2/special_tokens_map.json" in pipeline_files
    assert "text_encoder/model.safetensors" not in pipeline_files
    assert "unet/diffusion_pytorch_model.safetensors" not in pipeline_files
    assert set(result["vae"]["files"]) == {
        "config.json",
        "diffusion_pytorch_model.safetensors",
    }
