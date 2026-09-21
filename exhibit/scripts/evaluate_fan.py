#!/usr/bin/env python3
"""Run reproducible FAN diagnostics and image evaluation under one GPU lease."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from exhibit.config import FAN_UPSTREAM, GPU_PYTHON, ROOT, write_json
from exhibit.domain import digest, file_hash
from exhibit.evaluation import (
    attach_parent_pairs,
    build_experiment,
    load_evaluation_config,
    register_experiment,
    runtime_file_provenance,
    score_records,
    select_candidates,
    summarize_records,
    validate_evaluator_preparation,
    validate_heldout_catalog,
)
from exhibit.evaluation_worker import validate_policy_specs
from exhibit.gpu import gpu_lease, run_process


def _environment(upstream):
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(upstream)]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def run_encoding_controller(config_path, cancel, deadline, on_event):
    """Backward-compatible Task 2 controller for an explicit worker config."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate_policy_specs(config.get("policies"))
    if not isinstance(config.get("output"), str) or not config["output"]:
        raise ValueError("output is required")
    output = Path(config["output"]).resolve()
    upstream = Path(config.get("upstream", str(FAN_UPSTREAM))).resolve()
    directory = output.parent / ("encoding-" + uuid.uuid4().hex[:8])
    with gpu_lease(cancel, deadline):
        return run_process(
            [
                GPU_PYTHON,
                "-m",
                "exhibit.evaluation_worker",
                "encoding",
                "--config",
                str(config_path),
            ],
            directory,
            cancel,
            deadline,
            on_event,
            env=_environment(upstream),
        )


def _worker_item(job, directory, state=None):
    value = {
        "id": job["job_id"],
        "prompt": job["prompt"],
        "target_text_id": job["target_text_id"],
        "target_text": job["target_text"],
        "refs": copy.deepcopy(job.get("refs", [])),
        "seed": job["seed"],
        "path": str((directory / job["path"]).resolve()),
        "personalization": copy.deepcopy(job.get("personalization")),
        "effective_policy": copy.deepcopy(job["effective_policy"]),
        "policy_hash": job["policy_hash"],
    }
    if state and state.get("status") == "done" and isinstance(state.get("event"), dict):
        value["prior_event"] = copy.deepcopy(state["event"])
    return value


def _event_matches(event, job, directory, settings):
    personalization = job.get("personalization")
    expected_personalization = (
        personalization.get("personalization_hash") if personalization else None
    )
    expected_path = (directory / job["path"]).resolve()
    return (
        event.get("id") == job["job_id"]
        and Path(event.get("path", "")).resolve() == expected_path
        and expected_path.is_file()
        and event.get("sha256")
        == __import__("exhibit.domain", fromlist=["file_hash"]).file_hash(expected_path)
        and event.get("seed") == job["seed"]
        and event.get("prompt") == job["prompt"]
        and event.get("settings") == settings
        and event.get("effective_policy") == job["effective_policy"]
        and event.get("policy_hash") == job["policy_hash"]
        and event.get("personalization_hash") == expected_personalization
    )


def run_matrix_worker(request, directory, cancel, deadline, on_event):
    """Run the GPU matrix worker under the same physical lease as web generation."""
    upstream = Path(request.get("upstream", str(FAN_UPSTREAM))).resolve()
    run_directory = Path(directory) / ("matrix-" + uuid.uuid4().hex[:8])
    run_directory.mkdir(parents=True, exist_ok=True)
    request_file = run_directory / "request.json"
    write_json(request_file, request)
    with gpu_lease(cancel, deadline):
        return run_process(
            [
                GPU_PYTHON,
                "-m",
                "exhibit.evaluation_worker",
                "matrix",
                "--request",
                str(request_file),
            ],
            run_directory,
            cancel,
            deadline,
            on_event,
            env=_environment(upstream),
        )


def _timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execute_experiment(
    config,
    provenance,
    *,
    resume,
    cancel,
    deadline,
    on_event,
    runner=None,
):
    """Register, run, checkpoint, score, and select one image experiment."""
    experiment = build_experiment(config, provenance)
    directory, checkpoint = register_experiment(
        config["output_root"], experiment, resume=resume
    )
    checkpoint_path = directory / "checkpoint.json"
    jobs = {item["job_id"]: item for item in experiment["jobs"]}
    maximum_attempts = config.get("limits", {}).get("max_attempts", 2)
    pending = [
        jobs[job_id]
        for job_id, state in checkpoint["jobs"].items()
        if state.get("status") != "done" and state.get("attempts", 0) < maximum_attempts
    ]
    embeddings_path = directory / "embeddings.json"
    runner = runner or run_matrix_worker

    def embeddings_complete():
        if not embeddings_path.is_file():
            return False
        try:
            value = json.loads(embeddings_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        images = value.get("images")
        done_ids = {
            job_id
            for job_id, state in checkpoint["jobs"].items()
            if state.get("status") == "done"
        }
        return isinstance(images, dict) and done_ids <= set(images)

    evaluation_only = not pending and not embeddings_complete()
    run_entry = {
        "started_at": _timestamp(),
        "finished_at": None,
        "status": "running",
        "resume": bool(resume),
        "planned_images": len(pending),
        "evaluation_only": evaluation_only,
    }
    checkpoint.setdefault("runs", []).append(run_entry)
    write_json(checkpoint_path, checkpoint)
    on_event(
        {
            "type": "experiment_plan",
            "phase": config["phase"],
            "experiment_hash": experiment["experiment_hash"],
            "planned_images": len(pending),
            "total_images": len(jobs),
            "evaluation_only": evaluation_only,
        }
    )

    if pending or not embeddings_complete():
        request = {
            "schema_version": 1,
            "stage": "matrix",
            "phase": config["phase"],
            "settings": copy.deepcopy(config["generation"]),
            "upstream": str(FAN_UPSTREAM),
            "evaluator": copy.deepcopy(config["evaluator"]),
            "evaluator_preparation": copy.deepcopy(
                config.get("evaluator_preparation", {})
            ),
            "items": [_worker_item(job, directory) for job in pending],
            "all_items": [
                _worker_item(job, directory, checkpoint["jobs"][job["job_id"]])
                for job in experiment["jobs"]
            ],
            "embeddings_output": str(embeddings_path.resolve()),
        }

        def finish_attempt(state, status, *, error=None):
            now = _timestamp()
            state["finished_at"] = now
            if error is not None:
                state["error"] = error
            history = state.get("attempt_history", [])
            if history and history[-1].get("status") == "running":
                history[-1].update({"status": status, "finished_at": now})

        def receive(event):
            if event.get("type") == "image_started":
                job = jobs.get(event.get("id"))
                if job is None or job not in pending:
                    raise ValueError("worker image_started does not match manifest")
                state = checkpoint["jobs"][job["job_id"]]
                if (
                    state.get("status") == "done"
                    or state.get("status") == "running"
                    or state.get("attempts", 0) >= maximum_attempts
                ):
                    raise ValueError("worker image_started exceeds attempt contract")
                attempt = state.get("attempts", 0) + 1
                started_at = _timestamp()
                state.update(
                    {
                        "status": "running",
                        "attempts": attempt,
                        "started_at": started_at,
                        "finished_at": None,
                    }
                )
                state.setdefault("attempt_history", []).append(
                    {
                        "attempt": attempt,
                        "started_at": started_at,
                        "finished_at": None,
                        "status": "running",
                    }
                )
                state.pop("error", None)
                write_json(checkpoint_path, checkpoint)
            elif event.get("type") == "image":
                job = jobs.get(event.get("id"))
                state = checkpoint["jobs"].get(event.get("id"), {})
                if (
                    job is None
                    or job not in pending
                    or state.get("status") != "running"
                    or not _event_matches(event, job, directory, config["generation"])
                ):
                    raise ValueError("worker event contract does not match manifest")
                state.update(
                    {
                        "status": "done",
                        "sha256": event["sha256"],
                        "contract_hash": job["contract_hash"],
                        "event": copy.deepcopy(event),
                    }
                )
                finish_attempt(state, "done")
                write_json(checkpoint_path, checkpoint)
            elif event.get("type") == "evaluation_embeddings":
                if Path(event.get("path", "")).resolve() != embeddings_path.resolve():
                    raise ValueError("worker embeddings path does not match manifest")
            on_event(event)

        try:
            runner(request, directory, cancel, deadline, receive)
        except BaseException as error:
            for job in pending:
                state = checkpoint["jobs"][job["job_id"]]
                if state.get("status") == "running":
                    state["status"] = "failed"
                    finish_attempt(state, "failed", error=str(error))
            run_entry.update(
                {"status": "failed", "finished_at": _timestamp(), "error": str(error)}
            )
            write_json(checkpoint_path, checkpoint)
            raise
        for job in pending:
            state = checkpoint["jobs"][job["job_id"]]
            if state.get("status") == "running":
                state["status"] = "failed"
                finish_attempt(state, "failed", error="worker_missing_image_event")
        write_json(checkpoint_path, checkpoint)

    embeddings = (
        json.loads(embeddings_path.read_text())
        if embeddings_path.is_file()
        else {"images": {}, "texts": {}, "conditioning": {}}
    )
    embeddings.setdefault("images", {})
    for job_id, state in checkpoint["jobs"].items():
        if state.get("status") != "done":
            embeddings["images"][job_id] = {
                "status": state.get("status", "not_run"),
                "error": state.get("error"),
            }
    records = score_records(experiment, embeddings)
    ranking_records = records
    if config["phase"] == "refine":
        parent_records = config.get("parent_records", [])
        records = attach_parent_pairs(records, parent_records)
        parent_hashes = set(config.get("parent_selected_policy_hashes", []))
        ranking_records = [
            copy.deepcopy(item)
            for item in parent_records
            if item.get("role") == "candidate"
            and item.get("policy_hash") in parent_hashes
        ] + records
    expected_cases = (
        len(config["topics"]) * len(config["histories"]) * len(config["seeds"])
    )
    selection_rules = {
        **copy.deepcopy(config["rules"]),
        "stage": config["phase"],
        "expected_cases": expected_cases,
        "numerical": copy.deepcopy(config.get("numerical", {})),
    }
    decision = select_candidates(ranking_records, selection_rules)
    policies = {
        item["policy_hash"]: item
        for item in [
            *config.get("ranking_policies", []),
            *config.get("policies", []),
        ]
    }
    for selected in decision["selected"]:
        spec = policies[selected["policy_hash"]]
        selected["policy_id"] = spec.get("policy_id")
        selected["effective_policy"] = copy.deepcopy(spec["effective_policy"])
    decision["decision_hash"] = digest(
        {key: value for key, value in decision.items() if key != "decision_hash"}
    )
    metrics = summarize_records(ranking_records, config["phase"], config["rules"])
    run_entry.update({"status": "finished", "finished_at": _timestamp()})
    write_json(checkpoint_path, checkpoint)
    summary = {
        "schema_version": 1,
        "phase": config["phase"],
        "experiment_hash": experiment["experiment_hash"],
        "checkpoint": checkpoint,
        "records": records,
        "ranking_records": ranking_records,
        "metrics": metrics,
        "decision": decision,
        "directory": str(directory),
    }
    write_json(directory / "records.json", records)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "decision.json", decision)
    write_json(directory / "summary.json", summary)
    return summary


def _diagnostic_sources(settings):
    source = ROOT / "src/exhibit"
    scripts = ROOT / "scripts"
    upstream = Path(FAN_UPSTREAM)
    weights = upstream / "weight"
    patch_root = upstream.parent.parent / "patches"
    series = patch_root / "series"
    patch_names = (
        [
            line
            for line in series.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        if series.is_file()
        else []
    )
    return {
        "fan": {
            name: file_hash(Path(FAN_UPSTREAM) / "fan" / name)
            for name in ("model.py", "wrapper.py")
        },
        "decoders": {name: file_hash(weights / name) for name in ("L.pth", "bigG.pth")},
        "model_files": runtime_file_provenance(settings),
        "patches": {
            "series_hash": file_hash(series) if series.is_file() else None,
            "files": {name: file_hash(patch_root / name) for name in patch_names},
        },
        "integration": {
            name: file_hash(source / name)
            for name in (
                "fan_adapter.py",
                "workers.py",
                "gpu.py",
                "evaluation.py",
                "evaluation_worker.py",
            )
        }
        | {
            "scripts/evaluate_fan.py": file_hash(scripts / "evaluate_fan.py"),
            "scripts/prepare_evaluation.py": file_hash(
                scripts / "prepare_evaluation.py"
            ),
        },
    }


def build_diagnostic_config(config_path, phase_config, preparation):
    """Build a content-addressed exact-policy diagnostic request."""
    encoding = load_evaluation_config(config_path, "encoding")
    specs = [phase_config["legacy_policy"], *phase_config.get("policies", [])]
    unique = {}
    for spec in specs:
        unique.setdefault(spec["policy_hash"], copy.deepcopy(spec))
    identity = {
        "schema_version": 1,
        "phase": phase_config["phase"],
        "parents": copy.deepcopy(phase_config.get("parents", {})),
        "generation": copy.deepcopy(phase_config["generation"]),
        "prompts": copy.deepcopy(encoding["prompts"]),
        "histories": copy.deepcopy(encoding["histories"]),
        "policies": [
            {
                "policy_hash": item["policy_hash"],
                "effective_policy": copy.deepcopy(item["effective_policy"]),
            }
            for item in unique.values()
        ],
        "evaluator_hash": preparation["evaluator_hash"],
        "sources": _diagnostic_sources(phase_config["generation"]),
    }
    diagnostic_hash = digest(identity)
    output = (
        Path(phase_config["output_root"])
        / "diagnostics"
        / diagnostic_hash
        / "report.json"
    )
    runtime = {
        "output": str(output.resolve()),
        "settings": copy.deepcopy(phase_config["generation"]),
        "upstream": str(FAN_UPSTREAM),
        "prompts": copy.deepcopy(encoding["prompts"]),
        "histories": copy.deepcopy(encoding["histories"]),
        "policies": list(unique.values()),
        "diagnostic_identity": {**identity, "diagnostic_hash": diagnostic_hash},
    }
    return runtime


def ensure_diagnostics(
    config_path, phase_config, preparation, cancel, deadline, on_event
):
    runtime = build_diagnostic_config(config_path, phase_config, preparation)
    output = Path(runtime["output"])
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        runtime_path = output.parent / "request.json"
        write_json(runtime_path, runtime)
        run_encoding_controller(runtime_path, cancel, deadline, on_event)
    report = json.loads(output.read_text())
    if report.get("diagnostic_identity") != runtime["diagnostic_identity"]:
        raise ValueError("encoding diagnostic identity mismatch")
    expected_hashes = {item["policy_hash"] for item in runtime["policies"]}
    if set(report.get("policies", {})) != expected_hashes:
        raise ValueError("encoding report does not cover exact effective policies")
    report["report_hash"] = digest(report)
    return report


def numerical_evidence(report, policies):
    result = {}
    for spec in policies:
        policy_hash = spec["policy_hash"]
        value = report.get("policies", {}).get(policy_hash)
        if value is None:
            continue
        result[policy_hash] = {
            "passed": value.get("eligibility", {}).get("passed") is True,
            "failures": copy.deepcopy(value.get("eligibility", {}).get("failures", {})),
            "evidence_hash": digest(
                {
                    "diagnostic_identity": report.get("diagnostic_identity"),
                    "provenance": report.get("provenance"),
                    "policy_hash": policy_hash,
                    "eligibility": value.get("eligibility"),
                }
            ),
            "report_hash": report["report_hash"],
        }
    return result


def experiment_provenance(report, preparation):
    runtime = report.get("provenance", {})
    integration = runtime.get("integration_source", {})
    value = {
        "fan_pin": runtime.get("fan_commit"),
        "fan_source": copy.deepcopy(runtime.get("fan_source")),
        "patches": copy.deepcopy(runtime.get("patch_series")),
        "adapter_hash": integration.get("fan_adapter.py"),
        "worker_hash": integration.get("workers.py"),
        "evaluation_worker_hash": integration.get("evaluation_worker.py"),
        "evaluation_hash": integration.get("evaluation.py"),
        "controller_hash": integration.get("scripts/evaluate_fan.py"),
        "preparation_script_hash": integration.get("scripts/prepare_evaluation.py"),
        "decoder_hash": copy.deepcopy(runtime.get("decoders")),
        "tokenizer_hash": copy.deepcopy(runtime.get("tokenizers")),
        "model_hash": digest(runtime.get("model_files")),
        "model_files": copy.deepcopy(runtime.get("model_files")),
        "evaluator_hash": preparation["evaluator_hash"],
        "environment": {
            **copy.deepcopy(runtime.get("libraries", {})),
            **copy.deepcopy(report.get("versions", {})),
        },
    }
    if any(value.get(key) in (None, "", {}) for key in value):
        raise ValueError("encoding report has incomplete runtime provenance")
    return value


def _verified_parent(config, provenance):
    experiment = build_experiment(config, provenance)
    directory = Path(config["output_root"]) / experiment["experiment_hash"]
    register_experiment(directory, experiment, resume=True)
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"parent {config['phase']} summary is missing")
    summary = json.loads(summary_path.read_text())
    if summary.get("experiment_hash") != experiment["experiment_hash"]:
        raise ValueError("parent summary experiment hash mismatch")
    decision = summary.get("decision", {})
    expected_decision_hash = digest(
        {key: value for key, value in decision.items() if key != "decision_hash"}
    )
    if decision.get("decision_hash") != expected_decision_hash:
        raise ValueError("parent decision hash mismatch")
    if len(decision.get("selected", [])) != 2:
        raise ValueError(f"parent {config['phase']} has insufficient candidates")
    return summary


def _parent_value(summary):
    return {
        "manifest_hash": summary["experiment_hash"],
        "decision_hash": summary["decision"]["decision_hash"],
        "selected_policies": [
            {
                "policy_id": item.get("policy_id"),
                "policy_hash": item["policy_hash"],
                "effective_policy": copy.deepcopy(item["effective_policy"]),
            }
            for item in summary["decision"]["selected"]
        ],
    }


def run_phase(config_path, phase, *, resume, cancel, deadline, on_event):
    """Resolve exact parents/diagnostics and execute screen, refine, or heldout."""
    base = load_evaluation_config(config_path, "screen")
    preparation = validate_evaluator_preparation(
        base["preparation_manifest"],
        expected={
            "repo_id": base["evaluator"]["repo_id"],
            "resolved_revision": base["evaluator"]["expected_revision"],
            "weight_format": base["evaluator"]["weight_format"],
        },
    )
    base["evaluator_preparation"] = preparation
    screen_report = ensure_diagnostics(
        config_path, base, preparation, cancel, deadline, on_event
    )
    provenance = experiment_provenance(screen_report, preparation)
    base["numerical"] = numerical_evidence(screen_report, base["policies"])
    if phase == "screen":
        return execute_experiment(
            base,
            provenance,
            resume=resume,
            cancel=cancel,
            deadline=deadline,
            on_event=on_event,
        )

    screen_summary = _verified_parent(base, provenance)
    screen_parent = _parent_value(screen_summary)
    refine = load_evaluation_config(
        config_path, "refine", parents={"screen": screen_parent}
    )
    refine["evaluator_preparation"] = preparation
    refine["parent_records"] = screen_summary["records"]
    refine["parent_selected_policy_hashes"] = [
        item["policy_hash"] for item in screen_parent["selected_policies"]
    ]
    refine["ranking_policies"] = copy.deepcopy(screen_parent["selected_policies"])
    refine_report = ensure_diagnostics(
        config_path, refine, preparation, cancel, deadline, on_event
    )
    refine["numerical"] = {
        **numerical_evidence(screen_report, screen_parent["selected_policies"]),
        **numerical_evidence(refine_report, refine["policies"]),
    }
    if phase == "refine":
        return execute_experiment(
            refine,
            provenance,
            resume=resume,
            cancel=cancel,
            deadline=deadline,
            on_event=on_event,
        )

    refine_summary = _verified_parent(refine, provenance)
    refine_parent = _parent_value(refine_summary)
    heldout = load_evaluation_config(
        config_path, "heldout", parents={"refine": refine_parent}
    )
    heldout["evaluator_preparation"] = preparation
    heldout["catalog"] = validate_heldout_catalog(heldout)
    heldout_report = ensure_diagnostics(
        config_path, heldout, preparation, cancel, deadline, on_event
    )
    heldout["numerical"] = numerical_evidence(heldout_report, heldout["policies"])
    return execute_experiment(
        heldout,
        provenance,
        resume=resume,
        cancel=cancel,
        deadline=deadline,
        on_event=on_event,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("encoding", "screen", "refine", "heldout", "study"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--timeout", type=float, default=1800)
        if name != "encoding":
            command.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cancel = threading.Event()
    deadline = time.monotonic() + args.timeout

    def show(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    if args.command == "encoding":
        raw = json.loads(Path(args.config).read_text())
        if "fixtures" not in raw:
            result = run_encoding_controller(args.config, cancel, deadline, show)
            print(json.dumps({"type": "controller", **result}), flush=True)
            return
        config = load_evaluation_config(args.config, "screen")
        preparation = validate_evaluator_preparation(
            config["preparation_manifest"],
            expected={
                "repo_id": config["evaluator"]["repo_id"],
                "resolved_revision": config["evaluator"]["expected_revision"],
                "weight_format": config["evaluator"]["weight_format"],
            },
        )
        report = ensure_diagnostics(
            args.config, config, preparation, cancel, deadline, show
        )
        print(
            json.dumps(
                {
                    "type": "encoding_complete",
                    "path": build_diagnostic_config(args.config, config, preparation)[
                        "output"
                    ],
                    "report_hash": report["report_hash"],
                    "all_eligible": all(
                        item["eligibility"]["passed"]
                        for item in report["policies"].values()
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    if args.command == "study":
        raise SystemExit(
            "study participant mapping is provided by Task 8; no participant data was fabricated"
        )
    summary = run_phase(
        args.config,
        args.command,
        resume=args.resume,
        cancel=cancel,
        deadline=deadline,
        on_event=show,
    )
    print(
        json.dumps(
            {
                "type": "experiment_complete",
                "phase": args.command,
                "experiment_hash": summary["experiment_hash"],
                "directory": summary["directory"],
                "decision": summary["decision"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
