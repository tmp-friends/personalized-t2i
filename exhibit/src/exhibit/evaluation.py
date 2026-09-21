"""Deterministic FAN experiment manifests, checkpoints, scores, and decisions."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from .domain import digest, file_hash
from .fan_adapter import freeze_policy, resolve_policy, thaw_policy

EVALUATOR_FILES = {
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
}

PIPELINE_CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/special_tokens_map.json",
    "unet/config.json",
    "vae/config.json",
)


def runtime_file_provenance(settings, *, download=None):
    """Hash every pinned Hub file consumed by ``workers.load_pipeline``."""
    if download is None:

        def download(repo_id, *, filename, revision, local_files_only):
            if local_files_only is not True:
                raise ValueError("runtime provenance must remain offline")
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(char not in "0123456789abcdef" for char in revision)
            ):
                raise ValueError("runtime provenance requires a pinned Hub commit")
            cache = os.environ.get("HF_HUB_CACHE")
            if cache is None:
                home = Path(
                    os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
                )
                cache = home / "hub"
            path = (
                Path(cache)
                / ("models--" + repo_id.replace("/", "--"))
                / "snapshots"
                / revision
                / filename
            )
            if not path.is_file():
                raise ValueError(
                    f"pinned runtime file is not cached: {repo_id}/{filename}"
                )
            return str(path)

    def record(label, spec, filenames):
        if (
            not isinstance(spec, dict)
            or not spec.get("model")
            or not spec.get("revision")
        ):
            raise ValueError(f"{label} model and pinned revision are required")
        files = {}
        for filename in filenames:
            if not filename:
                raise ValueError(f"{label} filename is required")
            path = Path(
                download(
                    spec["model"],
                    filename=filename,
                    revision=spec["revision"],
                    local_files_only=True,
                )
            )
            files[filename] = file_hash(path)
        return {
            "repo_id": spec["model"],
            "revision": spec["revision"],
            "files": files,
        }

    checkpoint = record("checkpoint", settings, (settings.get("checkpoint"),))
    pipeline = record(
        "pipeline_config", settings.get("pipeline_config"), PIPELINE_CONFIG_FILES
    )
    vae = record(
        "vae",
        settings.get("vae"),
        ("config.json", "diffusion_pytorch_model.safetensors"),
    )
    return {"checkpoint": checkpoint, "pipeline_config": pipeline, "vae": vae}


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _policy(value):
    return thaw_policy(freeze_policy(value))


def _policy_spec(policy_id, value):
    effective = _policy(value)
    return {
        "policy_id": policy_id,
        "policy_hash": digest(effective),
        "effective_policy": effective,
    }


def _screen_policies(raw):
    policies = []
    for skip_pa in raw["skip_pa"]:
        skip_name = "skip1" if skip_pa == [0] else "skip8"
        for pooled in raw["pooled_mode"]:
            for profiling in raw["profiling"]:
                profile_name = "all" if profiling["mode"] == "all" else "ratio01"
                policies.append(
                    _policy_spec(
                        f"screen-{skip_name}-{pooled}-{profile_name}",
                        {
                            "alpha": raw["alpha"],
                            "skip": -2,
                            "skip_pa": skip_pa,
                            "use_attn_mask": False,
                            "pooled_mode": pooled,
                            "profiling": profiling,
                            "reference_unit": "aspect_phrase",
                        },
                    )
                )
    return policies


def _select(items, ids, label):
    by_id = {item["id"]: item for item in items}
    if len(by_id) != len(items) or set(ids) - set(by_id):
        raise ValueError(f"unknown or duplicate {label}")
    return [copy.deepcopy(by_id[item_id]) for item_id in ids]


def load_evaluation_config(path, phase, *, parents=None):
    """Resolve one phase without consulting arbitrary prior output directories."""
    path = Path(path).resolve()
    raw = _read(path)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported evaluation config schema")
    base = path.parent
    generation = _read(base / raw["generation_config"])["generation"]
    topics_doc = _read(base / raw["fixtures"]["topics"])
    topics = topics_doc["topics"]
    screen_histories = _read(base / raw["fixtures"]["screen_histories"])["histories"]
    heldout_histories = _read(base / raw["fixtures"]["heldout_histories"])["histories"]
    diagnostic_histories = _read(base / raw["fixtures"]["diagnostic_histories"])[
        "histories"
    ]
    screen_policies = _screen_policies(raw["screen"])
    legacy = _policy_spec(
        "legacy_exhibit",
        thaw_policy(
            resolve_policy("legacy_exhibit", _read(base / "fan-policies.json"))
        ),
    )
    common = {
        "schema_version": 1,
        "phase": phase,
        "output_root": str((base / raw["output_root"]).resolve()),
        "preparation_manifest": str((base / raw["preparation_manifest"]).resolve()),
        "generation": generation,
        "evaluator": copy.deepcopy(raw["evaluator"]),
        "limits": copy.deepcopy(raw["limits"]),
        "rules": copy.deepcopy(raw["rules"]),
        "legacy_policy": legacy,
        "parents": copy.deepcopy(parents or {}),
    }
    if phase == "encoding":
        prompts = [
            {"id": item["id"], "text": item["generation_prompt"]} for item in topics
        ] + copy.deepcopy(topics_doc["diagnostic_prompts"])
        return {
            **common,
            "prompts": prompts,
            "histories": copy.deepcopy(diagnostic_histories),
            "policies": [legacy, *screen_policies],
        }
    if phase == "screen":
        return {
            **common,
            "topics": _select(topics, raw["screen"]["topics"], "topic"),
            "histories": _select(
                screen_histories, raw["screen"]["histories"], "history"
            ),
            "seeds": copy.deepcopy(raw["screen"]["seeds"]),
            "policies": screen_policies,
            "expected_jobs": 222,
        }
    if phase == "refine":
        parent = (parents or {}).get("screen", {})
        policies = []
        for selected in parent.get("selected_policies", []):
            for alpha in raw["refine"]["added_alpha"]:
                effective = copy.deepcopy(selected["effective_policy"])
                effective["alpha"] = alpha
                policies.append(
                    _policy_spec(
                        f"{selected.get('policy_id', 'candidate')}-alpha-{alpha}",
                        effective,
                    )
                )
        return {
            **common,
            "topics": _select(topics, raw["screen"]["topics"], "topic"),
            "histories": _select(
                screen_histories, raw["screen"]["histories"], "history"
            ),
            "seeds": copy.deepcopy(raw["screen"]["seeds"]),
            "policies": policies,
            "expected_jobs": 96,
        }
    if phase == "heldout":
        parent = (parents or {}).get("refine", {})
        return {
            **common,
            "topics": _select(topics, raw["heldout"]["topics"], "topic"),
            "histories": _select(
                heldout_histories, raw["heldout"]["histories"], "history"
            ),
            "seeds": copy.deepcopy(raw["heldout"]["seeds"]),
            "policies": copy.deepcopy(parent.get("selected_policies", [])),
            "expected_jobs": 156,
        }
    if phase == "study":
        return {**common, "integration": "task8"}
    raise ValueError(f"unknown evaluation phase: {phase}")


def _job(contract):
    contract = copy.deepcopy(contract)
    contract_hash = digest(contract)
    job_id = contract_hash
    return {
        **contract,
        "job_id": job_id,
        "contract_hash": contract_hash,
        "path": f"images/{job_id}.png",
    }


def _personalization(refs, prompt, policy, provenance):
    effective = policy["effective_policy"]
    policy_hash = digest(effective)
    personalization_hash = digest(
        {
            "refs": refs,
            "prompt": prompt,
            "effective_policy": effective,
            "provenance": provenance,
        }
    )
    return {
        "refs": copy.deepcopy(refs),
        "effective_policy": copy.deepcopy(effective),
        "policy_hash": policy_hash,
        "policy_id": policy.get("policy_id"),
        "personalization_hash": personalization_hash,
        "hash": personalization_hash,
    }


def build_experiment(config, provenance):
    """Build a canonical immutable image matrix and its display metadata."""
    if not isinstance(config, dict) or config.get("phase") not in {
        "screen",
        "refine",
        "heldout",
    }:
        raise ValueError("an image evaluation phase is required")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("provenance is required")
    phase = config["phase"]
    policies = []
    labels = {}
    for item in config.get("policies", []):
        effective = _policy(item["effective_policy"])
        policy_hash = digest(effective)
        supplied = item.get("policy_hash")
        if supplied is not None and supplied != policy_hash:
            raise ValueError("policy hash does not match effective policy")
        policies.append({"policy_hash": policy_hash, "effective_policy": effective})
        labels.setdefault(policy_hash, item.get("policy_id", policy_hash[:12]))
    if len({item["policy_hash"] for item in policies}) != len(policies):
        raise ValueError("duplicate effective policy")
    legacy_effective = _policy(config["legacy_policy"]["effective_policy"])
    legacy_hash = digest(legacy_effective)
    labels.setdefault(legacy_hash, config["legacy_policy"].get("policy_id", "legacy"))

    jobs = []
    plain = {}
    legacy = {}
    if phase in {"screen", "heldout"}:
        for topic in config["topics"]:
            for seed in config["seeds"]:
                item = _job(
                    {
                        "phase": phase,
                        "role": "plain",
                        "topic_id": topic["id"],
                        "history_id": None,
                        "seed": seed,
                        "prompt": topic["generation_prompt"],
                        "target_text_id": "target:" + topic["id"],
                        "target_text": topic["target_text"],
                        "policy_hash": legacy_hash,
                        "effective_policy": legacy_effective,
                        "personalization": None,
                    }
                )
                jobs.append(item)
                plain[(topic["id"], seed)] = item["job_id"]
        for topic in config["topics"]:
            for history in config["histories"]:
                for seed in config["seeds"]:
                    personalization = _personalization(
                        history["refs"],
                        topic["generation_prompt"],
                        {"effective_policy": legacy_effective},
                        provenance,
                    )
                    item = _job(
                        {
                            "phase": phase,
                            "role": "legacy",
                            "topic_id": topic["id"],
                            "history_id": history["id"],
                            "seed": seed,
                            "prompt": topic["generation_prompt"],
                            "target_text_id": "target:" + topic["id"],
                            "target_text": topic["target_text"],
                            "refs": copy.deepcopy(history["refs"]),
                            "policy_hash": legacy_hash,
                            "effective_policy": legacy_effective,
                            "personalization": personalization,
                            "plain_job_id": plain[(topic["id"], seed)],
                        }
                    )
                    item["legacy_job_id"] = item["job_id"]
                    # Links are routing metadata and do not alter the image contract.
                    jobs.append(item)
                    legacy[(topic["id"], history["id"], seed)] = item["job_id"]

    for policy in policies:
        for topic in config["topics"]:
            for history in config["histories"]:
                for seed in config["seeds"]:
                    personalization = _personalization(
                        history["refs"], topic["generation_prompt"], policy, provenance
                    )
                    contract = {
                        "phase": phase,
                        "role": "candidate",
                        "topic_id": topic["id"],
                        "history_id": history["id"],
                        "seed": seed,
                        "prompt": topic["generation_prompt"],
                        "target_text_id": "target:" + topic["id"],
                        "target_text": topic["target_text"],
                        "refs": copy.deepcopy(history["refs"]),
                        "policy_hash": policy["policy_hash"],
                        "effective_policy": policy["effective_policy"],
                        "personalization": personalization,
                    }
                    item = _job(contract)
                    if phase in {"screen", "heldout"}:
                        item["plain_job_id"] = plain[(topic["id"], seed)]
                        item["legacy_job_id"] = legacy[
                            (topic["id"], history["id"], seed)
                        ]
                    jobs.append(item)

    maximum = config.get("limits", {}).get("max_images", 512)
    maximum_attempts = config.get("limits", {}).get("max_attempts", 2)
    if type(maximum) is not int or not 1 <= maximum <= 512:
        raise ValueError("max_images must be an integer from 1 through 512")
    if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 2:
        raise ValueError("max_attempts must be an integer from 1 through 2")
    if len(jobs) > maximum:
        raise ValueError(f"experiment exceeds the {maximum} image limit")
    expected = config.get("expected_jobs")
    if expected is not None and len(jobs) != expected:
        raise ValueError(f"expected {expected} jobs, got {len(jobs)}")
    ids = [item["job_id"] for item in jobs]
    paths = [item["path"] for item in jobs]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("duplicate job or artifact contract")

    identity_jobs = []
    for item in jobs:
        identity_jobs.append(copy.deepcopy(item))
    identity = {
        "schema_version": 1,
        "phase": phase,
        "generation": copy.deepcopy(config["generation"]),
        "evaluator": copy.deepcopy(config["evaluator"]),
        "limits": copy.deepcopy(config.get("limits", {})),
        "topics": copy.deepcopy(config["topics"]),
        "histories": copy.deepcopy(config["histories"]),
        "seeds": copy.deepcopy(config["seeds"]),
        "policies": policies,
        "legacy_policy": {
            "policy_hash": legacy_hash,
            "effective_policy": legacy_effective,
        },
        "parents": copy.deepcopy(config.get("parents", {})),
        "catalog": copy.deepcopy(config.get("catalog")),
        "provenance": copy.deepcopy(provenance),
        "jobs": identity_jobs,
    }
    return {
        "schema_version": 1,
        "experiment_hash": digest(identity),
        "identity": identity,
        "jobs": jobs,
        "policy_labels": labels,
    }


def register_experiment(root, experiment, *, resume):
    """Create or validate an immutable manifest and mutable checkpoint."""
    root = Path(root)
    if (root / "manifest.json").exists() or root.name == experiment["experiment_hash"]:
        directory = root
    else:
        directory = root / experiment["experiment_hash"]
    manifest_path = directory / "manifest.json"
    checkpoint_path = directory / "checkpoint.json"
    if not resume:
        if manifest_path.exists() or checkpoint_path.exists():
            raise ValueError("experiment already registered; use --resume")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "experiment_hash": experiment["experiment_hash"],
            "identity": experiment["identity"],
            "policy_labels": experiment.get("policy_labels", {}),
        }
        checkpoint = {
            "schema_version": 1,
            "manifest_hash": experiment["experiment_hash"],
            "created_at": _now(),
            "runs": [],
            "jobs": {
                item["job_id"]: {"status": "not_run", "attempts": 0}
                for item in experiment["jobs"]
            },
        }
        _write(manifest_path, manifest)
        _write(checkpoint_path, checkpoint)
        return directory, checkpoint
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("resume requires a registered manifest and checkpoint")
    manifest, checkpoint = _read(manifest_path), _read(checkpoint_path)
    if (
        manifest.get("experiment_hash") != experiment["experiment_hash"]
        or digest(manifest.get("identity")) != experiment["experiment_hash"]
        or manifest.get("identity") != experiment["identity"]
        or checkpoint.get("manifest_hash") != experiment["experiment_hash"]
    ):
        raise ValueError("resume manifest hash or identity mismatch")
    expected_ids = {item["job_id"] for item in experiment["jobs"]}
    if set(checkpoint.get("jobs", {})) != expected_ids:
        raise ValueError("resume job set does not match manifest hash")
    jobs = {item["job_id"]: item for item in experiment["jobs"]}
    normalized = False
    now = _now()
    for run in checkpoint.setdefault("runs", []):
        if run.get("status") == "running":
            run.update(
                {
                    "status": "interrupted",
                    "finished_at": now,
                    "error": "interrupted_before_resume",
                }
            )
            normalized = True
    for job_id, state in checkpoint["jobs"].items():
        attempts = state.get("attempts")
        if type(attempts) is not int or not 0 <= attempts <= 2:
            raise ValueError("invalid checkpoint attempt count")
        if state.get("status") == "running":
            state.update(
                {
                    "status": "failed",
                    "finished_at": now,
                    "error": "interrupted_before_resume",
                }
            )
            history = state.get("attempt_history", [])
            if history and history[-1].get("status") == "running":
                history[-1].update({"status": "interrupted", "finished_at": now})
            normalized = True
        if state.get("status") == "done":
            job = jobs[job_id]
            artifact = directory / job["path"]
            if (
                state.get("contract_hash") != job["contract_hash"]
                or not artifact.is_file()
                or state.get("sha256") != file_hash(artifact)
            ):
                raise ValueError(f"completed artifact integrity failure: {job_id}")
            event = state.get("event")
            expected_personalization = job.get("personalization")
            expected_personalization_hash = (
                expected_personalization.get("personalization_hash")
                if expected_personalization
                else None
            )
            if not isinstance(event, dict) or any(
                (
                    event.get("id") != job_id,
                    Path(event.get("path", "")).resolve() != artifact.resolve(),
                    event.get("sha256") != state.get("sha256"),
                    event.get("seed") != job["seed"],
                    event.get("prompt") != job["prompt"],
                    event.get("settings") != experiment["identity"]["generation"],
                    event.get("effective_policy") != job["effective_policy"],
                    event.get("policy_hash") != job["policy_hash"],
                    event.get("personalization_hash") != expected_personalization_hash,
                )
            ):
                raise ValueError(f"completed worker event contract mismatch: {job_id}")
    if normalized:
        _write(checkpoint_path, checkpoint)
    return directory, checkpoint


def _vector(value):
    if not isinstance(value, list) or not value:
        raise ValueError("embedding must be a non-empty vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("embedding must be finite")
    return result


def _cosine(left, right):
    left, right = _vector(left), _vector(right)
    if len(left) != len(right):
        raise ValueError("embedding shapes differ")
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("zero embedding vector")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _history(image, refs, texts):
    components = []
    total = 0.0
    weighted = 0.0
    for ref in refs:
        weight = float(ref["weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("reference weight must be positive and finite")
        key = "ref:" + ref["ref_id"]
        if key not in texts:
            raise KeyError(key)
        cosine = _cosine(image, texts[key])
        components.append({"ref_id": ref["ref_id"], "weight": weight, "cosine": cosine})
        total += weight
        weighted += weight * cosine
    return weighted / total, components


def score_records(manifest, embeddings):
    """Score all declared jobs; absent data remains explicit and ineligible."""
    jobs = manifest.get("identity", {}).get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("manifest jobs are required")
    ids = [item.get("job_id") for item in jobs]
    paths = [item.get("path") for item in jobs]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("duplicate job or artifact contract")
    images = embeddings.get("images", {})
    if not isinstance(images, dict) or set(images) - set(ids):
        raise ValueError("unknown or duplicate image embedding rows")
    texts = embeddings.get("texts", {})
    conditioning = embeddings.get("conditioning", {})
    records = []
    by_id = {}
    for job in jobs:
        record = {
            key: copy.deepcopy(job.get(key))
            for key in (
                "job_id",
                "role",
                "topic_id",
                "history_id",
                "seed",
                "policy_hash",
                "plain_job_id",
                "legacy_job_id",
            )
            if key in job
        }
        record.update(
            {
                "status": "unmeasured",
                "reasons": [],
                "target_score": None,
                "history_score": None,
                "history_components": [],
                "seconds": None,
            }
        )
        image_row = images.get(job["job_id"])
        if image_row is None:
            record["reasons"].append("missing_image_embedding")
        elif image_row.get("status") in {"failed", "not_run", "missing"}:
            image_status = image_row["status"]
            record["status"] = "failed" if image_status == "failed" else "unmeasured"
            record["reasons"].append(image_status)
            if image_row.get("error"):
                record["error"] = image_row["error"]
        elif image_row.get("valid") is not True:
            record["status"] = "failed"
            record["reasons"].append(image_row.get("invalid_reason", "invalid_image"))
        else:
            try:
                target_key = job["target_text_id"]
                if target_key not in texts:
                    raise KeyError(target_key)
                record["target_score"] = _cosine(
                    image_row["embedding"], texts[target_key]
                )
                if job["role"] != "plain":
                    record["history_score"], record["history_components"] = _history(
                        image_row["embedding"], job["refs"], texts
                    )
                record["seconds"] = image_row.get("seconds")
                record["image_sha256"] = image_row.get("sha256")
                record["status"] = "measured"
            except KeyError as error:
                record["target_score"] = None
                record["history_score"] = None
                record["history_components"] = []
                record["reasons"].append(f"missing_embedding:{error}")
            except (TypeError, ValueError) as error:
                record["status"] = "failed"
                record["target_score"] = None
                record["history_score"] = None
                record["history_components"] = []
                record["reasons"].append(f"invalid_embedding:{error}")
        condition = conditioning.get(job["job_id"])
        if condition is not None:
            record["conditioning_target_align"] = {
                name: value.get("cosine") if value.get("valid") else None
                for name, value in condition.items()
                if name in {"hidden", "pooled"}
            }
        records.append(record)
        by_id[job["job_id"]] = record

    for job, record in zip(jobs, records):
        if record["status"] != "measured" or job["role"] == "plain":
            continue
        image_row = images[job["job_id"]]
        plain = by_id.get(job.get("plain_job_id"))
        plain_image = images.get(job.get("plain_job_id"))
        if plain and plain["status"] == "measured" and plain_image:
            try:
                plain_history, _ = _history(
                    plain_image["embedding"], job["refs"], texts
                )
                record["plain_context_history_score"] = plain_history
                record["delta_vs_plain"] = {
                    "target_score": record["target_score"] - plain["target_score"],
                    "history_score": record["history_score"] - plain_history,
                }
            except (KeyError, TypeError, ValueError):
                record["reasons"].append("invalid_plain_pair")
        legacy = by_id.get(job.get("legacy_job_id"))
        if legacy and legacy["status"] == "measured":
            record["delta_vs_legacy"] = {
                "target_score": record["target_score"] - legacy["target_score"],
                "history_score": record["history_score"] - legacy["history_score"],
            }
    return records


def attach_parent_pairs(records, parent_records):
    """Pair refine images with immutable screen plain/legacy measurements."""
    plain = {
        (item.get("topic_id"), item.get("seed")): item
        for item in parent_records
        if item.get("role") == "plain"
    }
    legacy = {
        (item.get("topic_id"), item.get("history_id"), item.get("seed")): item
        for item in parent_records
        if item.get("role") == "legacy"
    }
    result = copy.deepcopy(records)
    for record in result:
        if record.get("role") != "candidate" or record.get("status") != "measured":
            continue
        plain_record = plain.get((record.get("topic_id"), record.get("seed")))
        legacy_record = legacy.get(
            (record.get("topic_id"), record.get("history_id"), record.get("seed"))
        )
        if (
            not plain_record
            or not legacy_record
            or plain_record.get("status") != "measured"
            or legacy_record.get("status") != "measured"
            or legacy_record.get("plain_context_history_score") is None
        ):
            record["status"] = "unmeasured"
            record.setdefault("reasons", []).append("missing_parent_pair")
            continue
        record["plain_job_id"] = plain_record["job_id"]
        record["legacy_job_id"] = legacy_record["job_id"]
        record["delta_vs_plain"] = {
            "target_score": record["target_score"] - plain_record["target_score"],
            "history_score": record["history_score"]
            - legacy_record["plain_context_history_score"],
        }
        record["delta_vs_legacy"] = {
            "target_score": record["target_score"] - legacy_record["target_score"],
            "history_score": record["history_score"] - legacy_record["history_score"],
        }
    return result


def _mean(values):
    return sum(values) / len(values)


def select_candidates(records, rules):
    """Apply fixed completeness, numerical, target, and history gates."""
    stage = rules.get("stage")
    if stage not in {"screen", "refine", "heldout"}:
        raise ValueError("selection stage must be screen, refine, or heldout")
    expected = rules.get("expected_cases")
    numerical = rules.get("numerical", {})
    grouped = {}
    seen_jobs = set()
    seen_cases = set()
    for record in records:
        if record.get("role") != "candidate":
            continue
        job_id = record.get("job_id")
        if not isinstance(job_id, str) or not job_id or job_id in seen_jobs:
            raise ValueError("duplicate record job_id")
        seen_jobs.add(job_id)
        case_key = (
            record.get("policy_hash"),
            record.get("topic_id"),
            record.get("history_id"),
            record.get("seed"),
        )
        if case_key in seen_cases:
            raise ValueError("duplicate case key")
        seen_cases.add(case_key)
        grouped.setdefault(record.get("policy_hash"), []).append(record)
    candidates = []
    for policy_hash, rows in sorted(grouped.items()):
        reasons = []
        evidence = numerical.get(policy_hash)
        if evidence is None:
            status = "unmeasured"
            reasons.append("missing_exact_policy_numerical_evidence")
        elif not evidence.get("passed"):
            status = "fail"
            reasons.append("numerical_failure")
        elif any(row.get("status") == "failed" for row in rows):
            status = "fail"
            reasons.append("failed_case")
            reasons.extend(
                reason
                for row in rows
                if row.get("status") == "failed"
                for reason in row.get("reasons", ["failed_case"])
                if reason not in reasons
            )
            if len(rows) != expected:
                reasons.append("incomplete_case_set")
        elif len(rows) != expected or any(
            row.get("status") != "measured" for row in rows
        ):
            status = "unmeasured"
            if len(rows) != expected:
                reasons.append("incomplete_case_set")
            reasons.extend(
                reason
                for row in rows
                if row.get("status") != "measured"
                for reason in row.get("reasons", ["unmeasured_case"])
                if reason not in reasons
            )
        elif any("delta_vs_legacy" not in row for row in rows):
            status = "unmeasured"
            reasons.append("missing_legacy_pair")
        else:
            history_delta = _mean(
                [row["delta_vs_legacy"]["history_score"] for row in rows]
            )
            target_delta = _mean(
                [row["delta_vs_legacy"]["target_score"] for row in rows]
            )
            status = "pass"
            if target_delta < rules["target_non_degradation"]:
                status = "fail"
                reasons.append("target_non_degradation")
            if stage == "heldout" and history_delta < rules["history_improvement"]:
                status = "fail"
                reasons.append("history_improvement")
        history_delta = (
            _mean([row["delta_vs_legacy"]["history_score"] for row in rows])
            if rows and all("delta_vs_legacy" in row for row in rows)
            else None
        )
        target_delta = (
            _mean([row["delta_vs_legacy"]["target_score"] for row in rows])
            if rows and all("delta_vs_legacy" in row for row in rows)
            else None
        )
        seconds = [row.get("seconds") for row in rows if row.get("seconds") is not None]
        candidates.append(
            {
                "policy_hash": policy_hash,
                "status": status,
                "reasons": reasons,
                "case_count": len(rows),
                "mean_history_delta_vs_legacy": history_delta,
                "mean_target_delta_vs_legacy": target_delta,
                "mean_generation_seconds": _mean(seconds) if seconds else None,
                "numerical_evidence_hash": evidence.get("evidence_hash")
                if evidence
                else None,
            }
        )
    eligible = [item for item in candidates if item["status"] == "pass"]
    eligible.sort(
        key=lambda item: (
            -item["mean_history_delta_vs_legacy"],
            -item["mean_target_delta_vs_legacy"],
            item["mean_generation_seconds"]
            if item["mean_generation_seconds"] is not None
            else math.inf,
            item["policy_hash"],
        )
    )
    selected = eligible[:2] if stage in {"screen", "refine"} else eligible
    if stage in {"screen", "refine"} and len(selected) < 2:
        decision_status = "insufficient_candidates"
    else:
        decision_status = "selected" if selected else "keep_legacy"
    result = {
        "schema_version": 1,
        "stage": stage,
        "status": decision_status,
        "candidates": candidates,
        "selected": copy.deepcopy(selected),
        "rules": {
            key: copy.deepcopy(value)
            for key, value in rules.items()
            if key not in {"numerical", "policy_registry"}
        },
    }
    result["decision_hash"] = digest(result)
    return result


def _metric_summary(rows):
    measured = [item for item in rows if item.get("status") == "measured"]
    metric_names = (
        "target_score",
        "history_score",
    )
    result = {
        "case_count": len(rows),
        "measured_count": len(measured),
        "status_counts": {},
    }
    for item in rows:
        status = item.get("status", "unmeasured")
        result["status_counts"][status] = result["status_counts"].get(status, 0) + 1
    for name in metric_names:
        values = [float(item[name]) for item in measured if item.get(name) is not None]
        result["mean_" + name] = _mean(values) if values else None
    for baseline in ("plain", "legacy"):
        for name in metric_names:
            values = [
                float(item["delta_vs_" + baseline][name])
                for item in measured
                if item.get("delta_vs_" + baseline, {}).get(name) is not None
            ]
            result[f"mean_{name}_delta_vs_{baseline}"] = (
                _mean(values) if values else None
            )
    return result


def summarize_records(records, stage, rules):
    """Report all statuses plus policy/topic/history metrics without scalar fusion."""
    policies = {}
    for policy_hash in sorted(
        {item.get("policy_hash") for item in records if item.get("role") == "candidate"}
    ):
        rows = [
            item
            for item in records
            if item.get("role") == "candidate"
            and item.get("policy_hash") == policy_hash
        ]
        summary = _metric_summary(rows)
        summary["by_topic"] = {
            value: _metric_summary(
                [item for item in rows if item.get("topic_id") == value]
            )
            for value in sorted({item.get("topic_id") for item in rows})
        }
        summary["by_history"] = {
            value: _metric_summary(
                [item for item in rows if item.get("history_id") == value]
            )
            for value in sorted({item.get("history_id") for item in rows})
        }
        if stage == "heldout":
            intervals = {}
            for metric, field in (
                ("target_delta_vs_legacy", "target_score"),
                ("history_delta_vs_legacy", "history_score"),
            ):
                clusters = {
                    history: [
                        item["delta_vs_legacy"][field]
                        for item in rows
                        if item.get("status") == "measured"
                        and item.get("history_id") == history
                        and item.get("delta_vs_legacy", {}).get(field) is not None
                    ]
                    for history in sorted({item.get("history_id") for item in rows})
                }
                if clusters and all(clusters.values()):
                    intervals[metric] = cluster_bootstrap_interval(
                        clusters,
                        draws=rules.get("bootstrap_draws", 2000),
                        seed=rules.get("bootstrap_seed", 0),
                        confidence=rules.get("bootstrap_confidence", 0.95),
                    )
            summary["bootstrap_95"] = intervals
        policies[policy_hash] = summary
    return {
        "stage": stage,
        "record_count": len(records),
        "status_counts": _metric_summary(records)["status_counts"],
        "policies": policies,
    }


def validate_heldout_catalog(config, *, catalog_loader=None):
    """Bind heldout refs to the authoritative, fully reviewed v2 catalog."""
    if catalog_loader is None:
        try:
            from .catalog import load_catalog
        except ImportError as error:
            raise ValueError(
                "heldout catalog is not prepared; complete Task 4 catalog generation and review"
            ) from error
        catalog_loader = load_catalog
    try:
        catalog = catalog_loader("catalog-v2", reviewed_only=True)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "heldout catalog is not prepared; complete Task 4 catalog generation and review"
        ) from error
    if not isinstance(catalog, dict) or catalog.get("catalog_id") != "catalog-v2":
        raise ValueError("heldout catalog loader returned the wrong catalog")
    eligible, all_cards = catalog.get("cards"), catalog.get("all_cards")
    if (
        not isinstance(eligible, list)
        or not isinstance(all_cards, list)
        or len(all_cards) != 64
        or len(eligible) != len(all_cards)
    ):
        raise ValueError(
            "heldout catalog is not fully reviewed: expected 64 of 64 cards"
        )
    if not isinstance(catalog.get("catalog_hash"), str) or not catalog["catalog_hash"]:
        raise ValueError("heldout catalog hash is missing")
    cards = {}
    for card in eligible:
        card_id = card.get("id") if isinstance(card, dict) else None
        if not card_id or card_id in cards:
            raise ValueError("heldout catalog has invalid or duplicate card IDs")
        cards[card_id] = card

    used = set()
    for history in config.get("histories", []):
        merged = {}
        for card_id in history.get("cards", []):
            card = cards.get(card_id)
            if card is None:
                raise ValueError(f"heldout card is not reviewed: {card_id}")
            aspects = card.get("aspects")
            if not isinstance(aspects, dict):
                raise TypeError(f"heldout card aspects are missing: {card_id}")
            for aspect in history.get("aspects", []):
                text = aspects.get(aspect)
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"heldout card aspect is missing: {card_id}:{aspect}"
                    )
                merged[text] = merged.get(text, 0.0) + 1.0
            used.add(card_id)
        expected = sorted(
            (ref["text"], float(ref["weight"])) for ref in history.get("refs", [])
        )
        if sorted(merged.items()) != expected:
            raise ValueError(
                f"heldout resolved refs do not match catalog: {history['id']}"
            )
    identity = {
        "catalog_id": "catalog-v2",
        "catalog_hash": catalog["catalog_hash"],
        "reviewed_card_count": len(eligible),
        "card_ids": sorted(used),
    }
    return {**identity, "identity_hash": digest(identity)}


def cluster_bootstrap_interval(values, *, draws=2000, seed=0, confidence=0.95):
    """Paired-difference bootstrap clustered by synthetic history fixture."""
    if not isinstance(values, dict) or not values or draws < 1:
        raise ValueError("bootstrap clusters and draws are required")
    clusters = []
    for name, rows in sorted(values.items()):
        if not rows or not all(math.isfinite(float(item)) for item in rows):
            raise ValueError(f"invalid bootstrap cluster: {name}")
        clusters.append(_mean([float(item) for item in rows]))
    generator = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(
            _mean([generator.choice(clusters) for _ in range(len(clusters))])
        )
    samples.sort()
    tail = (1 - confidence) / 2
    lower_index = min(draws - 1, max(0, math.floor(tail * draws)))
    upper_index = min(draws - 1, max(0, math.ceil((1 - tail) * draws) - 1))
    return {
        "unit": "synthetic_history",
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "mean": _mean(clusters),
        "lower": samples[lower_index],
        "upper": samples[upper_index],
        "diagnostic_only": True,
    }


def prepare_evaluator(
    config, output, *, snapshot_download=None, versions=None, local_files_only=False
):
    """Resolve and hash the exact local CLIP snapshot used by evaluation."""
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as download

        snapshot_download = download
    snapshot = Path(
        snapshot_download(
            repo_id=config["repo_id"],
            revision=config.get("revision", "main"),
            allow_patterns=sorted(EVALUATOR_FILES),
            ignore_patterns=["pytorch_model.bin"],
            local_files_only=local_files_only,
        )
    ).resolve()
    revision = snapshot.name
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("evaluator snapshot did not resolve to a Hub commit")
    missing = sorted(
        name for name in EVALUATOR_FILES if not (snapshot / name).is_file()
    )
    if missing:
        raise ValueError("evaluator snapshot is incomplete: " + ", ".join(missing))
    if config.get("expected_revision") and config["expected_revision"] != revision:
        raise ValueError("resolved evaluator revision differs from expected revision")
    if versions is None:
        import huggingface_hub
        import transformers

        versions = {
            "python": sys.version.split()[0],
            "huggingface_hub": huggingface_hub.__version__,
            "transformers": transformers.__version__,
        }
    files = {
        name: {
            "sha256": file_hash(snapshot / name),
            "size": (snapshot / name).stat().st_size,
        }
        for name in sorted(EVALUATOR_FILES)
    }
    identity = {
        "schema_version": 1,
        "repo_id": config["repo_id"],
        "requested_revision": config.get("revision", "main"),
        "resolved_revision": revision,
        "weight_format": config.get("weight_format", "safetensors"),
        "files": files,
        "versions": copy.deepcopy(versions),
    }
    result = {
        **identity,
        "snapshot_path": str(snapshot),
        "evaluator_hash": digest(identity),
    }
    _write(output, result)
    return result


def validate_evaluator_preparation(path, *, expected=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError("evaluator is not prepared; run prepare_evaluation.py")
    manifest = _read(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported evaluator preparation schema")
    snapshot = Path(manifest.get("snapshot_path", ""))
    files = manifest.get("files")
    if not snapshot.is_dir() or not isinstance(files, dict):
        raise ValueError("evaluator preparation is invalid")
    missing = sorted(EVALUATOR_FILES - set(files))
    if missing:
        raise ValueError("evaluator required file missing: " + ", ".join(missing))
    revision = manifest.get("resolved_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
        or snapshot.name != revision
    ):
        raise ValueError("evaluator resolved revision is invalid")
    identity = {
        key: copy.deepcopy(manifest.get(key))
        for key in (
            "schema_version",
            "repo_id",
            "requested_revision",
            "resolved_revision",
            "weight_format",
            "files",
            "versions",
        )
    }
    if manifest.get("evaluator_hash") != digest(identity):
        raise ValueError("evaluator identity hash mismatch")
    if expected:
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"evaluator preparation {key} mismatch")
    for name, expected_file in files.items():
        file_path = snapshot / name
        if (
            not file_path.is_file()
            or file_path.stat().st_size != expected_file.get("size")
            or file_hash(file_path) != expected_file.get("sha256")
        ):
            raise ValueError(f"evaluator file hash mismatch: {name}")
    return manifest
