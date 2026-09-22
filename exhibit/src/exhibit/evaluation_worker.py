"""GPU-side FAN diagnostics; torch and model libraries stay lazy."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

from .config import CONFIG, FAN_UPSTREAM, write_json
from .domain import digest, file_hash
from .evaluation import runtime_file_provenance
from .fan_adapter import (
    embed_gain,
    encode_conditioning,
    freeze_policy,
    profiling_argument,
    thaw_policy,
)
from .workers import build_encoder, load_pipeline

NO_REF_LIMITS = {"normalized_rmse": 1e-3, "mean_cosine": 0.9999}
PERSONAL_LIMITS = {"normalized_rmse": 5e-3, "mean_cosine": 0.999}


def emit(kind, **data):
    print(
        json.dumps({"type": kind, **data}, ensure_ascii=False, allow_nan=False),
        flush=True,
    )


def validate_policy_specs(specs):
    """Validate explicit effective policies and key results by their contents."""
    if not isinstance(specs, list) or not specs:
        raise ValueError("policies must be a non-empty list")
    result = []
    seen_ids = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise TypeError("each policy spec must be an object")
        policy_id = spec.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id or policy_id in seen_ids:
            raise ValueError("policy_id must be present and unique")
        if "effective_policy" not in spec:
            raise ValueError("effective_policy is required")
        effective = thaw_policy(freeze_policy(spec["effective_policy"]))
        result.append(
            {
                "policy_id": policy_id,
                "policy_hash": digest(effective),
                "effective_policy": effective,
            }
        )
        seen_ids.add(policy_id)
    return result


def tensor_metrics(candidate, base):
    """Section 9.3 tensor metrics, including finite and zero-vector checks."""
    import torch

    if tuple(candidate.shape) != tuple(base.shape):
        raise ValueError("tensor shapes differ")
    left = candidate.detach().float()
    right = base.detach().float()
    left_finite_vectors = torch.isfinite(left).all(dim=-1)
    right_finite_vectors = torch.isfinite(right).all(dim=-1)
    finite = bool(left_finite_vectors.all() and right_finite_vectors.all())
    left_norm = torch.linalg.vector_norm(
        torch.where(torch.isfinite(left), left, torch.zeros_like(left)), dim=-1
    )
    right_norm = torch.linalg.vector_norm(
        torch.where(torch.isfinite(right), right, torch.zeros_like(right)), dim=-1
    )
    result = {
        "shape": list(candidate.shape),
        "finite": finite,
        "candidate_nonfinite_values": int((~torch.isfinite(left)).sum().item()),
        "base_nonfinite_values": int((~torch.isfinite(right)).sum().item()),
        "candidate_zero_vectors": int(
            (left_finite_vectors & (left_norm == 0)).sum().item()
        ),
        "base_zero_vectors": int(
            (right_finite_vectors & (right_norm == 0)).sum().item()
        ),
    }
    if not finite:
        result.update(
            {
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "normalized_rmse": None,
                "mean_cosine": None,
                "min_cosine": None,
            }
        )
        return result

    delta = left - right
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    base_rms = torch.sqrt(torch.mean(right * right))
    normalized_rmse = torch.sqrt(torch.mean(delta * delta)) / torch.clamp(
        base_rms, min=1e-6
    )
    result.update(
        {
            "max_abs_diff": float(delta.abs().max().item()),
            "mean_abs_diff": float(delta.abs().mean().item()),
            "normalized_rmse": float(normalized_rmse.item()),
            "mean_cosine": float(cosine.mean().item()),
            "min_cosine": float(cosine.min().item()),
        }
    )
    return result


def compare_conditioning(candidate, base, limits):
    result = {
        "hidden": tensor_metrics(candidate["hidden"], base["hidden"]),
        "pooled": tensor_metrics(candidate["pooled"], base["pooled"]),
        "limits": copy.deepcopy(limits),
    }
    result["passed"] = all(
        value["finite"]
        and value["candidate_zero_vectors"] == 0
        and value["base_zero_vectors"] == 0
        and value["normalized_rmse"] <= limits["normalized_rmse"]
        and value["mean_cosine"] >= limits["mean_cosine"]
        for value in (result["hidden"], result["pooled"])
    )
    return result


def _raw_official(encoder, prompt, refs, policy):
    import torch

    texts = [ref["text"] for ref in refs] if refs else None
    weights = [float(ref["weight"]) for ref in refs] if refs else None
    with torch.no_grad():
        hidden, pooled = encoder(
            prompt,
            texts,
            weight=weights,
            alpha=policy["alpha"] if refs else None,
            skip=policy["skip"],
            sample_size=profiling_argument(policy),
            skip_pa=list(policy["skip_pa"]),
            use_attn_mask=policy["use_attn_mask"],
        )
    return {"hidden": hidden.to(torch.float16), "pooled": pooled.to(torch.float16)}


def _raw_eos_pooled(big_g, prompt, refs, policy):
    """The upstream bigG pooled pass of ``stable_diffusion_xl`` pooled at EOS."""
    import torch

    with torch.no_grad():
        hidden = big_g(
            prompt,
            [ref["text"] for ref in refs],
            weight=[float(ref["weight"]) for ref in refs],
            alpha=policy["alpha"],
            pooling=False,
            sample_size=profiling_argument(policy),
            skip=-1,
            skip_pa=list(policy["skip_pa"]),
            use_attn_mask=policy["use_attn_mask"],
            normalize=False,
        )
        pooled = big_g.pool_text_hidden_state(
            big_g.normalize_text_hidden_state(hidden), prompt
        )
        pooled = big_g.projection_text_hidden_state(pooled)
    return pooled.to(torch.float16)


def _expected_direct(encoder, direct_encoder, prompt, refs, policy):
    """What the adapter must reproduce, built only from upstream calls."""
    import torch

    direct_fan = _raw_official(direct_encoder, prompt, refs, policy)
    hidden = direct_fan["hidden"]
    pooled = direct_fan["pooled"]
    gain = embed_gain(policy)
    direct_plain = None
    if policy["pooled_mode"] != "fan" or gain != 1.0:
        direct_plain = _raw_official(direct_encoder, prompt, None, policy)
    if policy["pooled_mode"] == "plain":
        pooled = direct_plain["pooled"]
    elif policy["pooled_mode"] == "fan_eos":
        pooled = _raw_eos_pooled(
            encoder._fan_components["clip_g"], prompt, refs, policy
        )
    if gain != 1.0:
        base = direct_plain["hidden"].float()
        hidden = (base + gain * (hidden.float() - base)).to(torch.float16)
    return {"hidden": hidden, "pooled": pooled, "official": direct_fan}


def _pipeline_conditioning(pipe, prompt):
    import torch

    with torch.no_grad():
        hidden, _, pooled, _ = pipe.encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return {"hidden": hidden.to(torch.float16), "pooled": pooled.to(torch.float16)}


def _policy_variant(policy, **changes):
    value = thaw_policy(freeze_policy(policy))
    value.update(copy.deepcopy(changes))
    return thaw_policy(freeze_policy(value))


def _reference_variants(refs):
    combined = copy.deepcopy(refs)
    first = copy.deepcopy(combined[0])
    half = float(first["weight"]) / 2
    left, right = copy.deepcopy(first), copy.deepcopy(first)
    left["weight"] = half
    right["weight"] = half
    duplicate = [left, right, *copy.deepcopy(combined[1:])]
    scale = [
        {**copy.deepcopy(ref), "weight": float(ref["weight"]) * 3} for ref in combined
    ]
    order = list(reversed(copy.deepcopy(combined)))
    return {
        "duplicate_split": duplicate,
        "common_weight_scale": scale,
        "reference_order": order,
    }


def _token_detector_report(pipe, encoder, texts):
    import torch

    rows = []
    components = encoder._fan_components
    sources = (
        ("clip_l", components["clip_l"], pipe.tokenizer, pipe.text_encoder),
        ("clip_g", components["clip_g"], pipe.tokenizer_2, pipe.text_encoder_2),
    )
    for encoder_name, fan, tokenizer, model in sources:
        for label, text in texts:
            batch = tokenizer(
                text,
                padding="max_length",
                max_length=fan.n_token,
                truncation=True,
                return_tensors="pt",
            )
            ids = batch["input_ids"].to("cuda")
            attention = batch["attention_mask"].to("cuda")
            with torch.no_grad():
                states = model(
                    input_ids=ids,
                    attention_mask=None,
                    output_hidden_states=True,
                ).hidden_states
                unnormalized = states[-1]
                normalized = fan.normalize_text_hidden_state(unnormalized)
                eos_id = model.text_model.eos_token_id
                if eos_id == 2:
                    eos_index = ids.to(dtype=torch.int).argmax(dim=-1)
                else:
                    eos_index = (ids.to(dtype=torch.int) == eos_id).int().argmax(dim=-1)
                rows.append(
                    {
                        "encoder": encoder_name,
                        "text_id": label,
                        "text": text,
                        "target_length": int(attention.sum().item()),
                        "input_eos_index": int(eos_index.item()),
                        "detector_on_unnormalized": int(
                            fan.decoder(unnormalized).item()
                        ),
                        "detector_on_normalized": int(fan.decoder(normalized).item()),
                        "untruncated_token_count": len(
                            tokenizer(text, truncation=False)["input_ids"]
                        ),
                    }
                )
    return rows


def _personalized_detector_report(encoder, prompt, refs, policy):
    """Observe the official bigG detector before/after final norm, then restore it."""
    import torch

    fan = encoder._fan_components["clip_g"]
    tokenizer = fan.processor
    model = fan.model
    batch = tokenizer(
        prompt,
        padding="max_length",
        max_length=fan.n_token,
        truncation=True,
        return_tensors="pt",
    )
    ids = batch["input_ids"].to("cuda")
    attention = batch["attention_mask"].to("cuda")
    eos_id = model.text_model.eos_token_id
    if eos_id == 2:
        eos_index = ids.to(dtype=torch.int).argmax(dim=-1)
    else:
        eos_index = (ids.to(dtype=torch.int) == eos_id).int().argmax(dim=-1)

    rows = []
    alphas = []
    for alpha in (0.0, float(policy["alpha"])):
        if alpha not in alphas:
            alphas.append(alpha)
    for alpha in alphas:
        calls = []
        original = fan.decoder.forward

        def recording_forward(
            self, hidden_state, training=False, _original=original, _calls=calls
        ):
            official = _original(hidden_state, training=training)
            normalized = fan.normalize_text_hidden_state(hidden_state)
            normalized_index = _original(normalized, training=training)
            _calls.append(
                {
                    "hidden_shape": list(hidden_state.shape),
                    "detector_on_unnormalized": int(official.item()),
                    "detector_on_normalized": int(normalized_index.item()),
                }
            )
            return official

        fan.decoder.forward = types.MethodType(recording_forward, fan.decoder)
        try:
            observed_policy = _policy_variant(policy, alpha=alpha)
            _raw_official(encoder, prompt, refs, observed_policy)
        finally:
            fan.decoder.forward = original
        rows.append(
            {
                "alpha": alpha,
                "target_length": int(attention.sum().item()),
                "input_eos_index": int(eos_index.item()),
                "detector_calls": calls,
                "decoder_restored": fan.decoder.forward is original,
            }
        )
    return rows


def _runtime_provenance(pipe, upstream, settings):
    import diffusers
    import transformers

    upstream = Path(upstream)
    weights = upstream / "weight"

    def tokenizer_record(tokenizer):
        return {
            "class": type(tokenizer).__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "model_max_length": tokenizer.model_max_length,
            "vocab_hash": digest(tokenizer.get_vocab()),
        }

    source_root = Path(__file__).resolve().parent
    exhibit_root = source_root.parents[1]
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
        "fan_commit": CONFIG["fan"]["commit"],
        "fan_source": {
            name: file_hash(upstream / "fan" / name)
            for name in ("model.py", "wrapper.py")
        },
        "integration_source": {
            name: file_hash(source_root / name)
            for name in (
                "fan_adapter.py",
                "workers.py",
                "gpu.py",
                "evaluation_worker.py",
                "evaluation.py",
            )
        }
        | {
            "scripts/evaluate_fan.py": file_hash(
                exhibit_root / "scripts/evaluate_fan.py"
            ),
            "scripts/prepare_evaluation.py": file_hash(
                exhibit_root / "scripts/prepare_evaluation.py"
            ),
        },
        "patch_series": {
            "series_hash": file_hash(series) if series.is_file() else None,
            "patches": {name: file_hash(patch_root / name) for name in patch_names},
        },
        "decoders": {name: file_hash(weights / name) for name in ("L.pth", "bigG.pth")},
        "tokenizers": {
            "clip_l": tokenizer_record(pipe.tokenizer),
            "clip_g": tokenizer_record(pipe.tokenizer_2),
        },
        "pipeline": {
            key: copy.deepcopy(settings.get(key))
            for key in ("model", "revision", "checkpoint", "pipeline_config")
        },
        "model_files": runtime_file_provenance(settings),
        "libraries": {
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        },
    }


def _exception_restoration(encoder, prompt, refs, policy):
    before = encode_conditioning(encoder, prompt, None, policy)
    attention = (
        encoder._fan_components["clip_l"].model.text_model.encoder.layers[0].self_attn
    )
    original = attention.forward

    def injected_failure(self, *args, **kwargs):
        raise RuntimeError("diagnostic injected attention failure")

    injected = types.MethodType(injected_failure, attention)
    failure_policy = _policy_variant(policy, profiling={"mode": "all"}, skip_pa=[0])
    attention.forward = injected
    error = None
    restored_by_fan = False
    try:
        try:
            encode_conditioning(
                encoder, prompt, refs, failure_policy, collect_trace=True
            )
        except RuntimeError as exc:
            error = str(exc)
            restored_by_fan = attention.forward is injected
    finally:
        attention.forward = original
    after = encode_conditioning(encoder, prompt, None, policy)
    comparison = compare_conditioning(after, before, NO_REF_LIMITS)
    return {
        "injected_error": error,
        "fan_restored_saved_forward": restored_by_fan,
        "reference_free_after_exception": comparison,
        "passed": (
            error == "diagnostic injected attention failure"
            and restored_by_fan
            and comparison["passed"]
        ),
    }


def _diagnose_prompt(pipe, encoder, direct_encoder, prompt, refs, spec):
    policy = spec["effective_policy"]
    pipeline_plain = _pipeline_conditioning(pipe, prompt)
    fan_plain = encode_conditioning(encoder, prompt, None, policy)
    adapter = encode_conditioning(encoder, prompt, refs, policy, collect_trace=True)
    direct_expected = _expected_direct(encoder, direct_encoder, prompt, refs, policy)
    direct_fan = direct_expected["official"]

    alpha_zero_policy = _policy_variant(policy, alpha=0.0)
    alpha_zero = encode_conditioning(
        encoder, prompt, refs, alpha_zero_policy, collect_trace=True
    )
    all_policy = _policy_variant(policy, profiling={"mode": "all"})
    invariant_base = encode_conditioning(encoder, prompt, refs, all_policy)
    invariants = {}
    for name, variant_refs in _reference_variants(refs).items():
        variant = encode_conditioning(encoder, prompt, variant_refs, all_policy)
        invariants[name] = compare_conditioning(
            variant, invariant_base, PERSONAL_LIMITS
        )

    comparisons = {
        "fan_no_reference_vs_pipeline": compare_conditioning(
            fan_plain, pipeline_plain, NO_REF_LIMITS
        ),
        "adapter_vs_direct_official": compare_conditioning(
            adapter, direct_expected, NO_REF_LIMITS
        ),
        "alpha_zero_vs_no_reference": compare_conditioning(
            alpha_zero, fan_plain, PERSONAL_LIMITS
        ),
        "personalized_vs_no_reference": compare_conditioning(
            adapter, fan_plain, PERSONAL_LIMITS
        ),
        "invariants": invariants,
    }
    comparisons["personalized_vs_no_reference"]["diagnostic_only"] = True
    if policy["pooled_mode"] != "fan":
        comparisons["official_fan_pooled_vs_selected_pooled"] = tensor_metrics(
            direct_fan["pooled"], adapter["pooled"]
        )

    required = [
        ("fan_no_reference_vs_pipeline", comparisons["fan_no_reference_vs_pipeline"]),
        ("adapter_vs_direct_official", comparisons["adapter_vs_direct_official"]),
        ("alpha_zero_vs_no_reference", comparisons["alpha_zero_vs_no_reference"]),
    ]
    required.extend(("invariants." + name, value) for name, value in invariants.items())
    failures = [name for name, value in required if not value["passed"]]
    return {
        "trace": adapter["trace"],
        "alpha_zero_trace": alpha_zero["trace"],
        "personalized_detector": _personalized_detector_report(
            encoder, prompt, refs, policy
        ),
        "comparisons": comparisons,
        "eligibility": {"passed": not failures, "failures": failures},
    }


def _prompts(config):
    values = config.get("prompts")
    if values is None and isinstance(config.get("prompt"), str):
        values = [{"id": "default", "text": config["prompt"]}]
    if not isinstance(values, list) or not values:
        raise ValueError("prompt or prompts is required")
    result = []
    seen = set()
    for index, value in enumerate(values):
        if isinstance(value, str):
            value = {"id": f"prompt-{index}", "text": value}
        if not isinstance(value, dict):
            raise TypeError("each prompt must be a string or object")
        prompt_id, text = value.get("id"), value.get("text")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in seen:
            raise ValueError("prompt id must be present and unique")
        if not isinstance(text, str) or not text:
            raise ValueError("prompt text is required")
        result.append({"id": prompt_id, "text": text})
        seen.add(prompt_id)
    return result


def encoding_cases(config):
    """Expand explicit named prompts and histories without hiding coverage."""
    prompts = _prompts(config)
    histories = config.get("histories")
    if histories is None:
        refs = config.get("refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError("refs or histories must be provided")
        histories = [{"id": "default", "refs": refs}]
    if not isinstance(histories, list) or not histories:
        raise ValueError("histories must be a non-empty list")
    seen = set()
    result = []
    for history in histories:
        if not isinstance(history, dict):
            raise TypeError("each history must be an object")
        history_id, refs = history.get("id"), history.get("refs")
        if (
            not isinstance(history_id, str)
            or not history_id
            or history_id in seen
            or not isinstance(refs, list)
            or not refs
        ):
            raise ValueError("history id and refs must be present and unique")
        seen.add(history_id)
        for prompt in prompts:
            result.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt": prompt["text"],
                    "history_id": history_id,
                    "refs": copy.deepcopy(refs),
                }
            )
    return result


def run_encoding(config):
    import torch
    from fan.wrapper import stable_diffusion_xl

    started = time.monotonic()
    policies = validate_policy_specs(config.get("policies"))
    cases = encoding_cases(config)
    prompts = _prompts(config)
    histories = []
    seen_histories = set()
    for case in cases:
        if case["history_id"] not in seen_histories:
            histories.append(
                {"id": case["history_id"], "refs": copy.deepcopy(case["refs"])}
            )
            seen_histories.add(case["history_id"])
    settings = config.get("settings", CONFIG["generation"])
    upstream = config.get("upstream", str(FAN_UPSTREAM))
    output = Path(config["output"]).resolve()

    pipe = load_pipeline(settings)
    encoder = build_encoder(pipe, upstream)
    direct_encoder = stable_diffusion_xl(
        encoder._fan_components["clip_l"], encoder._fan_components["clip_g"]
    )
    results = {}
    for spec in policies:
        history_results = {}
        for history in histories:
            prompt_results = {}
            for case in cases:
                if case["history_id"] != history["id"]:
                    continue
                emit(
                    "diagnostic",
                    policy_id=spec["policy_id"],
                    policy_hash=spec["policy_hash"],
                    prompt_id=case["prompt_id"],
                    history_id=case["history_id"],
                )
                prompt_results[case["prompt_id"]] = _diagnose_prompt(
                    pipe,
                    encoder,
                    direct_encoder,
                    case["prompt"],
                    case["refs"],
                    spec,
                )
            history_results[history["id"]] = {
                "refs": copy.deepcopy(history["refs"]),
                "prompts": prompt_results,
                "eligibility": {
                    "passed": all(
                        item["eligibility"]["passed"]
                        for item in prompt_results.values()
                    ),
                    "failures": {
                        prompt_id: item["eligibility"]["failures"]
                        for prompt_id, item in prompt_results.items()
                        if item["eligibility"]["failures"]
                    },
                },
            }
        failures = {
            history_id: item["eligibility"]["failures"]
            for history_id, item in history_results.items()
            if item["eligibility"]["failures"]
        }
        result = {
            **spec,
            "histories": history_results,
            "eligibility": {"passed": not failures, "failures": failures},
        }
        if list(history_results) == ["default"]:
            result["prompts"] = history_results["default"]["prompts"]
        results[spec["policy_hash"]] = result

    detector_texts = [
        *[("prompt:" + item["id"], item["text"]) for item in prompts],
        *[
            ("ref:" + history["id"] + ":" + str(index), ref["text"])
            for history in histories
            for index, ref in enumerate(history["refs"])
        ],
    ]
    first_case = cases[0]
    exception = _exception_restoration(
        encoder,
        first_case["prompt"],
        first_case["refs"],
        policies[0]["effective_policy"],
    )
    if not exception["passed"]:
        for policy_result in results.values():
            policy_result["eligibility"]["passed"] = False
            policy_result["eligibility"]["failures"]["_worker"] = [
                "exception_restoration"
            ]
    report = {
        "schema_version": 2,
        "kind": "fan-encoding-diagnostic",
        "diagnostic_identity": copy.deepcopy(config.get("diagnostic_identity")),
        "settings": settings,
        "upstream": upstream,
        "provenance": _runtime_provenance(pipe, upstream, settings),
        "prompts": prompts,
        "histories": histories,
        "policies": results,
        "token_detector": _token_detector_report(pipe, encoder, detector_texts),
        "exception_restoration": exception,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "seconds": round(time.monotonic() - started, 3),
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    write_json(output, report)
    emit(
        "encoding_report",
        path=str(output),
        policy_count=len(policies),
        case_count=len(cases),
        all_eligible=all(item["eligibility"]["passed"] for item in results.values()),
        exception_restored=exception["passed"],
    )
    return report


def validate_image_file(path, width, height):
    """Decode as RGB and apply only the predeclared hard validity rules."""
    from PIL import Image, ImageStat

    path = Path(path)
    if not path.is_file():
        return {"valid": False, "invalid_reason": "missing_image"}
    try:
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
    except (OSError, ValueError) as error:
        return {
            "valid": False,
            "invalid_reason": "decode_error",
            "error": str(error),
        }
    extrema = image.getextrema()
    stats = ImageStat.Stat(image)
    result = {
        "valid": True,
        "mode": "RGB",
        "width": image.width,
        "height": image.height,
        "range": [[int(low), int(high)] for low, high in extrema],
        "variance": [float(value) for value in stats.var],
        "nonfinite_values": 0,
    }
    if image.size != (width, height):
        result.update({"valid": False, "invalid_reason": "wrong_dimensions"})
    elif all(high == 0 for _, high in extrema):
        result.update({"valid": False, "invalid_reason": "all_zero_rgb"})
    return result


def _normalized_features(value):
    import torch

    if hasattr(value, "pooler_output"):
        value = value.pooler_output
    norms = torch.linalg.vector_norm(value.float(), dim=-1, keepdim=True)
    if not bool(torch.isfinite(value).all()) or bool((norms == 0).any()):
        raise ValueError("evaluator returned nonfinite or zero embeddings")
    return value.float() / norms


def embed_evaluation(items, preparation, settings, seconds):
    """Encode validated images and frozen target/reference texts with local CLIP."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    snapshot = preparation["snapshot_path"]
    processor = CLIPProcessor.from_pretrained(
        snapshot, local_files_only=True, use_fast=False
    )
    model = CLIPModel.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    model.eval()
    images = {}
    valid = []
    for item in items:
        validation = validate_image_file(
            item["path"], settings["width"], settings["height"]
        )
        row = {
            **validation,
            "sha256": file_hash(item["path"]) if Path(item["path"]).is_file() else None,
            "seconds": seconds.get(item["id"]),
        }
        images[item["id"]] = row
        if validation["valid"]:
            valid.append(item)
    for offset in range(0, len(valid), 8):
        batch_items = valid[offset : offset + 8]
        opened = []
        try:
            for item in batch_items:
                with Image.open(item["path"]) as source:
                    opened.append(source.convert("RGB"))
            inputs = processor(images=opened, return_tensors="pt")
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.no_grad():
                features = _normalized_features(model.get_image_features(**inputs))
            for item, feature in zip(batch_items, features.cpu().tolist()):
                images[item["id"]]["embedding"] = feature
        finally:
            for image in opened:
                image.close()

    text_values = {}
    for item in items:
        target_id, target_text = item["target_text_id"], item["target_text"]
        if target_id in text_values and text_values[target_id] != target_text:
            raise ValueError("target text id maps to different text")
        text_values[target_id] = target_text
        for ref in item.get("refs", []):
            ref_id = "ref:" + ref["ref_id"]
            if ref_id in text_values and text_values[ref_id] != ref["text"]:
                raise ValueError("reference id maps to different text")
            text_values[ref_id] = ref["text"]
    text_ids = sorted(text_values)
    texts = {}
    for offset in range(0, len(text_ids), 64):
        batch_ids = text_ids[offset : offset + 64]
        inputs = processor(
            text=[text_values[text_id] for text_id in batch_ids],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.no_grad():
            features = _normalized_features(model.get_text_features(**inputs))
        for text_id, feature in zip(batch_ids, features.cpu().tolist()):
            texts[text_id] = feature
    return {"images": images, "texts": texts}


def _conditioning_alignment(item_id, candidate, plain):
    """The `conditioning_sink` contract of `workers.generate`; the id is unused."""
    result = {}
    for name in ("hidden", "pooled"):
        metrics = tensor_metrics(candidate[name], plain[name])
        valid = (
            metrics["finite"]
            and metrics["candidate_zero_vectors"] == 0
            and metrics["base_zero_vectors"] == 0
        )
        result[name] = {
            "valid": valid,
            "cosine": metrics["mean_cosine"] if valid else None,
            "shape": metrics["shape"],
        }
    return result


def run_matrix(request, *, generator=None, embedder=None):
    """Reuse the generation worker, then score every registered artifact locally."""
    from .workers import generate

    generator = generator or generate
    embedder = embedder or embed_evaluation
    image_events = {}
    conditioning = {
        item["id"]: copy.deepcopy(item["prior_event"]["conditioning_target_align"])
        for item in request.get("all_items", [])
        if isinstance(item.get("prior_event"), dict)
        and item["prior_event"].get("conditioning_target_align") is not None
    }
    seconds = {
        item["id"]: item["prior_event"]["seconds"]
        for item in request.get("all_items", [])
        if isinstance(item.get("prior_event"), dict)
        and item["prior_event"].get("seconds") is not None
    }

    def receive(kind, **data):
        if kind == "image":
            image_events[data["id"]] = copy.deepcopy(data)
            seconds[data["id"]] = data.get("seconds")
            if data.get("conditioning_target_align") is not None:
                conditioning[data["id"]] = copy.deepcopy(
                    data["conditioning_target_align"]
                )
        emit(kind, **data)

    generation_request = {
        "stage": "generate",
        "settings": request["settings"],
        "upstream": request.get("upstream", str(FAN_UPSTREAM)),
        "items": request.get("items", []),
        "emit_image_started": True,
    }
    if generation_request["items"]:
        generator(
            generation_request,
            event_sink=receive,
            conditioning_sink=_conditioning_alignment,
        )

    output = Path(request["embeddings_output"]).resolve()
    previous = (
        json.loads(output.read_text()) if output.is_file() else {"conditioning": {}}
    )
    result = embedder(
        request["all_items"],
        request["evaluator_preparation"],
        request["settings"],
        seconds,
    )
    result["conditioning"] = {
        **copy.deepcopy(previous.get("conditioning", {})),
        **conditioning,
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    write_json(output, result)
    emit(
        "evaluation_embeddings",
        path=str(output),
        image_count=len(result["images"]),
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    encoding = subparsers.add_parser("encoding")
    encoding.add_argument("--config", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--request", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "encoding":
        run_encoding(json.loads(Path(args.config).read_text()))
    elif args.command == "matrix":
        run_matrix(json.loads(Path(args.request).read_text()))


if __name__ == "__main__":
    main()
