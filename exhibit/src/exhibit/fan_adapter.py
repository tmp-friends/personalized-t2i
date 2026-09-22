"""Validated, import-light FAN encoding policies."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from types import MappingProxyType

POLICY_KEYS = frozenset(
    {
        "alpha",
        "skip",
        "skip_pa",
        "use_attn_mask",
        "pooled_mode",
        "profiling",
        "reference_unit",
    }
)
# Optional strength settings. They are dropped from the effective policy at
# their neutral value so every policy registered before them keeps its hash.
OPTIONAL_POLICY_KEYS = frozenset({"embed_gain"})
EMBED_GAIN_NEUTRAL = 1.0
# ``fan_eos`` personalizes the SDXL pooled channel like ``fan`` but pools at the
# prompt's EOS position instead of FAN's ClassTokenDecoder, which mislocates the
# class token on long tag prompts.
POOLED_MODES = frozenset({"plain", "fan", "fan_eos"})
REFERENCE_UNITS = frozenset({"aspect_phrase", "card_description"})


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def thaw_policy(policy):
    """Return a JSON-shaped copy suitable for an isolated worker payload."""
    if isinstance(policy, Mapping):
        return {key: thaw_policy(value) for key, value in policy.items()}
    if isinstance(policy, tuple):
        return [thaw_policy(value) for value in policy]
    return copy.deepcopy(policy)


def profiling_argument(policy):
    """Translate an explicit profiling mode into FAN's type-sensitive input."""
    profiling = policy.get("profiling") if isinstance(policy, Mapping) else None
    if not isinstance(profiling, Mapping):
        raise TypeError("profiling must be an object")
    mode = profiling.get("mode")
    if mode == "all":
        if set(profiling) != {"mode"}:
            raise ValueError("profiling all accepts no value")
        return 0
    if set(profiling) != {"mode", "value"}:
        raise ValueError("profiling has unknown fields")
    value = profiling.get("value")
    if mode == "count":
        if type(value) is not int or value < 1:
            raise ValueError("profiling count must be an integer of at least 1")
        return value
    if mode == "ratio":
        if not _finite_number(value) or not 0 < value <= 1:
            raise ValueError("profiling ratio must be finite and in (0, 1]")
        return float(value)
    raise ValueError("Unknown profiling mode")


def _validate_policy(policy):
    if (
        not isinstance(policy, Mapping)
        or not POLICY_KEYS <= set(policy)
        or set(policy) - POLICY_KEYS - OPTIONAL_POLICY_KEYS
    ):
        raise ValueError("Policy must contain exactly the supported settings")
    if "embed_gain" in policy and (
        not _finite_number(policy["embed_gain"]) or not 0 <= policy["embed_gain"] <= 4
    ):
        raise ValueError("embed_gain must be finite and in [0, 4]")
    if not _finite_number(policy["alpha"]) or not 0 <= policy["alpha"] <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if type(policy["skip"]) is not int:
        raise ValueError("skip must be an integer")
    if not isinstance(policy["skip_pa"], (list, tuple)) or any(
        type(index) is not int or index < 0 for index in policy["skip_pa"]
    ):
        raise ValueError("skip_pa must contain non-negative integers")
    if type(policy["use_attn_mask"]) is not bool:
        raise ValueError("use_attn_mask must be a boolean")
    if policy["pooled_mode"] not in POOLED_MODES:
        raise ValueError("Unknown pooled_mode")
    if policy["reference_unit"] not in REFERENCE_UNITS:
        raise ValueError("Unknown reference_unit")
    profiling_argument(policy)


def freeze_policy(policy):
    """Validate and recursively freeze a caller-supplied effective policy."""
    value = thaw_policy(policy)
    _validate_policy(value)
    value["alpha"] = float(value["alpha"])
    if value["profiling"]["mode"] == "ratio":
        value["profiling"]["value"] = float(value["profiling"]["value"])
    value["skip_pa"] = sorted(set(value["skip_pa"]))
    if "embed_gain" in value:
        value["embed_gain"] = float(value["embed_gain"])
        if value["embed_gain"] == EMBED_GAIN_NEUTRAL:
            del value["embed_gain"]
    return _freeze(value)


def embed_gain(policy):
    """The hidden-state gain of a policy; absent means the neutral 1.0."""
    return float(policy.get("embed_gain", EMBED_GAIN_NEUTRAL))


def resolve_policy(policy_id, policies):
    """Resolve and freeze one registered policy, independent of its display name."""
    if not isinstance(policy_id, str) or not isinstance(policies, Mapping):
        raise TypeError("Invalid policy")
    registered = policies.get("policies")
    if not isinstance(registered, Mapping) or policy_id not in registered:
        raise ValueError("Unknown policy")
    policy = thaw_policy(registered[policy_id])
    return freeze_policy(policy)


TRACE_PHASES = ("clip_l_hidden", "clip_g_hidden", "clip_g_pool")


def _trace_indices(value):
    if value is None:
        return None
    for method in ("detach", "cpu"):
        operation = getattr(value, method, None)
        if operation is not None:
            value = operation()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("sample_reference indices must be a sequence or None")
    return [int(index) for index in value]


def _trace_call(phase, refs, indices, called):
    source = copy.deepcopy(refs)
    return {
        "phase": phase,
        "input_refs": source,
        "selected_indices": indices,
        "selected_refs": (
            source
            if indices is None
            else [copy.deepcopy(source[index]) for index in indices]
        ),
        "sample_reference_called": called,
    }


def _validated_refs(refs):
    values = []
    for ref in refs:
        if not isinstance(ref, Mapping):
            raise TypeError("each reference must be an object")
        text = ref.get("text")
        weight = ref.get("weight")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("reference text must be non-empty")
        if not _finite_number(weight) or weight <= 0:
            raise ValueError("reference weight must be finite and positive")
        values.append(copy.deepcopy(dict(ref)))
    return values


def _share_big_g_pool_call(calls):
    shared = copy.deepcopy(calls[1])
    shared["phase"] = "clip_g_pool"
    shared["sample_reference_called"] = False
    shared["shared_selection_with"] = "clip_g_hidden"
    calls.append(shared)


def _encode_once(encoder, prompt, refs, effective, collect_trace):
    import torch

    ref_values = copy.deepcopy(refs) if refs else None
    texts = [ref["text"] for ref in ref_values] if ref_values else None
    weights = [float(ref["weight"]) for ref in ref_values] if ref_values else None
    sample_size = profiling_argument(effective)
    calls = []
    fan_model = getattr(encoder, "_fan_model_module", None)
    original = None

    if collect_trace and ref_values and sample_size:
        if fan_model is None:
            import importlib

            fan_model = importlib.import_module("fan.model")
        original = fan_model.sample_reference

        # Return the official tensor rather than rebuilding it. Recording is kept
        # separate so reference selection remains entirely upstream-owned.
        def recording_sample_reference(*args, **kwargs):
            selected = original(*args, **kwargs)
            indices = _trace_indices(selected)
            position = len(calls)
            phase = (
                TRACE_PHASES[position]
                if position < len(TRACE_PHASES)
                else f"unexpected_{position}"
            )
            calls.append(_trace_call(phase, ref_values, indices, True))
            return selected

        fan_model.sample_reference = recording_sample_reference

    try:
        with torch.no_grad():
            hidden, pooled = encoder(
                prompt,
                texts,
                weight=weights,
                alpha=effective["alpha"] if ref_values else None,
                skip=effective["skip"],
                sample_size=sample_size,
                skip_pa=list(effective["skip_pa"]),
                use_attn_mask=effective["use_attn_mask"],
            )
    finally:
        if original is not None:
            fan_model.sample_reference = original

    if collect_trace and ref_values:
        if not sample_size:
            calls = [
                _trace_call(phase, ref_values, None, False) for phase in TRACE_PHASES
            ]
            if effective["skip"] == -1:
                calls.pop()
                _share_big_g_pool_call(calls)
        elif effective["skip"] == -1 and len(calls) == 2:
            _share_big_g_pool_call(calls)
        elif len(calls) != len(TRACE_PHASES):
            raise RuntimeError(
                "Expected official sample_reference calls for every SDXL phase, "
                f"got {len(calls)}"
            )
    return hidden.to(torch.float16), pooled.to(torch.float16), calls


def _encode_eos_pooled(encoder, prompt, refs, effective):
    """bigG personalized like the ``fan`` pooled path, pooled at the EOS token.

    Mirrors ``fan.wrapper.stable_diffusion_xl``: one more bigG pass with
    ``skip=-1`` and no normalization, then final layer norm and projection.
    Only the pooling position differs from upstream, so the reference selection
    and the personalized attention stay upstream-owned.
    """
    import torch

    components = getattr(encoder, "_fan_components", None)
    if not isinstance(components, Mapping) or "clip_g" not in components:
        raise RuntimeError("fan_eos pooled mode needs the pipeline's bigG FAN encoder")
    big_g = components["clip_g"]
    texts = [ref["text"] for ref in refs]
    weights = [float(ref["weight"]) for ref in refs]
    with torch.no_grad():
        hidden = big_g(
            prompt,
            texts,
            weight=weights,
            alpha=effective["alpha"],
            pooling=False,
            sample_size=profiling_argument(effective),
            skip=-1,
            skip_pa=list(effective["skip_pa"]),
            use_attn_mask=effective["use_attn_mask"],
            normalize=False,
        )
        pooled = big_g.pool_text_hidden_state(
            big_g.normalize_text_hidden_state(hidden), prompt
        )
        pooled = big_g.projection_text_hidden_state(pooled)
    return pooled.to(torch.float16)


def _apply_embed_gain(hidden, plain_hidden, gain):
    """``plain + gain * (personalized - plain)`` in the encoder's own dtype."""
    import torch

    personalized = hidden.float()
    base = plain_hidden.float()
    return (base + gain * (personalized - base)).to(torch.float16)


def encode_conditioning(encoder, prompt, refs, policy, *, collect_trace=False):
    """Encode one SDXL prompt using a validated FAN policy.

    Reference selection stays inside the pinned FAN implementation. The adapter
    only records the official selections and chooses whether the generator sees
    FAN's pooled embedding, an EOS-pooled personalized embedding, or a
    reference-free pooled embedding. An optional ``embed_gain`` scales the
    personalized hidden-state difference from the reference-free encoding.
    """
    if not isinstance(prompt, str) or (refs and not prompt):
        raise ValueError("prompt is required")
    if refs is not None and not isinstance(refs, (list, tuple)):
        raise TypeError("refs must be a list or None")
    effective = freeze_policy(policy)
    effective_value = thaw_policy(effective)
    ref_values = _validated_refs(refs) if refs else None

    hidden, fan_pooled, calls = _encode_once(
        encoder, prompt, ref_values, effective, collect_trace
    )
    pooled = fan_pooled
    pooled_source = "fan" if ref_values else "plain"
    gain = embed_gain(effective)
    plain_hidden = None
    if ref_values and (effective["pooled_mode"] == "plain" or gain != 1.0):
        plain_hidden, plain_pooled, _ = _encode_once(
            encoder, prompt, None, effective, False
        )
        if effective["pooled_mode"] == "plain":
            pooled = plain_pooled
            pooled_source = "plain"
    if ref_values and effective["pooled_mode"] == "fan_eos":
        pooled = _encode_eos_pooled(encoder, prompt, ref_values, effective)
        pooled_source = "fan_eos"
    if ref_values and gain != 1.0:
        hidden = _apply_embed_gain(hidden, plain_hidden, gain)

    trace = {
        "calls": calls,
        "pooled_source": pooled_source,
        "profiling": copy.deepcopy(effective_value["profiling"]),
    }
    if ref_values and gain != 1.0:
        trace["embed_gain"] = gain
    return {
        "hidden": hidden,
        "pooled": pooled,
        "trace": trace,
        "effective_policy": effective_value,
    }
