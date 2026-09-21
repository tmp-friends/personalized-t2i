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
POOLED_MODES = frozenset({"plain", "fan"})
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
    if not isinstance(policy, Mapping) or set(policy) != POLICY_KEYS:
        raise ValueError("Policy must contain exactly the supported settings")
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
    return _freeze(value)


def resolve_policy(policy_id, policies):
    """Resolve and freeze one registered policy, independent of its display name."""
    if not isinstance(policy_id, str) or not isinstance(policies, Mapping):
        raise TypeError("Invalid policy")
    registered = policies.get("policies")
    if not isinstance(registered, Mapping) or policy_id not in registered:
        raise ValueError("Unknown policy")
    policy = thaw_policy(registered[policy_id])
    return freeze_policy(policy)
