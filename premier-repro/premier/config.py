"""Tiny YAML config loader with `_base_` inheritance and `a.b=c` CLI overrides."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access (nested)."""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    @staticmethod
    def wrap(d: Any) -> Any:
        if isinstance(d, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in d.items()})
        if isinstance(d, list):
            return [Cfg.wrap(x) for x in d]
        return d

    def to_dict(self) -> dict:
        def unwrap(x):
            if isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            if isinstance(x, list):
                return [unwrap(v) for v in x]
            return x
        return unwrap(self)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _load_yaml_with_base(path: Path) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("_base_", None)
    if base:
        base_path = (path.parent / base).resolve()
        d = deep_update(_load_yaml_with_base(base_path), d)
    return d


def apply_overrides(d: dict, overrides: list[str] | None) -> dict:
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        cur = d
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return d


def load_config(path: str | Path, overrides: list[str] | None = None) -> Cfg:
    d = _load_yaml_with_base(Path(path).resolve())
    d = apply_overrides(d, overrides)
    return Cfg.wrap(d)


def config_summary(cfg: Cfg) -> str:
    return json.dumps(cfg.to_dict(), indent=1, ensure_ascii=False)
