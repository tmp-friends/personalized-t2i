# Task 1 report — FAN policy, snapshot, and identity

## Implementation

- Added `exhibit/configs/fan-policies.json` with immutable-at-resolution `legacy_exhibit` and `official_encoder` policies.
- Added import-light `exhibit.fan_adapter`: `resolve_policy`, `profiling_argument`, and `thaw_policy`. It imports no torch, rejects invalid/non-finite policy inputs, preserves FAN's `all`/`ratio`/`count` type distinction, and canonicalizes duplicate `skip_pa` indices after validation.
- Replaced the public domain builder with `build_personalization(snapshot, *, prompt, policy, provenance)`. It rejects list inputs, resolves card IDs only from the server catalog, normalizes selection/reference order, aggregates aspect weights, requires provenance for FAN/adapter/decoder/tokenizer/generation/seeds, and hashes effective policy and content rather than `policy_id` or revision.
- Added `legacy_snapshot_from_selection` and `build_legacy_personalization`. Service, preflight, preparation, fixture, and legacy tests explicitly use the latter, preserving the old v1 payload and cache hash while later tasks migrate the service contract.

## RED

```text
PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_fan_policy.py -q
ImportError: cannot import name 'FAN_POLICIES' from 'exhibit.config'
```

After review regressions were added, the focused command failed as expected for: non-1 uniform `card_description` gains, mappingproxy return values that cannot be deep-copied, equivalent numeric values yielding different hashes, and negative revisions being accepted.

## GREEN / verification

```text
PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests/test_fan_policy.py exhibit/tests/test_domain.py -q
34 passed in 0.04s

PYTHONPATH=exhibit/src fan-repro/.venv/bin/python -c 'import exhibit.fan_adapter; print("fan_adapter Python 3.10 import ok")'
fan_adapter Python 3.10 import ok

PYTHONPATH=exhibit/src exhibit/.venv/bin/python -m pytest exhibit/tests -q
93 passed, 2 third-party deprecation warnings in 1.07s

git diff --check
# no output
```

## Modified files

- `exhibit/configs/fan-policies.json`
- `exhibit/src/exhibit/config.py`
- `exhibit/src/exhibit/fan_adapter.py`
- `exhibit/src/exhibit/domain.py`
- Legacy call-site updates: `service.py`, `preflight.py`, `scripts/prepare.py`, fixture/tests, and stale script comments.
- `exhibit/tests/test_fan_policy.py`

## Interfaces and remaining concerns

- `resolve_policy(policy_id, policies)` returns a recursively immutable mapping. `thaw_policy(policy)` provides a detached JSON-shaped worker/session payload.
- `profiling_argument(policy)` returns `0` for all, `float` for ratio, and `int` for count. It rejects bool, NaN, invalid bounds, and unknown nested fields.
- `build_personalization(snapshot, *, prompt, policy, provenance)` returns `refs`, mutable detached `effective_policy`, `policy_hash`, `provenance`, `sample_size`, `hash`, and `personalization_hash`.
- The current live service still intentionally uses `build_legacy_personalization`; a later task must construct server-validated snapshots and use the new builder/provenance contract end-to-end.
- No GPU inference, weights download, or upstream edits were performed.
