"""FAN adapter behavior without importing the GPU stack."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from exhibit.fan_adapter import encode_conditioning


class Tensor:
    def __init__(self, name, dtype="fp32"):
        self.name = name
        self.dtype = dtype

    def to(self, dtype):
        return Tensor(self.name, dtype)


def policy(*, pooled_mode="fan", profiling=None, alpha=0.4):
    return {
        "alpha": alpha,
        "skip": -2,
        "skip_pa": [0],
        "use_attn_mask": False,
        "pooled_mode": pooled_mode,
        "profiling": profiling or {"mode": "count", "value": 1},
        "reference_unit": "aspect_phrase",
    }


@pytest.fixture
def fake_torch(monkeypatch):
    @contextmanager
    def no_grad():
        yield

    module = SimpleNamespace(float16="fp16", no_grad=no_grad)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def make_encoder(sample_reference, *, fail=False):
    fan_model = SimpleNamespace(sample_reference=sample_reference)
    calls = []

    def encoder(prompt, refs=None, **kwargs):
        calls.append({"prompt": prompt, "refs": refs, **kwargs})
        if refs and kwargs["sample_size"]:
            call_count = 2 if kwargs["skip"] == -1 else 3
            for _ in range(call_count):
                fan_model.sample_reference("target", "context", kwargs["weight"])
        if fail:
            raise RuntimeError("encoder failed")
        suffix = "personal" if refs else "plain"
        return Tensor(f"hidden-{suffix}"), Tensor(f"pooled-{suffix}")

    encoder._fan_model_module = fan_model
    return encoder, calls, fan_model


def refs():
    return [
        {"ref_id": "warm", "text": "warm palette", "weight": 2.0},
        {"ref_id": "cool", "text": "cool palette", "weight": 1.0},
    ]


def test_fan_pooled_is_returned_and_alpha_zero_still_uses_references(fake_torch):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[1]])

    result = encode_conditioning(
        encoder, "target", refs(), policy(alpha=0.0), collect_trace=False
    )

    assert result["hidden"].name == "hidden-personal"
    assert result["pooled"].name == "pooled-personal"
    assert result["hidden"].dtype == result["pooled"].dtype == "fp16"
    assert calls == [
        {
            "prompt": "target",
            "refs": ["warm palette", "cool palette"],
            "weight": [2.0, 1.0],
            "alpha": 0.0,
            "skip": -2,
            "sample_size": 1,
            "skip_pa": [0],
            "use_attn_mask": False,
        }
    ]


def test_plain_pooled_comes_from_one_reference_free_positive_encoding(fake_torch):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    result = encode_conditioning(
        encoder, "target", refs(), policy(pooled_mode="plain"), collect_trace=False
    )

    assert result["hidden"].name == "hidden-personal"
    assert result["pooled"].name == "pooled-plain"
    assert [call["refs"] for call in calls] == [
        ["warm palette", "cool palette"],
        None,
    ]
    assert calls[1]["alpha"] is None


def test_reference_free_encoding_is_single_call_for_either_pooled_policy(fake_torch):
    for mode in ("fan", "plain"):
        encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])
        result = encode_conditioning(
            encoder, "negative", None, policy(pooled_mode=mode)
        )
        assert result["hidden"].name == "hidden-plain"
        assert result["pooled"].name == "pooled-plain"
        assert len(calls) == 1
        assert calls[0]["refs"] is None
        assert calls[0]["weight"] is None
        assert calls[0]["alpha"] is None


def test_trace_labels_official_calls_and_preserves_reference_order(fake_torch):
    encoder, _, _ = make_encoder(lambda *args, **kwargs: [[1]])

    result = encode_conditioning(
        encoder, "target", refs(), policy(), collect_trace=True
    )

    assert result["trace"]["pooled_source"] == "fan"
    assert [call["phase"] for call in result["trace"]["calls"]] == [
        "clip_l_hidden",
        "clip_g_hidden",
        "clip_g_pool",
    ]
    for call in result["trace"]["calls"]:
        assert call["input_refs"] == refs()
        assert call["selected_indices"] == [1]
        assert call["selected_refs"] == [refs()[1]]
        assert call["sample_reference_called"] is True


def test_trace_records_all_references_when_official_sampling_is_bypassed(fake_torch):
    encoder, _, _ = make_encoder(lambda *args, **kwargs: pytest.fail("not called"))

    result = encode_conditioning(
        encoder,
        "target",
        refs(),
        policy(profiling={"mode": "all"}),
        collect_trace=True,
    )

    assert len(result["trace"]["calls"]) == 3
    for call in result["trace"]["calls"]:
        assert call["selected_indices"] is None
        assert call["selected_refs"] == refs()
        assert call["sample_reference_called"] is False


def test_trace_hook_is_restored_when_the_encoder_raises(fake_torch):
    original = lambda *args, **kwargs: [[0]]
    encoder, _, fan_model = make_encoder(original, fail=True)

    with pytest.raises(RuntimeError, match="encoder failed"):
        encode_conditioning(encoder, "target", refs(), policy(), collect_trace=True)

    assert fan_model.sample_reference is original


def test_effective_policy_is_a_detached_json_value(fake_torch):
    source = policy()
    encoder, _, _ = make_encoder(lambda *args, **kwargs: [[0]])
    result = encode_conditioning(encoder, "target", refs(), source)

    source["skip_pa"].append(7)
    assert result["effective_policy"]["skip_pa"] == [0]


def test_skip_minus_one_shares_the_big_g_selection_between_hidden_and_pool(fake_torch):
    encoder, _, _ = make_encoder(lambda *args, **kwargs: [[1]])
    skip_minus_one = {**policy(), "skip": -1}

    result = encode_conditioning(
        encoder, "target", refs(), skip_minus_one, collect_trace=True
    )

    calls = result["trace"]["calls"]
    assert [call["phase"] for call in calls] == [
        "clip_l_hidden",
        "clip_g_hidden",
        "clip_g_pool",
    ]
    assert calls[1]["selected_indices"] == calls[2]["selected_indices"] == [1]
    assert calls[2]["sample_reference_called"] is False
    assert calls[2]["shared_selection_with"] == "clip_g_hidden"


@pytest.mark.parametrize(
    "bad_refs",
    [
        [{"text": "", "weight": 1.0}],
        [{"text": "warm", "weight": 0}],
        [{"text": "warm", "weight": -1}],
        [{"text": "warm", "weight": float("nan")}],
        [{"text": "warm", "weight": True}],
        ["warm"],
    ],
)
def test_direct_reference_fixtures_require_text_and_finite_positive_weight(
    fake_torch, bad_refs
):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    with pytest.raises((TypeError, ValueError)):
        encode_conditioning(encoder, "target", bad_refs, policy())

    assert calls == []


def test_diagnostic_policy_specs_are_validated_and_keyed_by_effective_hash():
    from exhibit.domain import digest
    from exhibit.evaluation_worker import validate_policy_specs

    official = policy(pooled_mode="fan")
    legacy = policy(pooled_mode="plain", profiling={"mode": "all"}, alpha=0.5)
    specs = validate_policy_specs(
        [
            {"policy_id": "screen-a", "effective_policy": official},
            {"policy_id": "screen-b", "effective_policy": legacy},
        ]
    )

    assert [item["policy_id"] for item in specs] == ["screen-a", "screen-b"]
    assert specs[0]["policy_hash"] == digest(specs[0]["effective_policy"])
    official["skip_pa"].append(7)
    assert specs[0]["effective_policy"]["skip_pa"] == [0]


@pytest.mark.parametrize(
    "specs",
    [
        [],
        [{"policy_id": "missing"}],
        [{"policy_id": "bad", "effective_policy": {}}],
        [
            {"policy_id": "same", "effective_policy": policy()},
            {"policy_id": "same", "effective_policy": policy(pooled_mode="plain")},
        ],
    ],
)
def test_diagnostic_policy_specs_reject_missing_invalid_or_duplicate_entries(specs):
    from exhibit.evaluation_worker import validate_policy_specs

    with pytest.raises((TypeError, ValueError)):
        validate_policy_specs(specs)
