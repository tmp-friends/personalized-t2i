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


@pytest.fixture
def recorded_gain(monkeypatch):
    """The fake tensors carry no arithmetic, so record the gain application."""
    from exhibit import fan_adapter

    applied = []

    def apply(hidden, plain_hidden, gain):
        applied.append((hidden, plain_hidden, gain))
        return Tensor("hidden-gained")

    monkeypatch.setattr(fan_adapter, "_apply_embed_gain", apply)
    return applied


@pytest.fixture
def recorded_eos(monkeypatch):
    from exhibit import fan_adapter

    pooled_calls = []

    def encode(encoder, prompt, refs, effective):
        pooled_calls.append({"prompt": prompt, "refs": refs, "policy": effective})
        return Tensor("pooled-eos")

    monkeypatch.setattr(fan_adapter, "_encode_eos_pooled", encode)
    return pooled_calls


def test_a_neutral_embed_gain_leaves_the_encoding_and_the_trace_untouched(
    fake_torch, recorded_gain
):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    result = encode_conditioning(
        encoder, "target", refs(), {**policy(), "embed_gain": 1.0}, collect_trace=True
    )

    assert "embed_gain" not in result["effective_policy"]
    assert "embed_gain" not in result["trace"]
    assert result["hidden"].name == "hidden-personal"
    assert recorded_gain == []
    assert len(calls) == 1


def test_embed_gain_adds_the_reference_free_encoding_even_for_fan_pooled(
    fake_torch, recorded_gain
):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    result = encode_conditioning(
        encoder, "target", refs(), {**policy(), "embed_gain": 2.0}, collect_trace=True
    )

    assert result["effective_policy"]["embed_gain"] == 2.0
    assert [call["refs"] for call in calls] == [
        ["warm palette", "cool palette"],
        None,
    ]
    assert calls[1]["alpha"] is None
    ((hidden, plain_hidden, gain),) = recorded_gain
    assert hidden.name == "hidden-personal"
    assert plain_hidden.name == "hidden-plain"
    assert gain == 2.0
    assert result["hidden"].name == "hidden-gained"
    # The pooled channel still comes from FAN; only the hidden states are scaled.
    assert result["pooled"].name == "pooled-personal"
    assert result["trace"]["embed_gain"] == 2.0
    assert result["trace"]["pooled_source"] == "fan"


def test_embed_gain_applies_on_top_of_the_plain_pooled_encoding(
    fake_torch, recorded_gain
):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    result = encode_conditioning(
        encoder,
        "target",
        refs(),
        {**policy(pooled_mode="plain"), "embed_gain": 1.5},
        collect_trace=False,
    )

    # One personalized and one reference-free call, shared by both features.
    assert len(calls) == 2
    assert recorded_gain[0][2] == 1.5
    assert result["hidden"].name == "hidden-gained"
    assert result["pooled"].name == "pooled-plain"


def test_fan_eos_pooling_replaces_the_pooled_channel_and_labels_the_trace(
    fake_torch, recorded_eos
):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    result = encode_conditioning(
        encoder, "target", refs(), policy(pooled_mode="fan_eos"), collect_trace=True
    )

    assert result["pooled"].name == "pooled-eos"
    assert result["hidden"].name == "hidden-personal"
    assert result["trace"]["pooled_source"] == "fan_eos"
    assert len(calls) == 1
    assert recorded_eos[0]["prompt"] == "target"
    assert recorded_eos[0]["refs"] == refs()
    assert recorded_eos[0]["policy"]["alpha"] == 0.4


def test_reference_free_encodings_never_scale_or_repool(
    fake_torch, recorded_gain, recorded_eos
):
    for mode, gain in (("fan_eos", 1.0), ("fan", 2.0), ("plain", 2.5)):
        encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])
        result = encode_conditioning(
            encoder, "negative", None, {**policy(pooled_mode=mode), "embed_gain": gain}
        )
        assert result["hidden"].name == "hidden-plain"
        assert result["pooled"].name == "pooled-plain"
        assert result["trace"]["pooled_source"] == "plain"
        assert "embed_gain" not in result["trace"]
        assert len(calls) == 1
    assert recorded_gain == []
    assert recorded_eos == []


def test_an_empty_prompt_is_only_allowed_without_references(fake_torch):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    # The official sampler comparison drops the negative prompt entirely.
    result = encode_conditioning(encoder, "", None, policy())
    assert result["hidden"].name == "hidden-plain"
    assert calls[0]["prompt"] == ""

    with pytest.raises(ValueError, match="prompt"):
        encode_conditioning(encoder, "", refs(), policy())
    assert len(calls) == 1


@pytest.mark.parametrize("gain", [5, -0.1, True, float("nan"), "1.5"])
def test_embed_gain_is_rejected_outside_the_declared_range(fake_torch, gain):
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])

    with pytest.raises(ValueError, match="embed_gain"):
        encode_conditioning(encoder, "target", refs(), {**policy(), "embed_gain": gain})
    assert calls == []


def test_unknown_policy_keys_stay_rejected_next_to_the_optional_gain(fake_torch):
    encoder, _, _ = make_encoder(lambda *args, **kwargs: [[0]])

    with pytest.raises(ValueError, match="supported settings"):
        encode_conditioning(encoder, "target", refs(), {**policy(), "embed_boost": 2.0})


def test_the_supported_pooled_modes_are_plain_fan_and_fan_eos():
    from exhibit.fan_adapter import POOLED_MODES, embed_gain, freeze_policy

    assert POOLED_MODES == {"plain", "fan", "fan_eos"}
    assert freeze_policy(policy(pooled_mode="fan_eos"))["pooled_mode"] == "fan_eos"
    assert embed_gain(freeze_policy(policy())) == 1.0
    assert embed_gain(freeze_policy({**policy(), "embed_gain": 0})) == 0.0
    with pytest.raises(ValueError, match="pooled_mode"):
        freeze_policy(policy(pooled_mode="eos"))


def test_the_attention_mask_reaches_only_the_encodes_that_have_references(fake_torch):
    """``use_attn_mask`` excludes reference pads, so a plain encode never gets it."""
    encoder, calls, _ = make_encoder(lambda *args, **kwargs: [[0]])
    masked = {**policy(pooled_mode="plain"), "use_attn_mask": True}

    encode_conditioning(encoder, "target", refs(), masked)
    encode_conditioning(encoder, "negative", None, masked)

    assert [(call["refs"] is not None, call["use_attn_mask"]) for call in calls] == [
        (True, True),
        (False, False),
        (False, False),
    ]

