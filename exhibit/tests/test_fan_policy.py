import copy
import hashlib
import json
import subprocess
import sys

import pytest
from exhibit.config import FAN_POLICIES
from exhibit.catalog import load_catalog
from exhibit.domain import CARDS, build_personalization, legacy_snapshot_from_selection
from exhibit.fan_adapter import profiling_argument, resolve_policy


def snapshot(selection=None, *, gains=None, revision=1):
    cards = list(CARDS)
    return {
        "revision": revision,
        "catalog_id": "catalog-v1",
        "catalog_hash": load_catalog("catalog-v1", reviewed_only=False)["catalog_hash"],
        "selection": selection
        or [
            {"card_id": cards[0], "strength": 2, "aspects": ["color", "mood"]},
            {"card_id": cards[4], "strength": 1, "aspects": ["color", "mood"]},
            {"card_id": cards[2], "strength": 1, "aspects": ["texture"]},
        ],
        "aspect_gains": gains or {"color": 2, "lighting": 1, "texture": 0.5, "mood": 1},
    }


def provenance(*, decoder="decoder-a"):
    return {
        "fan_pin": "9d0b768",
        "adapter_hash": "adapter-a",
        "decoder_hash": decoder,
        "tokenizer_hash": "tokenizer-a",
        "generation": {"steps": 30},
        "seeds": [230923],
    }


def test_resolve_policy_is_a_detached_effective_copy_and_rejects_unknown_modes():
    policies = copy.deepcopy(FAN_POLICIES)
    effective = resolve_policy("legacy_exhibit", policies)
    with pytest.raises(TypeError):
        effective["skip_pa"][0] = 99

    assert policies["policies"]["legacy_exhibit"]["skip_pa"] == list(range(8))
    assert resolve_policy("official_encoder", policies)["profiling"] == {
        "mode": "ratio",
        "value": 0.1,
    }
    with pytest.raises(ValueError, match="pooled_mode"):
        resolve_policy(
            "broken", {"policies": {"broken": {**effective, "pooled_mode": "unknown"}}}
        )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ({"profiling": {"mode": "all"}}, 0),
        ({"profiling": {"mode": "ratio", "value": 0.1}}, 0.1),
        ({"profiling": {"mode": "count", "value": 4}}, 4),
        ({"profiling": {"mode": "count", "value": 1}}, 1),
        ({"profiling": {"mode": "ratio", "value": 1}}, 1.0),
    ],
)
def test_profiling_argument_preserves_the_policy_mode_and_numeric_type(
    policy, expected
):
    actual = profiling_argument(policy)
    assert actual == expected
    assert type(actual) is type(expected)


@pytest.mark.parametrize(
    "policy",
    [
        {"profiling": {"mode": "count", "value": True}},
        {"profiling": {"mode": "count", "value": 1.0}},
        {"profiling": {"mode": "count", "value": 0}},
        {"profiling": {"mode": "count", "value": -1}},
        {"profiling": {"mode": "ratio", "value": True}},
        {"profiling": {"mode": "ratio", "value": float("nan")}},
        {"profiling": {"mode": "ratio", "value": -0.1}},
        {"profiling": {"mode": "ratio", "value": 1.1}},
        {"profiling": {"mode": "other", "value": 1}},
    ],
)
def test_profiling_argument_rejects_ambiguous_or_invalid_values(policy):
    with pytest.raises(ValueError):
        profiling_argument(policy)


def test_build_personalization_requires_snapshot_and_normalizes_content_identity():
    policy = resolve_policy("legacy_exhibit", FAN_POLICIES)
    value = snapshot()
    built = build_personalization(
        value, prompt="a target", policy=policy, provenance=provenance()
    )
    reordered = build_personalization(
        {**value, "revision": 99, "selection": list(reversed(value["selection"]))},
        prompt="a target",
        policy=policy,
        provenance=provenance(),
    )

    assert built["hash"] == reordered["hash"]
    assert [ref["text"] for ref in built["refs"]] == sorted(
        ref["text"] for ref in built["refs"]
    )
    color = CARDS[next(iter(CARDS))]["aspects"]["color"]
    assert next(ref for ref in built["refs"] if ref["text"] == color)["weight"] == 6.0
    assert all(len(ref["ref_id"]) == 64 for ref in built["refs"])
    with pytest.raises(TypeError, match="snapshot"):
        build_personalization(
            [], prompt="a target", policy=policy, provenance=provenance()
        )


def test_personalization_hash_tracks_effective_policy_and_provenance_content():
    policy = resolve_policy("legacy_exhibit", FAN_POLICIES)
    base = build_personalization(
        snapshot(), prompt="a target", policy=policy, provenance=provenance()
    )
    changes = [
        ({**policy, "pooled_mode": "fan"}, provenance()),
        ({**policy, "skip": -1}, provenance()),
        ({**policy, "profiling": {"mode": "count", "value": 2}}, provenance()),
        ({**policy, "skip_pa": [0]}, provenance()),
        (policy, provenance(decoder="decoder-b")),
    ]

    for changed_policy, changed_provenance in changes:
        assert (
            build_personalization(
                snapshot(),
                prompt="a target",
                policy=changed_policy,
                provenance=changed_provenance,
            )["hash"]
            != base["hash"]
        )


def test_card_description_rejects_non_uniform_gains_and_legacy_conversion_is_explicit():
    policy = {
        **resolve_policy("legacy_exhibit", FAN_POLICIES),
        "reference_unit": "card_description",
    }
    with pytest.raises(ValueError, match="aspect_gains"):
        build_personalization(
            snapshot(), prompt="a target", policy=policy, provenance=provenance()
        )

    legacy = legacy_snapshot_from_selection(
        [{"card_id": card_id, "aspects_off": []} for card_id in list(CARDS)[:3]]
    )
    assert legacy["selection"][0] == {
        "card_id": next(iter(CARDS)),
        "strength": 1,
        "aspects": ["color", "lighting", "texture", "mood"],
    }


def test_same_effective_policy_has_the_same_identity_regardless_of_registry_name():
    policies = copy.deepcopy(FAN_POLICIES)
    policies["policies"]["legacy_alias"] = copy.deepcopy(
        policies["policies"]["legacy_exhibit"]
    )
    base = build_personalization(
        snapshot(),
        prompt="a target",
        policy=resolve_policy("legacy_exhibit", policies),
        provenance=provenance(),
    )
    alias = build_personalization(
        snapshot(),
        prompt="a target",
        policy=resolve_policy("legacy_alias", policies),
        provenance=provenance(),
    )
    assert alias["hash"] == base["hash"]


def test_personalization_identity_uses_actual_target_and_generation_inputs():
    policy = resolve_policy("legacy_exhibit", FAN_POLICIES)
    base = build_personalization(
        snapshot(), prompt="a target", policy=policy, provenance=provenance()
    )
    variants = [
        ("another target", provenance()),
        (
            "a target",
            {**provenance(), "generation": {"steps": 30, "negative": "different"}},
        ),
        ("a target", {**provenance(), "seeds": [230924]}),
    ]
    for prompt, source in variants:
        assert (
            build_personalization(
                snapshot(), prompt=prompt, policy=policy, provenance=source
            )["hash"]
            != base["hash"]
        )


def test_card_description_requires_every_gain_to_be_one():
    policy = {
        **resolve_policy("legacy_exhibit", FAN_POLICIES),
        "reference_unit": "card_description",
    }
    for gain in (0.5, 2):
        with pytest.raises(ValueError, match="aspect_gains"):
            build_personalization(
                snapshot(
                    gains={
                        aspect: gain
                        for aspect in ("color", "lighting", "texture", "mood")
                    }
                ),
                prompt="a target",
                policy=policy,
                provenance=provenance(),
            )


def test_built_effective_policy_is_json_shaped_and_detached():
    built = build_personalization(
        snapshot(),
        prompt="a target",
        policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
        provenance=provenance(),
    )
    rebuilt = json.loads(json.dumps(built))
    copied = copy.deepcopy(rebuilt["effective_policy"])
    copied["skip_pa"].append(99)
    assert built["effective_policy"]["skip_pa"] == list(range(8))


def test_policy_canonicalizes_equivalent_numbers_and_rejects_bool_skip_pa():
    base_policy = resolve_policy("legacy_exhibit", FAN_POLICIES)
    equivalent = {**base_policy, "alpha": 0, "profiling": {"mode": "ratio", "value": 1}}
    floating = {
        **base_policy,
        "alpha": 0.0,
        "profiling": {"mode": "ratio", "value": 1.0},
    }
    assert (
        build_personalization(
            snapshot(
                gains={aspect: 1 for aspect in ("color", "lighting", "texture", "mood")}
            ),
            prompt="a target",
            policy=equivalent,
            provenance=provenance(),
        )["hash"]
        == build_personalization(
            snapshot(
                gains={
                    aspect: 1.0 for aspect in ("color", "lighting", "texture", "mood")
                }
            ),
            prompt="a target",
            policy=floating,
            provenance=provenance(),
        )["hash"]
    )
    with pytest.raises(ValueError):
        resolve_policy(
            "bad",
            {"policies": {"bad": {**base_policy, "skip_pa": [0, False]}}},
        )


def test_snapshot_rejects_negative_revision():
    with pytest.raises(ValueError, match="snapshot identity"):
        build_personalization(
            snapshot(revision=-1),
            prompt="a target",
            policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
            provenance=provenance(),
        )


def test_fan_adapter_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import exhibit.fan_adapter; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_card_description_aggregates_cards_in_canonical_aspect_order():
    cards = list(CARDS)
    policy = {
        **resolve_policy("legacy_exhibit", FAN_POLICIES),
        "reference_unit": "card_description",
    }
    value = snapshot(
        selection=[
            {
                "card_id": cards[0],
                "strength": 2,
                "aspects": ["mood", "texture", "color", "lighting"],
            },
            {
                "card_id": cards[4],
                "strength": 1,
                "aspects": ["lighting", "color", "mood", "texture"],
            },
        ],
        gains={aspect: 1 for aspect in ("color", "lighting", "texture", "mood")},
    )
    built = build_personalization(
        value, prompt="a target", policy=policy, provenance=provenance()
    )
    expected = ", ".join(
        CARDS[cards[0]]["aspects"][aspect]
        for aspect in ("color", "lighting", "texture", "mood")
    )
    assert built["refs"] == [
        {
            "ref_id": hashlib.sha256(expected.encode()).hexdigest(),
            "text": expected,
            "weight": 3.0,
            "card_ids": sorted([cards[0], cards[4]]),
            "aspects": [],
        }
    ]
    assert CARDS[cards[0]]["subject_id"] not in built["refs"][0]["text"]
