import copy
import hashlib
import json
import subprocess
import sys

import pytest
from exhibit.catalog import CATALOG_ID, build_catalog
from exhibit.config import FAN_POLICIES, ROOT, read_json
from exhibit.domain import build_personalization, digest
from exhibit.fan_adapter import profiling_argument, resolve_policy

# A complete in-memory catalog: these tests are about the encoding identity, not
# about which cards a human has reviewed on disk.
ALL_CARDS = build_catalog(read_json(ROOT / "configs/catalog-v2.json"))
CARDS = {card["id"]: card for card in ALL_CARDS}
IDS = list(CARDS)
CATALOG = {
    "catalog_id": CATALOG_ID,
    "catalog_hash": digest(ALL_CARDS),
    "all_cards": ALL_CARDS,
    "cards": ALL_CARDS,
}


def build(value, **kwargs):
    return build_personalization(value, catalog=CATALOG, **kwargs)


def snapshot(selection=None, *, gains=None, revision=1):
    return {
        "revision": revision,
        "catalog_id": CATALOG["catalog_id"],
        "catalog_hash": CATALOG["catalog_hash"],
        "selection": selection
        or [
            # The first two share a profile, so their phrases merge by weight.
            {"card_id": IDS[0], "strength": 2, "aspects": ["color", "mood"]},
            {"card_id": IDS[16], "strength": 1, "aspects": ["color", "mood"]},
            {"card_id": IDS[2], "strength": 1, "aspects": ["texture"]},
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
    built = build(value, prompt="a target", policy=policy, provenance=provenance())
    reordered = build(
        {**value, "revision": 99, "selection": list(reversed(value["selection"]))},
        prompt="a target",
        policy=policy,
        provenance=provenance(),
    )

    assert built["hash"] == reordered["hash"]
    assert [ref["text"] for ref in built["refs"]] == sorted(
        ref["text"] for ref in built["refs"]
    )
    color = CARDS[IDS[0]]["aspects"]["color"]
    assert next(ref for ref in built["refs"] if ref["text"] == color)["weight"] == 6.0
    assert all(len(ref["ref_id"]) == 64 for ref in built["refs"])
    with pytest.raises(TypeError, match="snapshot"):
        build([], prompt="a target", policy=policy, provenance=provenance())


def test_personalization_hash_tracks_effective_policy_and_provenance_content():
    policy = resolve_policy("legacy_exhibit", FAN_POLICIES)
    base = build(snapshot(), prompt="a target", policy=policy, provenance=provenance())
    changes = [
        ({**policy, "pooled_mode": "fan"}, provenance()),
        ({**policy, "skip": -1}, provenance()),
        ({**policy, "profiling": {"mode": "count", "value": 2}}, provenance()),
        ({**policy, "skip_pa": [0]}, provenance()),
        (policy, provenance(decoder="decoder-b")),
    ]

    for changed_policy, changed_provenance in changes:
        assert (
            build(
                snapshot(),
                prompt="a target",
                policy=changed_policy,
                provenance=changed_provenance,
            )["hash"]
            != base["hash"]
        )


def test_card_description_rejects_non_uniform_gains():
    policy = {
        **resolve_policy("legacy_exhibit", FAN_POLICIES),
        "reference_unit": "card_description",
    }
    with pytest.raises(ValueError, match="aspect_gains"):
        build(snapshot(), prompt="a target", policy=policy, provenance=provenance())


def test_same_effective_policy_has_the_same_identity_regardless_of_registry_name():
    policies = copy.deepcopy(FAN_POLICIES)
    policies["policies"]["legacy_alias"] = copy.deepcopy(
        policies["policies"]["legacy_exhibit"]
    )
    base = build(
        snapshot(),
        prompt="a target",
        policy=resolve_policy("legacy_exhibit", policies),
        provenance=provenance(),
    )
    alias = build(
        snapshot(),
        prompt="a target",
        policy=resolve_policy("legacy_alias", policies),
        provenance=provenance(),
    )
    assert alias["hash"] == base["hash"]


def test_personalization_identity_uses_actual_target_and_generation_inputs():
    policy = resolve_policy("legacy_exhibit", FAN_POLICIES)
    base = build(snapshot(), prompt="a target", policy=policy, provenance=provenance())
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
            build(snapshot(), prompt=prompt, policy=policy, provenance=source)["hash"]
            != base["hash"]
        )


def test_card_description_requires_every_gain_to_be_one():
    policy = {
        **resolve_policy("legacy_exhibit", FAN_POLICIES),
        "reference_unit": "card_description",
    }
    for gain in (0.5, 2):
        with pytest.raises(ValueError, match="aspect_gains"):
            build(
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
    built = build(
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
        build(
            snapshot(
                gains={aspect: 1 for aspect in ("color", "lighting", "texture", "mood")}
            ),
            prompt="a target",
            policy=equivalent,
            provenance=provenance(),
        )["hash"]
        == build(
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
        build(
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
    cards = IDS
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
                "card_id": cards[16],
                "strength": 1,
                "aspects": ["lighting", "color", "mood", "texture"],
            },
        ],
        gains={aspect: 1 for aspect in ("color", "lighting", "texture", "mood")},
    )
    built = build(value, prompt="a target", policy=policy, provenance=provenance())
    expected = ", ".join(
        CARDS[cards[0]]["aspects"][aspect]
        for aspect in ("color", "lighting", "texture", "mood")
    )
    assert built["refs"] == [
        {
            "ref_id": hashlib.sha256(expected.encode()).hexdigest(),
            "text": expected,
            "weight": 3.0,
            "card_ids": sorted([cards[0], cards[16]]),
            "aspects": [],
        }
    ]
    assert CARDS[cards[0]]["subject_id"] not in built["refs"][0]["text"]


def test_personalization_rejects_unreviewed_catalog_cards(monkeypatch):
    from exhibit import catalog as catalog_module

    card = copy.deepcopy(CARDS[IDS[0]])
    monkeypatch.setattr(
        catalog_module,
        "load_catalog",
        lambda *, reviewed_only: {
            "catalog_id": "catalog-v2",
            "catalog_hash": "catalog-hash",
            "all_cards": [card],
            "cards": [] if reviewed_only else [card],
        },
    )
    value = {
        **snapshot(),
        "catalog_id": "catalog-v2",
        "catalog_hash": "catalog-hash",
        "selection": [{"card_id": card["id"], "strength": 1, "aspects": ["color"]}],
    }
    with pytest.raises(ValueError, match="Unknown or duplicate card"):
        build_personalization(
            value,
            prompt="a target",
            policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
            provenance=provenance(),
        )


def test_personalization_rejects_stale_hash_and_cross_catalog_card(monkeypatch):
    from exhibit import catalog as catalog_module

    v2_card = copy.deepcopy(CARDS[IDS[0]])
    v2_card["id"] = "v2-card"
    monkeypatch.setattr(
        catalog_module,
        "load_catalog",
        lambda *, reviewed_only: {
            "catalog_id": "catalog-v2",
            "catalog_hash": "current-hash",
            "all_cards": [v2_card],
            "cards": [v2_card],
        },
    )
    base = {
        **snapshot(),
        "catalog_id": "catalog-v2",
        "catalog_hash": "stale-hash",
        "selection": [{"card_id": v2_card["id"], "strength": 1, "aspects": ["color"]}],
    }
    with pytest.raises(ValueError, match="Stale catalog hash"):
        build_personalization(
            base,
            prompt="a target",
            policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
            provenance=provenance(),
        )

    cross_catalog = {
        **base,
        "catalog_hash": "current-hash",
        "selection": [{"card_id": IDS[0], "strength": 1, "aspects": ["color"]}],
    }
    with pytest.raises(ValueError, match="Unknown or duplicate card"):
        build_personalization(
            cross_catalog,
            prompt="a target",
            policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
            provenance=provenance(),
        )


def test_snapshot_rejects_client_supplied_reference_text():
    value = snapshot()
    value["selection"][0]["ref_en"] = "client supplied"
    with pytest.raises(ValueError, match="Invalid selection entry"):
        build(
            value,
            prompt="a target",
            policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
            provenance=provenance(),
        )


def test_personalization_rejects_snapshot_missing_catalog_id_without_key_error():
    value = snapshot()
    value.pop("catalog_id")
    with pytest.raises(ValueError, match="unknown or missing fields"):
        build(
            value,
            prompt="a target",
            policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
            provenance=provenance(),
        )


def test_a_neutral_embed_gain_keeps_every_policy_hash_registered_before_it():
    """The optional strength knob must not re-key policies measured without it."""
    from exhibit.fan_adapter import freeze_policy, thaw_policy

    legacy = FAN_POLICIES["policies"]["legacy_exhibit"]
    baseline = digest(thaw_policy(freeze_policy(legacy)))
    neutral = digest(thaw_policy(freeze_policy({**legacy, "embed_gain": 1.0})))
    integral = digest(thaw_policy(freeze_policy({**legacy, "embed_gain": 1})))
    raised = digest(thaw_policy(freeze_policy({**legacy, "embed_gain": 1.5})))

    assert baseline == digest(
        thaw_policy(resolve_policy("legacy_exhibit", FAN_POLICIES))
    )
    assert neutral == integral == baseline
    assert raised != baseline
    assert (
        thaw_policy(freeze_policy({**legacy, "embed_gain": 1.5}))["embed_gain"] == 1.5
    )
