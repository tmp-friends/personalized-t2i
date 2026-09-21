import copy

import pytest
from exhibit.catalog import build_catalog, load_catalog
from exhibit.config import FAN_POLICIES, ROOT, read_json
from exhibit.domain import ASPECTS, build_personalization, digest
from exhibit.elicitation import (
    EXPLORE_SLOTS,
    SELECTION,
    PreferenceError,
    RoundError,
    next_round,
    normalize_preferences,
)
from exhibit.fan_adapter import resolve_policy

ROUND_SIZE = SELECTION["round_size"]


def catalog_of(cards, *, reviewed=None, catalog_id="catalog-test"):
    """A loaded-catalog shape; `reviewed` narrows the eligible subset."""
    eligible = cards if reviewed is None else [c for c in cards if c["id"] in reviewed]
    return {
        "catalog_id": catalog_id,
        "catalog_hash": digest([card["id"] for card in cards]),
        "all_cards": copy.deepcopy(cards),
        "cards": copy.deepcopy(eligible),
    }


def card(subject, levels):
    axis_levels = dict(zip(ASPECTS, levels))
    profile = "-".join(f"{axis[0]}{level}" for axis, level in axis_levels.items())
    return {
        "id": f"{subject}-{profile}",
        "subject_id": subject,
        "axis_levels": axis_levels,
    }


def v2_catalog():
    """catalog-v2 has no generated images yet, so build the 64 cards in memory."""
    cards = build_catalog(read_json(ROOT / "configs/catalog-v2.json"))
    return catalog_of(cards, catalog_id="catalog-v2")


def gains(value=1):
    return {aspect: value for aspect in ASPECTS}


def snapshot_of(catalog, selection):
    return {
        "catalog_id": catalog["catalog_id"],
        "catalog_hash": catalog["catalog_hash"],
        "selection": selection,
        "aspect_gains": gains(),
    }


def empty(catalog):
    return snapshot_of(catalog, [])


def liked(catalog, card_id, aspects, strength=1):
    return snapshot_of(
        catalog, [{"card_id": card_id, "strength": strength, "aspects": list(aspects)}]
    )


def first_round(catalog, seed="seed-a"):
    return next_round(
        catalog, empty(catalog), shown_ids=[], round_index=0, session_seed=seed
    )


def test_the_first_round_is_reproducible_from_the_seed_and_history():
    catalog = v2_catalog()
    again = first_round(catalog)
    assert first_round(catalog) == again
    assert len(again["card_ids"]) == ROUND_SIZE
    assert again["round_index"] == 0
    assert again["shortfall_reason"] is None
    assert again["subject_cap_relaxed"] is False
    assert again["session_seed"] == "seed-a"
    assert len(again["round_id"]) == 64


def test_a_different_seed_still_returns_a_valid_but_not_identical_round():
    catalog = v2_catalog()
    known = {c["id"] for c in catalog["cards"]}
    rounds = [first_round(catalog, seed=f"seed-{index}") for index in range(4)]
    for value in rounds:
        assert len(set(value["card_ids"])) == ROUND_SIZE
        assert set(value["card_ids"]) <= known
    assert len({tuple(value["card_ids"]) for value in rounds}) > 1
    assert len({value["round_id"] for value in rounds}) == len(rounds)


def test_the_first_round_covers_every_level_of_every_axis():
    catalog = v2_catalog()
    cards = {c["id"]: c for c in catalog["cards"]}
    for seed in ("seed-a", "seed-b", "seed-c"):
        chosen = [cards[card_id] for card_id in first_round(catalog, seed)["card_ids"]]
        for axis in ASPECTS:
            assert {c["axis_levels"][axis] for c in chosen} == {0, 1, 2, 3}


def test_rounds_never_repeat_a_shown_card_and_keep_subjects_adjacent():
    catalog = v2_catalog()
    cards = {c["id"]: c for c in catalog["cards"]}
    shown = []
    for index in range(SELECTION["max_rounds"]):
        value = next_round(
            catalog,
            empty(catalog),
            shown_ids=list(shown),
            round_index=index,
            session_seed="seed-a",
        )
        assert not set(value["card_ids"]) & set(shown)
        subjects = [cards[card_id]["subject_id"] for card_id in value["card_ids"]]
        assert len(list(dict.fromkeys(subjects))) == len(set(subjects))
        shown.extend(value["card_ids"])
    assert len(set(shown)) == ROUND_SIZE * SELECTION["max_rounds"]


def test_a_fourth_round_is_refused():
    catalog = v2_catalog()
    with pytest.raises(RoundError):
        next_round(
            catalog,
            empty(catalog),
            shown_ids=[],
            round_index=SELECTION["max_rounds"],
            session_seed="seed-a",
        )
    with pytest.raises(RoundError):
        next_round(
            catalog, empty(catalog), shown_ids=[], round_index=-1, session_seed="s"
        )


def test_every_subject_stays_at_three_cards_per_round():
    catalog = v2_catalog()
    cards = {c["id"]: c for c in catalog["cards"]}
    value = first_round(catalog)
    counted = {}
    for card_id in value["card_ids"]:
        subject = cards[card_id]["subject_id"]
        counted[subject] = counted.get(subject, 0) + 1
    assert set(counted.values()) == {3}
    assert value["subject_cap"] == 3


def test_the_subject_cap_relaxes_to_four_only_when_candidates_run_short():
    cards = [card(f"s{s}", (a, a, a, a)) for s in range(3) for a in range(4)]
    catalog = catalog_of(cards)
    value = first_round(catalog)
    counted = {}
    for card_id in value["card_ids"]:
        counted[card_id.split("-")[0]] = counted.get(card_id.split("-")[0], 0) + 1
    assert len(value["card_ids"]) == ROUND_SIZE
    assert set(counted.values()) == {4}
    assert value["subject_cap"] == 4
    assert value["subject_cap_relaxed"] is True
    assert value["shortfall_reason"] is None


def test_a_short_round_reports_a_machine_readable_reason():
    two_subjects = [
        card(f"s{s}", (a, b, 0, 0))
        for s in range(2)
        for a in range(4)
        for b in range(4)
    ]
    capped = first_round(catalog_of(two_subjects))
    assert len(capped["card_ids"]) == 8
    assert capped["shortfall_reason"] == "subject_cap_limit"
    assert capped["subject_cap_relaxed"] is True

    thin = catalog_of([card("s0", (a, a, a, a)) for a in range(4)])
    short = first_round(thin)
    assert len(short["card_ids"]) == 4
    assert short["shortfall_reason"] == "insufficient_unseen_cards"

    exhausted = next_round(
        thin,
        empty(thin),
        shown_ids=[c["id"] for c in thin["cards"]],
        round_index=1,
        session_seed="seed-a",
    )
    assert exhausted["card_ids"] == []
    assert exhausted["shortfall_reason"] == "no_unseen_cards"


def test_zero_selections_make_the_whole_round_exploration():
    catalog = v2_catalog()
    first = first_round(catalog)
    assert {pick["slot"] for pick in first["cards"]} == {"explore"}
    later = next_round(
        catalog,
        empty(catalog),
        shown_ids=first["card_ids"],
        round_index=1,
        session_seed="seed-a",
    )
    assert {pick["slot"] for pick in later["cards"]} == {"explore"}


def test_a_selection_with_no_answered_aspect_is_not_a_known_preference():
    catalog = v2_catalog()
    first = first_round(catalog)
    unanswered = snapshot_of(
        catalog, [{"card_id": first["card_ids"][0], "strength": 2, "aspects": []}]
    )
    value = next_round(
        catalog,
        unanswered,
        shown_ids=first["card_ids"],
        round_index=1,
        session_seed="seed-a",
    )
    exploration = next_round(
        catalog,
        empty(catalog),
        shown_ids=first["card_ids"],
        round_index=1,
        session_seed="seed-a",
    )
    assert {pick["slot"] for pick in value["cards"]} == {"explore"}
    assert value["card_ids"] == exploration["card_ids"]


def test_later_rounds_keep_exactly_four_exploration_slots():
    catalog = v2_catalog()
    first = first_round(catalog)
    value = next_round(
        catalog,
        liked(catalog, first["card_ids"][0], ["color"]),
        shown_ids=first["card_ids"],
        round_index=1,
        session_seed="seed-a",
    )
    slots = [pick["slot"] for pick in value["cards"]]
    assert slots.count("explore") == EXPLORE_SLOTS
    assert slots.count("similar") == ROUND_SIZE - EXPLORE_SLOTS


def test_similarity_reads_liked_aspects_only_and_ignores_the_rest():
    """Liking warm colour must not make a texture level a known preference."""
    picked = card("s0", (0, 0, 0, 0))
    same_texture = card("s1", (0, 1, 0, 1))
    other_texture = card("s1", (0, 1, 1, 1))
    catalog = catalog_of([picked, same_texture, other_texture])
    snapshot = liked(catalog, picked["id"], ["color"])
    value = next_round(
        catalog,
        snapshot,
        shown_ids=[picked["id"]],
        round_index=1,
        session_seed="seed-a",
    )
    colour_only = value["card_ids"]

    texture_too = next_round(
        catalog,
        liked(catalog, picked["id"], ["color", "texture"]),
        shown_ids=[picked["id"]],
        round_index=1,
        session_seed="seed-a",
    )["card_ids"]
    ids = {same_texture["id"], other_texture["id"]}
    assert set(colour_only) == set(texture_too) == ids
    # With texture unanswered both candidates tie and `seed-a` ranks the
    # texture-mismatching card first; liking texture as well reverses that.
    assert colour_only[0] == other_texture["id"]
    assert texture_too[0] == same_texture["id"]


def test_unselected_cards_are_never_negative_evidence():
    warm = [card(f"s{s}", (0, b, b, b)) for s in range(3) for b in range(4)]
    cool = [card("s3", (1, b, b, b)) for b in range(4)]
    catalog = catalog_of(warm + cool)
    # Two warm cards were shown and passed over; that must not demote warm cards.
    shown = [warm[0]["id"], warm[1]["id"], warm[2]["id"]]
    value = next_round(
        catalog,
        liked(catalog, warm[0]["id"], ["color"], strength=2),
        shown_ids=shown,
        round_index=1,
        session_seed="seed-a",
    )
    cards = {c["id"]: c for c in catalog["cards"]}
    similar = [pick["card_id"] for pick in value["cards"] if pick["slot"] == "similar"]
    assert len(similar) == ROUND_SIZE - EXPLORE_SLOTS
    assert all(cards[card_id]["axis_levels"]["color"] == 0 for card_id in similar)


def test_unreviewed_cards_are_never_offered_or_selectable():
    cards = [card("s0", (a, a, a, a)) for a in range(4)]
    catalog = catalog_of(cards, reviewed={cards[0]["id"], cards[1]["id"]})
    value = first_round(catalog)
    assert set(value["card_ids"]) == {cards[0]["id"], cards[1]["id"]}
    with pytest.raises(PreferenceError, match="Unknown or duplicate card"):
        next_round(
            catalog,
            liked(catalog, cards[3]["id"], ["color"]),
            shown_ids=[],
            round_index=1,
            session_seed="seed-a",
        )
    with pytest.raises(PreferenceError, match="Unknown or duplicate card"):
        normalize_preferences(
            {
                "cards": [
                    {"card_id": cards[3]["id"], "strength": 1, "aspects": ["color"]}
                ],
                "aspect_gains": gains(),
            },
            catalog,
            commit=False,
        )


def test_a_stale_catalog_hash_is_refused():
    catalog = v2_catalog()
    stale = {**empty(catalog), "catalog_hash": "stale"}
    with pytest.raises(PreferenceError, match="Stale catalog"):
        next_round(catalog, stale, shown_ids=[], round_index=0, session_seed="seed-a")


def payload(count=3, *, aspects=("color",), strength=1, catalog=None):
    ids = [c["id"] for c in catalog["cards"]]
    return {
        "cards": [
            {"card_id": ids[index], "strength": strength, "aspects": list(aspects)}
            for index in range(count)
        ],
        "aspect_gains": gains(),
    }


def test_a_draft_holds_zero_to_ten_cards_and_a_commit_needs_three():
    catalog = v2_catalog()
    empty_draft = normalize_preferences(
        {"cards": [], "aspect_gains": gains()}, catalog, commit=False
    )
    assert empty_draft["selection"] == []
    assert normalize_preferences(payload(10, catalog=catalog), catalog, commit=True)
    for count in (0, 1, 2):
        with pytest.raises(PreferenceError):
            normalize_preferences(payload(count, catalog=catalog), catalog, commit=True)
    with pytest.raises(PreferenceError):
        normalize_preferences(payload(11, catalog=catalog), catalog, commit=False)


def test_an_unanswered_card_is_a_draft_only_state_and_never_becomes_all_aspects():
    catalog = v2_catalog()
    value = payload(3, aspects=(), catalog=catalog)
    draft = normalize_preferences(value, catalog, commit=False)
    assert [entry["aspects"] for entry in draft["selection"]] == [[], [], []]
    with pytest.raises(PreferenceError, match="Invalid aspects"):
        normalize_preferences(value, catalog, commit=True)


def test_the_normalized_form_is_canonical_and_order_independent():
    catalog = v2_catalog()
    value = payload(3, aspects=("mood", "color"), catalog=catalog)
    reversed_value = {
        "cards": list(reversed(value["cards"])),
        "aspect_gains": gains(),
    }
    normalized = normalize_preferences(value, catalog, commit=True)
    assert normalized == normalize_preferences(reversed_value, catalog, commit=True)
    assert normalized["catalog_id"] == catalog["catalog_id"]
    assert normalized["catalog_hash"] == catalog["catalog_hash"]
    assert [entry["card_id"] for entry in normalized["selection"]] == sorted(
        entry["card_id"] for entry in normalized["selection"]
    )
    assert normalized["selection"][0]["aspects"] == ["color", "mood"]
    assert normalized["aspect_gains"] == {aspect: 1.0 for aspect in ASPECTS}


def test_the_server_never_trusts_a_client_catalog_hash_or_unknown_fields():
    catalog = v2_catalog()
    value = payload(3, catalog=catalog)
    assert (
        normalize_preferences(
            value, {**catalog, "catalog_hash": "server-owned"}, commit=True
        )["catalog_hash"]
        == "server-owned"
    )
    for broken in (
        {**value, "catalog_hash": "client"},
        {**value, "catalog_id": "catalog-v9"},
        {"cards": value["cards"]},
        [],
        None,
    ):
        with pytest.raises(PreferenceError):
            normalize_preferences(broken, catalog, commit=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["cards"].append(copy.deepcopy(value["cards"][0])),
        lambda value: value["cards"][0].update(strength=3),
        lambda value: value["cards"][0].update(strength=True),
        lambda value: value["cards"][0].update(strength=1.0),
        lambda value: value["cards"][0].update(aspects=["color", "color"]),
        lambda value: value["cards"][0].update(aspects=["hairstyle"]),
        lambda value: value["cards"][0].update(aspects="color"),
        lambda value: value["cards"][0].update(card_id="unknown-card"),
        lambda value: value["cards"][0].update(card_id=["unhashable"]),
        lambda value: value["cards"][0].update(aspects=[["color"]]),
        lambda value: value["cards"][0].update(aspects=[{"axis": "color"}]),
        lambda value: value["cards"][0].update(note="extra"),
        lambda value: value["cards"].__setitem__(0, "card-id"),
        lambda value: value.update(cards={}),
        lambda value: value["aspect_gains"].update(color=3),
        lambda value: value["aspect_gains"].update(color=True),
        lambda value: value["aspect_gains"].update(color=float("nan")),
        lambda value: value["aspect_gains"].pop("mood"),
        lambda value: value["aspect_gains"].update(extra=1),
        lambda value: value.update(aspect_gains=[]),
    ],
)
def test_broken_preference_payloads_are_refused(mutate):
    catalog = v2_catalog()
    value = payload(3, catalog=catalog)
    mutate(value)
    with pytest.raises(PreferenceError):
        normalize_preferences(value, catalog, commit=True)


@pytest.mark.parametrize("gain", [0.5, 1, 2])
def test_every_documented_gain_step_survives_normalization(gain):
    catalog = v2_catalog()
    value = payload(3, catalog=catalog)
    value["aspect_gains"]["color"] = gain
    assert normalize_preferences(value, catalog, commit=True)["aspect_gains"][
        "color"
    ] == float(gain)


def test_a_committed_snapshot_is_accepted_by_build_personalization():
    catalog = load_catalog("catalog-v1", reviewed_only=True)
    assert catalog["cards"]
    ids = [c["id"] for c in catalog["cards"]]
    value = {
        "cards": [
            {"card_id": ids[0], "strength": 2, "aspects": ["color", "mood"]},
            {"card_id": ids[1], "strength": 1, "aspects": ["texture"]},
            {"card_id": ids[2], "strength": 1, "aspects": ["color"]},
        ],
        "aspect_gains": {"color": 2, "lighting": 1, "texture": 0.5, "mood": 1},
    }
    normalized = normalize_preferences(value, catalog, commit=True)
    built = build_personalization(
        {"revision": 2, **normalized},
        prompt="a target",
        policy=resolve_policy("legacy_exhibit", FAN_POLICIES),
        provenance={
            "fan_pin": "9d0b768",
            "adapter_hash": "adapter-a",
            "decoder_hash": "decoder-a",
            "tokenizer_hash": "tokenizer-a",
            "generation": {"steps": 30},
            "seeds": [230923],
        },
    )
    assert built["refs"]
    assert built["hash"] == built["personalization_hash"]


def test_a_committed_snapshot_feeds_the_next_round_of_the_same_catalog():
    catalog = load_catalog("catalog-v1", reviewed_only=True)
    ids = [c["id"] for c in catalog["cards"]]
    normalized = normalize_preferences(
        {
            "cards": [{"card_id": ids[0], "strength": 1, "aspects": ["color"]}],
            "aspect_gains": gains(),
        },
        catalog,
        commit=False,
    )
    value = next_round(
        catalog,
        normalized,
        shown_ids=ids[:4],
        round_index=1,
        session_seed="seed-a",
        selection={"round_size": 6},
    )
    assert len(value["card_ids"]) == 6
    assert not set(value["card_ids"]) & set(ids[:4])


def test_unhashable_selection_values_are_refused_by_the_round_policy():
    catalog = v2_catalog()
    entry = {"card_id": ["unhashable"], "strength": 1, "aspects": ["color"]}
    with pytest.raises(PreferenceError):
        next_round(
            catalog,
            {"selection": [entry]},
            shown_ids=[],
            round_index=1,
            session_seed="seed-a",
        )
