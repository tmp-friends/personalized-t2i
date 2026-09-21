import copy

import pytest
from exhibit.config import CONFIG
from exhibit.domain import (
    CARDS,
    build_cards,
    build_personalization,
    fan_settings,
    normalize_selection,
    personalization_hash,
    ref_text,
    run_cache_key,
)

IDS = list(CARDS)


def selection(count=3, aspects_off=()):
    return [{"card_id": IDS[i], "aspects_off": list(aspects_off)} for i in range(count)]


def test_sixteen_cards_never_put_the_subject_into_the_reference():
    cards = build_cards()
    assert len(cards) == 16
    for card in cards:
        assert card["ref_en"] == ", ".join(card["aspects"].values())
        subject = next(
            s for s in CONFIG["card_subjects"] if s["id"] == card["subject_id"]
        )
        assert subject["basic_prompt_en"] not in card["ref_en"]
        assert card["prompt"].startswith(subject["basic_prompt_en"])
        assert card["ref_en"] in card["prompt"]


def test_aspects_off_removes_exactly_those_phrases():
    card = CARDS[IDS[0]]
    full = ref_text(card)
    reduced = ref_text(card, ["mood", "texture"])
    assert card["aspects"]["mood"] in full and card["aspects"]["mood"] not in reduced
    assert card["aspects"]["texture"] not in reduced
    assert card["aspects"]["color"] in reduced
    with pytest.raises(ValueError):
        ref_text(card, ["color", "lighting", "texture", "mood"])
    with pytest.raises(ValueError):
        ref_text(card, ["hairstyle"])


def hash_with(refs, alpha, **overrides):
    arguments = {
        "commit": CONFIG["fan"]["commit"],
        "generation": CONFIG["generation"],
        "seeds": CONFIG["seeds"],
        "fan": fan_settings(),
        **overrides,
    }
    return personalization_hash(refs, alpha, **arguments)


def test_references_are_deduplicated_aspect_phrases_with_merged_weights():
    base = build_personalization(selection())
    calm = CARDS[IDS[0]]["aspects"]["mood"]
    assert calm == CARDS[IDS[1]]["aspects"]["mood"]
    phrases = [ref["text"] for ref in base["refs"]]
    assert len(phrases) == len(set(phrases)) == 11
    # First-seen order, subject phrases never included.
    assert phrases[:4] == list(CARDS[IDS[0]]["aspects"].values())
    assert phrases[3] == calm
    merged = next(ref for ref in base["refs"] if ref["text"] == calm)
    assert merged["weight"] == 2.0
    assert merged["aspect"] == "mood"
    assert merged["card_ids"] == [IDS[0], IDS[1]]
    assert all("card_id" not in ref for ref in base["refs"])


def test_personalization_hash_changes_with_order_alpha_and_settings():
    base = build_personalization(selection())
    swapped = build_personalization([selection()[1], selection()[0], selection()[2]])
    assert base["hash"] != swapped["hash"]
    stronger = build_personalization(selection(), {**CONFIG, "alpha": 0.6})
    assert stronger["hash"] != base["hash"]
    assert build_personalization(selection())["hash"] == base["hash"]
    assert base["alpha"] == CONFIG["alpha"] == 0.5

    refs, alpha = base["refs"], base["alpha"]
    assert hash_with(refs, alpha) == base["hash"]
    settings = copy.deepcopy(CONFIG["generation"])
    settings["steps"] += 1
    assert hash_with(refs, alpha, generation=settings) != base["hash"]
    assert hash_with(refs, alpha, commit="other-commit") != base["hash"]
    assert hash_with(refs, alpha, seeds=[1, 2, 3, 4]) != base["hash"]
    for knob, value in (("skip_pa", [0, 1]), ("use_attn_mask", True), ("skip", -1)):
        assert (
            hash_with(refs, alpha, fan={**fan_settings(), knob: value})
            != (base["hash"])
        )


def test_the_measured_fan_settings_are_the_ones_that_are_hashed():
    assert fan_settings() == {
        "skip": -2,
        "sample_size": 0,
        "skip_pa": [0, 1, 2, 3, 4, 5, 6, 7],
        "use_attn_mask": False,
    }


def test_aspects_off_removes_only_that_phrase():
    base = build_personalization(selection())
    reduced = build_personalization(selection(aspects_off=["mood"]))
    assert base["hash"] != reduced["hash"]
    moods = {CARDS[i]["aspects"]["mood"] for i in IDS[:3]}
    assert not moods & {ref["text"] for ref in reduced["refs"]}
    assert len(reduced["refs"]) == len(base["refs"]) - 2  # calm merged, serious alone


def test_selection_enforces_min_max_and_known_cards():
    assert len(normalize_selection(selection(5))) == 5
    with pytest.raises(ValueError):
        normalize_selection(selection(2))
    with pytest.raises(ValueError):
        normalize_selection(selection(3) + [{"card_id": IDS[3]}] * 3)
    with pytest.raises(ValueError):
        normalize_selection([{"card_id": "unknown-card"}] + selection(2))
    with pytest.raises(ValueError):
        normalize_selection([{"card_id": IDS[0]}] * 3)
    with pytest.raises(ValueError):
        normalize_selection(
            selection(2) + [{"card_id": IDS[2], "aspects_off": ["subject"]}]
        )


def test_cache_key_covers_topic_and_personalization():
    base = build_personalization(selection())
    other = build_personalization(selection(aspects_off=["mood"]))
    assert run_cache_key("cat", base) == run_cache_key("cat", base)
    assert run_cache_key("cat", base) != run_cache_key("tokyo", base)
    assert run_cache_key("cat", base) != run_cache_key("cat", other)
