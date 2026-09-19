import pytest
from exhibit.domain import build_persona, cache_key, effective_context, validate_prompt


def persona():
    choices = [
        {"pair_id": "p1", "chosen_id": "a"},
        {"pair_id": "p2", "chosen_id": None},
    ]
    evidence = [
        {
            "id": "e1",
            "pair_id": "p1",
            "chosen_id": "a",
            "dimensions": ["color"],
            "values": {"color": "warm"},
            "text": "Warm tones relative to the other image.",
            "reviewed": True,
        },
        {
            "id": "e2",
            "pair_id": "p1",
            "chosen_id": "a",
            "dimensions": ["lighting", "mood"],
            "values": {"lighting": "soft", "mood": "calm"},
            "text": "Soft light with a quiet atmosphere.",
            "reviewed": True,
        },
        {
            "id": "e3",
            "pair_id": "p2",
            "chosen_id": "c",
            "dimensions": ["color"],
            "values": {"color": "cool"},
            "text": "Cool tones.",
            "reviewed": True,
        },
    ]
    return build_persona(choices, evidence)


def test_skips_do_not_create_evidence():
    p = persona()
    assert p["axes"]["color"]["value"] == "warm"
    assert p["axes"]["color"]["count"] == 1
    assert {e["id"] for e in p["evidence"]} == {"e1", "e2"}


def test_off_removes_whole_multi_axis_fragment_from_both_routes():
    c = effective_context(persona(), {"lighting": None})
    assert [e["id"] for e in c["evidence"]] == ["e1"]
    assert "Soft light" not in c["text"]
    assert "lighting" not in c["preferences"]
    assert c["hash"] != effective_context(persona(), {})["hash"]


def test_correction_removes_conflicting_raw_evidence():
    c = effective_context(persona(), {"color": "cool"})
    assert "Warm tones" not in c["text"]
    assert c["preferences"]["color"] == "cool"
    assert c["overrides"] == {"color": "cool"}
    assert "User explicitly requests" in c["text"]


def test_all_off_and_invalid_edits():
    assert not effective_context(persona(), {k: None for k in persona()["axes"]})[
        "preferences"
    ]
    with pytest.raises(ValueError):
        effective_context(persona(), {"age": "old"})
    with pytest.raises(ValueError):
        effective_context(persona(), {"color": "<script>"})


def test_cache_changes_on_every_generation_input():
    a = cache_key("cat", {"revision": "r1", "steps": 20}, 1)
    assert a != cache_key("cat", {"revision": "r1", "steps": 21}, 1)
    assert a != cache_key("cat", {"revision": "r2", "steps": 20}, 1)
    assert a != cache_key("cat", {"revision": "r1", "steps": 20}, 2)
    assert a != cache_key("cat", {"revision": "r1", "steps": 20}, 1, {"hash": "new"})


class Tokenizer:
    model_max_length = 12

    def __call__(self, text, **kw):
        return {"input_ids": text.split() + [0, 1]}


def test_rewrite_preserves_basic_prompt_and_both_tokenizer_limits():
    topic = {"basic_prompt_en": "One red cat by a window."}
    assert validate_prompt(
        "One red cat by a window. Soft light.", topic, [Tokenizer(), Tokenizer()]
    )
    assert not validate_prompt(
        "One blue cat by a window.", topic, [Tokenizer(), Tokenizer()]
    )
    assert not validate_prompt(
        "One red cat by a window. " + "bright " * 20, topic, [Tokenizer(), Tokenizer()]
    )


def test_effective_context_does_not_forward_unreviewed_raw_vlm_output():
    import json

    p = persona()
    p["evidence"][0]["raw"] = "UNREVIEWED_RAW_OUTPUT: dramatic light and a loud mood"
    context = effective_context(p, {"lighting": None})
    assert "UNREVIEWED_RAW_OUTPUT" not in json.dumps(context)
    assert context["evidence"][0]["text"] == "Warm tones relative to the other image."
