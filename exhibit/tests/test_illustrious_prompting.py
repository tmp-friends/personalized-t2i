from exhibit import domain
from exhibit.config import CONFIG


def test_compose_prompt_places_quality_first_and_resolution_last():
    topic = {
        "basic_prompt_en": (
            "masterpiece, best quality, amazing quality, very aesthetic, newest, "
            "safe, 1girl, solo, brown hair, green eyes, sweater, holding cat, "
            "window, upper body, detailed eyes"
        )
    }

    prompt = domain.compose_prompt(
        topic,
        ["soft lighting", "depth of field"],
        {"positive_prompt_tail": "absurdres, highres"},
    )

    assert prompt == (
        "masterpiece, best quality, amazing quality, very aesthetic, newest, safe, "
        "1girl, solo, brown hair, green eyes, sweater, holding cat, window, upper body, "
        "detailed eyes, soft lighting, depth of field, absurdres, highres."
    )


def test_all_exhibit_topics_use_the_illustrious_v2_quality_prefix():
    expected_prefix = (
        "masterpiece, best quality, amazing quality, very aesthetic, newest, safe, "
    )

    for topic in CONFIG["topics"]:
        prompt = domain.compose_prompt(topic, [], CONFIG["generation"])
        assert prompt.startswith(expected_prefix), topic["id"]
        assert prompt.endswith("absurdres, highres."), topic["id"]


def test_all_preference_pair_prompts_use_the_same_quality_envelope():
    expected_prefix = (
        "masterpiece, best quality, amazing quality, very aesthetic, newest, safe, "
    )

    for pair in CONFIG["pairs"]:
        assert len(pair["prompts"]) == 2, pair["id"]
        for prompt in pair["prompts"]:
            assert prompt.startswith(expected_prefix), pair["id"]
            assert prompt.endswith("absurdres, highres."), pair["id"]


def test_generation_profile_matches_the_reviewed_illustrious_v2_contract():
    generation = CONFIG["generation"]

    assert generation["scheduler"] == "EulerAncestralDiscreteScheduler"
    assert generation["steps"] == 28
    assert generation["guidance_scale"] == 5.0
    assert generation["negative_prompt"] == (
        "lowres, worst quality, bad quality, bad anatomy, bad hands, jpeg artifacts, "
        "poorly drawn, blurry, watermark, signature, artistic error, artistic failure, "
        "bad proportions, bad perspective, multiple views, text, artist sign, "
        "weibo username, extra digits, fewer digits, multiple people, 2girls, 2boys, "
        "nsfw, nude"
    )
    # Diffusers already uses hidden_states[-2]; WebUI-style 2 would select [-4].
    assert "clip_skip" not in generation
