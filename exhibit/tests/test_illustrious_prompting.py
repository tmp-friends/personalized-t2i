"""The Illustrious prompt envelope and the reviewed generation profile stay fixed."""

from exhibit.config import CONFIG

from exhibit import domain

QUALITY_PREFIX = (
    "masterpiece, best quality, amazing quality, very aesthetic, newest, safe, "
)


def test_compose_prompt_places_quality_first_and_resolution_last():
    prompt = domain.compose_prompt(
        QUALITY_PREFIX + "1girl, solo, holding cat, window, upper body",
        ["soft lighting", "depth of field"],
        {"positive_prompt_tail": "absurdres, highres"},
    )
    assert prompt == (
        QUALITY_PREFIX + "1girl, solo, holding cat, window, upper body, "
        "soft lighting, depth of field, absurdres, highres."
    )


def test_all_exhibit_topics_use_the_illustrious_v2_quality_prefix():
    for topic in CONFIG["topics"]:
        assert topic["basic_prompt_en"].startswith(QUALITY_PREFIX), topic["id"]
        target = domain.target_prompt(topic)
        assert target.startswith(QUALITY_PREFIX)
        assert target.endswith(CONFIG["generation"]["positive_prompt_tail"] + ".")


def test_all_card_prompts_use_the_same_quality_envelope():
    for card in domain.CARDS.values():
        assert card["prompt"].startswith(QUALITY_PREFIX), card["id"]
        assert card["prompt"].endswith(
            CONFIG["generation"]["positive_prompt_tail"] + "."
        )
        # The subject never enters a reference; only the expression phrases do.
        assert QUALITY_PREFIX not in card["ref_en"]
        assert card["ref_en"] == ", ".join(card["aspects"].values())


def test_generation_profile_matches_the_reviewed_illustrious_v2_contract():
    g = CONFIG["generation"]
    assert g["model"] == "OnomaAIResearch/Illustrious-XL-v2.0"
    assert g["revision"] == "69459c1fe6f46db41ab31e6114f05acc0e06bcaa"
    assert g["checkpoint"] == "Illustrious-XL-v2.0.safetensors"
    assert g["vae"] == {
        "model": "madebyollin/sdxl-vae-fp16-fix",
        "revision": "207b116dae70ace3637169f1ddd2434b91b3a8cd",
    }
    assert g["scheduler"] == "DPMSolverMultistepScheduler"
    assert g["scheduler_kwargs"] == {
        "algorithm_type": "sde-dpmsolver++",
        "use_karras_sigmas": True,
    }
    assert (g["width"], g["height"], g["steps"], g["guidance_scale"]) == (
        1024,
        1280,
        30,
        5.0,
    )
    assert g["precision"] == "fp16"
    assert "clip_skip" not in g
    assert g["positive_prompt_tail"] == "absurdres, highres"
    assert "nsfw" in g["negative_prompt"] and "multiple people" in g["negative_prompt"]
