"""The Illustrious prompt envelope and the reviewed generation profile stay fixed."""

from exhibit.config import CONFIG

from exhibit import domain

QUALITY_PREFIX = (
    "masterpiece, best quality, amazing quality, very aesthetic, newest, safe, "
)
# catalog-v2 dropped `amazing quality` to buy tokens for the four aspect phrases.
V2_QUALITY_PREFIX = "masterpiece, best quality, very aesthetic, newest, safe, "


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


def test_v2_catalog_prompts_keep_all_four_reference_phrases_in_the_quality_envelope():
    from exhibit.catalog import build_catalog
    from exhibit.config import ROOT, read_json

    cards = build_catalog(read_json(ROOT / "configs/catalog-v2.json"))
    assert len(cards) == 64
    assert "amazing quality" not in V2_QUALITY_PREFIX
    for card in cards:
        assert card["prompt"].startswith(V2_QUALITY_PREFIX), card["id"]
        assert "amazing quality" not in card["prompt"], card["id"]
        assert card["prompt"].endswith(
            CONFIG["generation"]["positive_prompt_tail"] + "."
        )
        assert card["ref_en"] == ", ".join(card["aspects"].values())
        assert all(phrase in card["prompt"] for phrase in card["aspects"].values())
