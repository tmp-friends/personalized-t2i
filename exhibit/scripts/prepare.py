#!/usr/bin/env python3
"""Explicit preparation only; serving never downloads or manufactures assets."""

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

from exhibit.catalog import build_catalog
from exhibit.config import (
    ASSETS,
    CONFIG,
    FAN_UPSTREAM,
    OUTPUTS,
    ROOT,
    read_json,
    write_json,
)
from exhibit.domain import CARDS, build_legacy_personalization, target_prompt
from exhibit.gpu import run_stage

# Representative selections for the offline sample experiences (3 cards each).
SAMPLE_SELECTIONS = [
    (
        "s1",
        "あたたかい表現を選んだ人",
        ["girl-warm_soft", "student-warm_soft", "barista-warm_soft"],
    ),
    (
        "s2",
        "涼しくくっきりした表現を選んだ人",
        ["girl-cool_clean", "traveler-cool_clean", "barista-cool_clean"],
    ),
    (
        "s3",
        "劇的な光と鮮やかさを選んだ人",
        ["student-dramatic", "traveler-dramatic", "girl-vivid_lively"],
    ),
]


def stage(request, name, seconds=3600):
    events = []

    def receive(event):
        events.append(event)
        print(event["type"], event.get("id", ""), event.get("seconds", ""), flush=True)

    timing = run_stage(
        {**request, "upstream": str(FAN_UPSTREAM)},
        OUTPUTS / "preparation" / name,
        threading.Event(),
        time.monotonic() + seconds,
        receive,
    )
    write_json(
        OUTPUTS / "preparation" / f"{name}.json", {"events": events, "timing": timing}
    )
    return events


def merge_manifest(new_images, metrics):
    manifest = read_json(ASSETS / "manifest.json", {}) or {}
    images = {
        key: value
        for key, value in (manifest.get("images", {}) or {}).items()
        if key not in new_images
    }
    images.update(new_images)
    write_json(
        ASSETS / "manifest.json",
        {
            "version": 5,
            "generation": CONFIG["generation"],
            "fan": CONFIG["fan"],
            "images": images,
            "metrics": (manifest.get("metrics", []) or []) + metrics,
        },
    )


def image_records(events, extra=None):
    return {
        event["id"]: {
            **event,
            **(extra(event["id"]) if extra else {}),
            "path": str(Path(event["path"]).relative_to(ASSETS)),
            "source": "local-illustrious-xl-v2.0",
            "license": "creativeml-openrail-m",
            "asset_version": 5,
        }
        for event in events
        if event["type"] == "image"
    }


def prepare_cards(catalog="v1"):
    if catalog == "v2":
        definition = read_json(ROOT / "configs/catalog-v2.json")
        cards = {card["id"]: card for card in build_catalog(definition)}
        token_path = OUTPUTS / "preparation" / "catalog-v2" / "token-counts.json"
        check = [str(ROOT.parent / CONFIG["fan"]["python"]), str(ROOT / "scripts/check_card_tokens.py"), "--catalog", "v2", "--output", str(token_path)]
        token_env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        completed = subprocess.run(check, check=False, text=True, capture_output=True, env=token_env)
        if completed.returncode:
            raise RuntimeError("v2 card token validation failed before generation: " + completed.stderr)
        items = [{"id": card["id"], "prompt": card["prompt"], "seed": card["seed"], "path": str(ASSETS / card["path"])} for card in cards.values()]
        events = stage({"stage": "generate", "items": items}, "catalog-v2/card-images")
        images = image_records(events, extra=lambda key: {k: cards[key][k] for k in ("ref_en", "aspects", "subject_id", "profile_id", "axis_levels")})
        write_json(ASSETS / "catalog-v2.json", {"version": 1, "catalog_id": "catalog-v2", "generation": CONFIG["generation"], "token_counts": str(token_path.relative_to(OUTPUTS)), "images": images})
        print("v2 cards prepared; review them in configs/cards-v2-review.json", flush=True)
        return
    items = [
        {
            "id": card["id"],
            "prompt": card["prompt"],
            "seed": card["seed"],
            "path": str(ASSETS / "cards" / f"{card['id']}.png"),
        }
        for card in CARDS.values()
    ]
    events = stage({"stage": "generate", "items": items}, "card-images")
    merge_manifest(
        image_records(
            events,
            extra=lambda key: {
                "ref_en": CARDS[key]["ref_en"],
                "aspects": CARDS[key]["aspects"],
                "subject_id": CARDS[key]["subject_id"],
                "profile_id": CARDS[key]["profile_id"],
            },
        ),
        [e for e in events if e["type"] == "metrics"],
    )
    print("cards prepared; review them in configs/cards-review.json", flush=True)


def prepare_generic():
    prompts = {}
    items = []
    for topic in CONFIG["topics"]:
        prompt = target_prompt(topic)
        prompts[topic["id"]] = {"prompt": prompt, "topic_id": topic["id"]}
        for index, seed in enumerate(CONFIG["seeds"]):
            items.append(
                {
                    "id": f"{topic['id']}-{index}",
                    "prompt": prompt,
                    "seed": seed,
                    "path": str(ASSETS / "generic" / f"{topic['id']}-{index}.png"),
                }
            )
    write_json(ASSETS / "generic-prompts.json", prompts)
    events = stage({"stage": "generate", "items": items}, "generic-images")
    merge_manifest(image_records(events), [e for e in events if e["type"] == "metrics"])


def prepare_samples():
    samples = []
    items = []
    for prefix, label, card_ids in SAMPLE_SELECTIONS:
        selection = [{"card_id": card_id, "aspects_off": []} for card_id in card_ids]
        personalization = build_legacy_personalization(selection)
        for topic in CONFIG["topics"][:2]:
            sample_id = f"{prefix}-{topic['id']}"
            samples.append(
                {
                    "id": sample_id,
                    "topic_id": topic["id"],
                    "label": f"{label} · {topic['label']}",
                    "selection": selection,
                    "personalization": personalization,
                    "mode": "sample",
                    "images": [],
                }
            )
            for index, seed in enumerate(CONFIG["seeds"]):
                items.append(
                    {
                        "id": f"{sample_id}-{index}",
                        "prompt": target_prompt(topic),
                        "seed": seed,
                        "personalization": personalization,
                        "path": str(ASSETS / "samples" / f"{sample_id}-{index}.png"),
                    }
                )
    events = stage({"stage": "generate", "items": items}, "sample-images")
    for sample in samples:
        sample["images"] = [
            {**event, "path": str(Path(event["path"]).relative_to(ASSETS))}
            for event in events
            if event["type"] == "image" and event["id"].startswith(sample["id"] + "-")
        ]
    write_json(ASSETS / "samples.json", samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["cards", "generic", "samples"])
    parser.add_argument("--catalog", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    if args.step == "cards":
        prepare_cards(args.catalog)
    else:
        {"generic": prepare_generic, "samples": prepare_samples}[args.step]()


if __name__ == "__main__":
    main()
