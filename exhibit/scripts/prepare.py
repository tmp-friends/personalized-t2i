#!/usr/bin/env python3
"""Explicit preparation only; serving never downloads or manufactures assets."""

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

from exhibit.catalog import build_catalog, card_settings, validate_token_report
from exhibit.config import (
    ASSETS,
    CONFIG,
    FAN_UPSTREAM,
    OUTPUTS,
    ROOT,
    read_json,
    write_json,
)
from exhibit.domain import ASPECTS, build_personalization, target_prompt
from exhibit.elicitation import normalize_preferences
from exhibit.gpu import run_stage
from exhibit.service import active_catalog, default_policy, web_provenance

# Representative selections for the offline sample experiences (design §6.2):
# three selections of three reviewed cards, each one profile seen on three
# subjects, so the shared thing is the expression and never the subject.
SAMPLE_SELECTIONS = [
    (
        "s1",
        "暖かな色と強い日差しを選んだ人",
        ["girl-c0-l3-t1-m1", "student-c0-l3-t1-m1", "traveler-c0-l3-t1-m1"],
        1,
        ["color", "lighting"],
    ),
    (
        "s2",
        "涼しい色と厚い筆致を選んだ人",
        ["student-c1-l1-t3-m1", "traveler-c1-l1-t3-m1", "barista-c1-l1-t3-m1"],
        1,
        ["color", "texture"],
    ),
    (
        "s3",
        "逆光と線のない平塗りを選んだ人",
        ["student-c2-l2-t2-m1", "traveler-c2-l2-t2-m1", "barista-c2-l2-t2-m1"],
        2,
        ["lighting", "texture"],
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
    """The topic manifest: the reference-free images every comparison starts from."""
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


def prepare_cards(only=None):
    """Generate the card images; `only` re-rolls named cards and keeps the rest."""
    definition = read_json(ROOT / "configs/catalog-v2.json")
    cards = {card["id"]: card for card in build_catalog(definition)}
    # The cards keep their own protective negative prompt; demo.json is untouched.
    settings = card_settings(definition)
    token_path = OUTPUTS / "preparation" / "catalog-v2" / "token-counts.json"
    check = [
        str(ROOT.parent / CONFIG["fan"]["python"]),
        str(ROOT / "scripts/check_card_tokens.py"),
        "--output",
        str(token_path),
    ]
    token_env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        check, check=False, text=True, capture_output=True, env=token_env
    )
    if completed.returncode:
        raise RuntimeError(
            "card token validation failed before generation: " + completed.stderr
        )
    try:
        token_report = validate_token_report(
            read_json(token_path), list(cards.values()), generation=settings
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("card token validation report is invalid") from exc
    selected = list(cards)
    if only is not None:
        unknown = [card_id for card_id in only if card_id not in cards]
        if unknown:
            raise RuntimeError("Unknown card: " + ", ".join(unknown))
        selected = list(dict.fromkeys(only))
        if not selected:
            raise RuntimeError("--only needs at least one card id")
    items = [
        {
            "id": card_id,
            "prompt": cards[card_id]["prompt"],
            "seed": cards[card_id]["seed"],
            "path": str(ASSETS / cards[card_id]["path"]),
        }
        for card_id in selected
    ]
    events = stage(
        {"stage": "generate", "items": items, "settings": settings},
        "catalog-v2/card-images" + ("-partial" if only is not None else ""),
    )
    fields = (
        "ref_en",
        "aspects",
        "aspects_ja",
        "label",
        "profile_label",
        "subject_id",
        "profile_id",
        "axis_levels",
    )
    images = image_records(
        events,
        extra=lambda key: {field: cards[key][field] for field in fields},
    )
    if only is not None:
        # Untouched cards keep the manifest entry their own review is bound to.
        previous = (read_json(ASSETS / "catalog-v2.json", {}) or {}).get("images")
        images = {**(previous if isinstance(previous, dict) else {}), **images}
    images = {card_id: images[card_id] for card_id in cards if card_id in images}
    write_json(
        ASSETS / "catalog-v2.json",
        {
            "version": 2,
            "catalog_id": "catalog-v2",
            "generation": settings,
            "token_validation": token_report,
            "images": images,
        },
    )
    print(
        f"cards prepared ({len(items)} generated, {len(images)} in the "
        "manifest); review them in configs/cards-v2-review.json",
        flush=True,
    )


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
    """Every sample is a real reviewed selection run through the served policy."""
    catalog = active_catalog()
    policy_id, policy = default_policy()
    provenance = web_provenance()
    samples = []
    items = []
    for prefix, label, card_ids, strength, aspects in SAMPLE_SELECTIONS:
        snapshot = {
            "revision": 0,
            **normalize_preferences(
                {
                    "cards": [
                        {"card_id": card_id, "strength": strength, "aspects": aspects}
                        for card_id in card_ids
                    ],
                    "aspect_gains": {aspect: 1 for aspect in ASPECTS},
                },
                catalog,
                commit=True,
                selection=CONFIG["selection"],
            ),
        }
        for topic in CONFIG["topics"][:2]:
            sample_id = f"{prefix}-{topic['id']}"
            personalization = build_personalization(
                snapshot,
                prompt=target_prompt(topic),
                policy=policy,
                provenance=provenance,
                catalog=catalog,
            )
            personalization["policy_id"] = policy_id
            samples.append(
                {
                    "id": sample_id,
                    "topic_id": topic["id"],
                    "label": f"{label} · {topic['label']}",
                    "preference": snapshot,
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
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="CARD_ID",
        help="regenerate just these cards and keep the rest of the manifest",
    )
    args = parser.parse_args()
    if args.only is not None and args.step != "cards":
        parser.error("--only applies to `cards`")
    if args.step == "cards":
        prepare_cards(only=args.only)
    else:
        {"generic": prepare_generic, "samples": prepare_samples}[args.step]()


if __name__ == "__main__":
    main()
