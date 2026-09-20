#!/usr/bin/env python3
"""Explicit preparation only; serving never downloads or manufactures assets."""

import argparse
import threading
import time
from pathlib import Path

from exhibit.config import ASSETS, CONFIG, OUTPUTS, read_json, write_json
from exhibit.domain import AXES
from exhibit.gpu import run_stage


def stage(request, name, seconds=1800):
    events = []

    def receive(e):
        events.append(e)
        print(e["type"], e.get("id", ""), e.get("seconds", ""), flush=True)

    timing = run_stage(
        request,
        OUTPUTS / "preparation" / name,
        threading.Event(),
        time.monotonic() + seconds,
        receive,
    )
    write_json(
        OUTPUTS / "preparation" / f"{name}.json", {"events": events, "timing": timing}
    )
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["base", "analyze", "samples"])
    args = parser.parse_args()
    if args.step == "base":
        rewrites = stage(
            {
                "stage": "rewrite",
                "items": [{"id": t["id"], "topic": t} for t in CONFIG["topics"]],
            },
            "generic-rewrites",
        )
        prompts = {
            e["id"]: e for e in rewrites if e["type"] == "rewrite" and e["valid"]
        }
        if len(prompts) != 6:
            raise RuntimeError("Generic rewrite validation failed")
        write_json(ASSETS / "generic-prompts.json", prompts)
        items = []
        for pair in CONFIG["pairs"]:
            for i, value in enumerate(pair["values"]):
                items.append(
                    {
                        "id": pair["image_ids"][i],
                        "prompt": pair.get("prompts", [])[i]
                        if pair.get("prompts")
                        else pair["basic_prompt_en"]
                        + " "
                        + AXES[pair["dimension"]]["values"][value][1]
                        + ".",
                        "seed": 600 + int(pair["id"][1:]),
                        "path": str(ASSETS / "pairs" / f"{pair['image_ids'][i]}.png"),
                    }
                )
        for topic in CONFIG["topics"]:
            for i, seed in enumerate(CONFIG["seeds"]):
                items.append(
                    {
                        "id": f"{topic['id']}-{i}",
                        "prompt": prompts[topic["id"]]["prompt"],
                        "seed": seed,
                        "path": str(ASSETS / "generic" / f"{topic['id']}-{i}.png"),
                    }
                )
        events = stage({"stage": "generate", "items": items}, "base-images")
        images = {
            e["id"]: {
                **e,
                "path": str(Path(e["path"]).relative_to(ASSETS)),
                "source": "local-illustrious-xl-v2.0",
                "license": "creativeml-openrail-m",
                "asset_version": 4,
            }
            for e in events
            if e["type"] == "image"
        }
        write_json(
            ASSETS / "manifest.json",
            {
                "version": 4,
                "generation": CONFIG["generation"],
                "images": images,
                "metrics": [e for e in events if e["type"] == "metrics"],
            },
        )
    elif args.step == "analyze":
        items = []
        for pair in CONFIG["pairs"]:
            for i, image_id in enumerate(pair["image_ids"]):
                items.append(
                    {
                        "id": image_id,
                        "images": [
                            str(ASSETS / "pairs" / f"{key}.png")
                            for key in pair["image_ids"]
                        ],
                        "chosen_position": i + 1,
                        "dimension": pair["dimension"],
                        "basic_prompt_en": pair["basic_prompt_en"],
                    }
                )
        stage({"stage": "analyze", "items": items}, "vlm-evidence")
    else:
        from exhibit.domain import build_persona, effective_context

        evidence = read_json(ASSETS / "evidence.json", [])
        histories = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0]]
        requests = []
        samples = []
        for h, answers in enumerate(histories):
            choices = [
                {"pair_id": p["id"], "chosen_id": p["image_ids"][i]}
                for p, i in zip(CONFIG["pairs"], answers)
            ]
            context = effective_context(build_persona(choices, evidence), {})
            for topic in CONFIG["topics"][:2]:
                sid = f"h{h + 1}-{topic['id']}"
                requests.append({"id": sid, "topic": topic, "context": context})
                samples.append(
                    {
                        "id": sid,
                        "topic_id": topic["id"],
                        "choices": choices,
                        "context": context,
                        "mode": "sample",
                        "recommendation": None,
                        "images": [],
                    }
                )
        outputs = stage({"stage": "rewrite", "items": requests}, "sample-rewrites")
        rewrites = {
            e["id"]: e for e in outputs if e["type"] == "rewrite" and e["valid"]
        }
        items = []
        for sample in samples:
            if sample["id"] not in rewrites:
                raise RuntimeError("Sample rewrite failed")
            sample["rewrite"] = rewrites[sample["id"]]
            for i, seed in enumerate(CONFIG["seeds"]):
                items.append(
                    {
                        "id": f"{sample['id']}-{i}",
                        "prompt": sample["rewrite"]["prompt"],
                        "context_hash": sample["context"]["hash"],
                        "seed": seed,
                        "path": str(ASSETS / "samples" / f"{sample['id']}-{i}.png"),
                    }
                )
        events = stage({"stage": "generate", "items": items}, "sample-images")
        for sample in samples:
            sample["images"] = [
                {**e, "path": str(Path(e["path"]).relative_to(ASSETS))}
                for e in events
                if e["type"] == "image" and e["id"].startswith(sample["id"] + "-")
            ]
        write_json(ASSETS / "samples.json", samples)


if __name__ == "__main__":
    main()
