#!/usr/bin/env python3
"""Checked clauses for the shipped asset set. Re-review after regenerating pairs."""

from exhibit.config import ASSETS, CONFIG, OUTPUTS, read_json, write_json
from exhibit.domain import digest

CLAUSES = [
    (
        "p1-a",
        "color",
        "warm",
        "The selected harbor image has warmer colors than the other image.",
        "もう一方より、暖かな色の港を選びました。",
    ),
    (
        "p1-b",
        "color",
        "cool",
        "The selected harbor image has cooler greens and blues than the other image.",
        "もう一方より、涼しげな緑や青の港を選びました。",
    ),
    (
        "p2-a",
        "lighting",
        "soft",
        "The selected flowers have softer, more diffused lighting than the other image.",
        "花に柔らかな光が当たるほうを選びました。",
    ),
    (
        "p2-b",
        "lighting",
        "dramatic",
        "The selected flowers have stronger shadows and highlights than the other image.",
        "花の陰影がはっきりしたほうを選びました。",
    ),
    (
        "p3-a",
        "composition",
        "spacious",
        "The selected image emphasizes open sea around a distant lighthouse.",
        "遠くの灯台の周りに、海の余白が広がるほうを選びました。",
    ),
    (
        "p3-b",
        "composition",
        "close",
        "The selected image shows a closer, more detailed view of the lighthouse and rocky shore.",
        "灯台と岩場を、より近くから捉えたほうを選びました。",
    ),
    (
        "p4-a",
        "texture",
        "painterly",
        "The selected forest path has softer, more painterly texture than the other image.",
        "森の小道を、絵画のように柔らかく描いたほうを選びました。",
    ),
    (
        "p4-b",
        "texture",
        "photographic",
        "The selected forest path has sharper, more defined natural textures than the other image.",
        "森の小道の質感が、よりくっきりしたほうを選びました。",
    ),
    (
        "p5-a",
        "mood",
        "calm",
        "The selected cafe scene is quieter, with empty tables, than the other image.",
        "席が空いていて、より静かなカフェを選びました。",
    ),
    (
        "p5-b",
        "mood",
        "lively",
        "The selected cafe scene has a livelier atmosphere with people than the other image.",
        "人がいて、活気のあるカフェを選びました。",
    ),
]


def main():
    raw = read_json(OUTPUTS / "preparation/vlm-evidence.json")
    manifest = read_json(ASSETS / "manifest.json")
    outputs = {e["id"]: e for e in raw["events"] if e["type"] == "analysis"}
    evidence = []
    for image_id, axis, value, text, ja in CLAUSES:
        pair = next(p for p in CONFIG["pairs"] if image_id in p["image_ids"])
        evidence.append(
            {
                "id": image_id + "-e1",
                "pair_id": pair["id"],
                "pair_version": pair["version"],
                "chosen_id": image_id,
                "dimensions": [axis],
                "values": {axis: value},
                "text": text,
                "text_ja": ja,
                "reviewed": True,
                "review_method": "agent visual inspection + raw VLM clause review; staff acceptance pending",
                "context_source": "generic_vlm",
                "source": CONFIG["llm"],
                "raw": outputs[image_id]["raw"],
                "image_hashes": {
                    key: manifest["images"][key]["sha256"] for key in pair["image_ids"]
                },
                "basic_prompt_en": pair["basic_prompt_en"],
                "template_version": "single-axis-relative-v1",
                "template_hash": digest("single-axis-relative-v1"),
                "preprocessing": "RGB; aspect preserved; longest edge 448",
            }
        )
    write_json(ASSETS / "evidence.json", evidence)
    write_json(ASSETS / "raw-evidence.json", outputs)


if __name__ == "__main__":
    main()
