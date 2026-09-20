#!/usr/bin/env python3
"""Checked clauses for the shipped asset set. Re-review after regenerating pairs."""

from exhibit.config import ASSETS, CONFIG, OUTPUTS, read_json, write_json
from exhibit.domain import digest

CLAUSES = [
    (
        "p1-a",
        "color",
        "warm",
        "The selected character illustration uses warmer orange and golden tones in the clothes and background.",
        "服や背景に、暖かなオレンジや金色を使ったほうを選びました。",
    ),
    (
        "p1-b",
        "color",
        "cool",
        "The selected character illustration uses more cool teal in the hat, clothes, and town background than the warmer alternative.",
        "帽子や服、街の背景に涼しげな青緑を使ったほうを選びました。",
    ),
    (
        "p2-a",
        "lighting",
        "soft",
        "The selected library portrait has softer, more diffused light on the character than the other image.",
        "人物に柔らかな光が当たるほうを選びました。",
    ),
    (
        "p2-b",
        "lighting",
        "dramatic",
        "The selected library portrait has stronger light and shadow contrast around the character and bookshelves.",
        "人物や本棚の光と影が、よりはっきりしたほうを選びました。",
    ),
    (
        "p3-a",
        "composition",
        "spacious",
        "The selected illustration shows more of the traveler and the surrounding grassland and sky.",
        "旅人の周りに草原や空が広く見えるほうを選びました。",
    ),
    (
        "p3-b",
        "composition",
        "close",
        "The selected illustration frames the traveler more closely, with the face filling more of the image.",
        "旅人の顔を近くから、大きく描いたほうを選びました。",
    ),
    (
        "p4-a",
        "texture",
        "painterly",
        "The selected witch illustration has softer, blended shading and watercolor-like flowers compared with the outlined alternative.",
        "人物や花を、絵筆のように柔らかく描いたほうを選びました。",
    ),
    (
        "p4-b",
        "texture",
        "cel_shaded",
        "The selected witch illustration has clearer outlines and more distinct areas of color than the softer alternative.",
        "人物や花の輪郭と色の境界を、くっきり描いたほうを選びました。",
    ),
    (
        "p5-a",
        "mood",
        "calm",
        "The selected cafe character has a gentle, closed-mouth expression and a calmer atmosphere than the laughing alternative.",
        "穏やかな表情で、落ち着いた雰囲気のほうを選びました。",
    ),
    (
        "p5-b",
        "mood",
        "lively",
        "The selected cafe character has a laughing, open-mouth expression and a more cheerful atmosphere.",
        "笑顔が大きく、楽しげな雰囲気のほうを選びました。",
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
                "review_note": (
                    "The reverse-direction VLM description conflicted with visual inspection. "
                    "The reviewed clause records the more defined outlines and color boundaries visible in p4-b."
                    if image_id == "p4-b"
                    else ""
                ),
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
