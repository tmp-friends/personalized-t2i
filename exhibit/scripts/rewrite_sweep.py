#!/usr/bin/env python3
"""Six topics x three histories: actual local rewriting and dual-tokenizer checks."""

import threading
import time

from exhibit.config import ASSETS, CONFIG, OUTPUTS, read_json, write_json
from exhibit.domain import build_persona, effective_context
from exhibit.gpu import run_stage


def main():
    evidence = read_json(ASSETS / "evidence.json")
    items = []
    for index, answers in enumerate(
        [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0]]
    ):
        choices = [
            {"pair_id": p["id"], "chosen_id": p["image_ids"][i]}
            for p, i in zip(CONFIG["pairs"], answers)
        ]
        context = effective_context(build_persona(choices, evidence), {})
        for topic in CONFIG["topics"]:
            items.append(
                {
                    "id": f"h{index + 1}-{topic['id']}",
                    "topic": topic,
                    "context": context,
                }
            )
    events = []
    timing = run_stage(
        {"stage": "rewrite", "items": items},
        OUTPUTS / "rewrite-sweep",
        threading.Event(),
        time.monotonic() + 120,
        events.append,
    )
    rewrites = [e for e in events if e["type"] == "rewrite"]
    result = {
        "cases": len(rewrites),
        "valid": sum(e["valid"] for e in rewrites),
        "timing": timing,
        "events": events,
        "scope": "Original prompt prefix and both SDXL tokenizer limits; not visual quality evaluation",
    }
    write_json(OUTPUTS / "rewrite-sweep.json", result)
    print({k: v for k, v in result.items() if k != "events"})
    assert len(rewrites) == 18 and all(e["valid"] for e in rewrites)


if __name__ == "__main__":
    main()
