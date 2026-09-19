#!/usr/bin/env python3
"""Measure real evaluator parsing and order consistency. Does not enable live mode."""

import argparse
import itertools
import threading
import time

from exhibit.config import ASSETS, CONFIG, OUTPUTS, REPO, read_json, write_json
from exhibit.gpu import GPUError, run_stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    records = []
    timings = []
    errors = []
    samples = read_json(ASSETS / "samples.json", [])
    for n, s in enumerate(samples[:1] if a.smoke else samples):
        candidates = [{**c, "path": str(ASSETS / c["path"])} for c in s["images"]]
        pairs = list(itertools.combinations([c["id"] for c in candidates], 2))[
            : 1 if a.smoke else 4 if n < 2 else 3
        ]
        pairs = [list(pair) for pair in pairs for pair in (pair, pair[::-1])]
        topic = next(t for t in CONFIG["topics"] if t["id"] == s["topic_id"])
        request = {
            "stage": "evaluate",
            "candidates": candidates,
            "pairs": pairs,
            "context": s["context"],
            "basic_prompt_en": topic["basic_prompt_en"],
        }
        events = []

        def receive(e, events=events, sample_id=s["id"]):
            events.append(e)
            if e["type"] == "judgment":
                print(sample_id, e["status"], e["winner_id"], flush=True)

        try:
            timings.append(
                run_stage(
                    request,
                    OUTPUTS / "pigreward-probe" / s["id"],
                    threading.Event(),
                    time.monotonic() + 480,
                    receive,
                )
            )
        except (GPUError, OSError) as exc:
            errors.append(str(exc))
        records.extend({"sample_id": s["id"], **e} for e in events)
    judgments = [e for e in records if e["type"] == "judgment"]
    complete_pairs = [judgments[i : i + 2] for i in range(0, len(judgments) - 1, 2)]
    consistent = sum(
        a["status"] == b["status"] == "valid" and a["winner_id"] == b["winner_id"]
        for a, b in complete_pairs
    )
    forward = sum(a["status"] == "valid" for a, b in complete_pairs)
    result = {
        "probe": "smoke" if a.smoke else "g0",
        "input_pairs": len(complete_pairs),
        "forward_parseable": forward,
        "reverse_consistent": consistent,
        "g0_passed": not a.smoke
        and len(complete_pairs) == 20
        and forward >= 18
        and consistent >= 18
        and not errors,
        "timings": timings,
        "errors": errors,
        "records": records,
        "model": read_json(REPO / "pigreward-repro/configs/model.json"),
    }
    write_json(
        OUTPUTS / ("pigreward-smoke.json" if a.smoke else "pigreward-g0.json"), result
    )
    print(
        {k: v for k, v in result.items() if k not in ("records", "model")}, flush=True
    )


if __name__ == "__main__":
    main()
