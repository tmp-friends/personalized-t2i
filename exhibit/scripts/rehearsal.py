#!/usr/bin/env python3
"""Actual sequential localhost generation timings, with every session destroyed."""

import argparse
import json
import math
import time

import httpx
from exhibit.config import CONFIG, OUTPUTS, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--url", default="http://127.0.0.1:7860")
    a = p.parse_args()
    rows = []
    with httpx.Client(base_url=a.url, timeout=15) as client:

        def call(method, path, **kwargs):
            r = client.request(method, path, **kwargs)
            r.raise_for_status()
            return r.json()

        for n in range(a.sessions):
            session = call("POST", "/api/sessions")
            sid = session["id"]
            try:
                for i, pair in enumerate(CONFIG["pairs"]):
                    call(
                        "POST",
                        f"/api/sessions/{sid}/choices",
                        json={
                            "pair_id": pair["id"],
                            "chosen_id": pair["image_ids"][(n + i) % 2],
                        },
                    )
                started = time.monotonic()
                call(
                    "POST",
                    f"/api/sessions/{sid}/runs",
                    json={
                        "topic_id": CONFIG["topics"][n % 6]["id"],
                        "edits": {},
                        "request_id": f"rehearsal-{n}",
                    },
                )
                while time.monotonic() - started < 135:
                    data = call("GET", f"/api/sessions/{sid}")
                    if data["run"]["status"] == "done":
                        break
                    time.sleep(0.2)
                run = data["run"]
                row = {
                    "index": n,
                    "topic": run["topic_id"],
                    "seconds": round(time.monotonic() - started, 3),
                    "images": len(run["personalized"]),
                    "mode": run["mode"],
                    "error": run.get("error"),
                    "timings": run["timings"],
                    "metrics": run.get("metrics", []),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            finally:
                call("DELETE", f"/api/sessions/{sid}")
            while call("GET", "/api/health")["gpu_busy"]:
                time.sleep(0.1)
    times = sorted(r["seconds"] for r in rows)
    result = {
        "sessions": len(rows),
        "successes": sum(r["images"] == 4 and not r["error"] for r in rows),
        "p95_seconds": times[math.ceil(0.95 * len(times)) - 1],
        "median_seconds": times[len(times) // 2],
        "scope": "ZIPP-style generation only; PIGReward disabled; every stage loaded in a fresh subprocess",
        "rows": rows,
    }
    write_json(OUTPUTS / "rehearsal.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
