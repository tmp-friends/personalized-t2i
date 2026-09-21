#!/usr/bin/env python3
"""Actual sequential localhost timings: selection, then one plain/personalized comparison."""

import argparse
import json
import math
import time

from exhibit.config import CONFIG, OUTPUTS, write_json
from exhibit.domain import CARDS


def selection_for(index):
    """A rotating 3-card selection so consecutive sessions differ."""
    ids = list(CARDS)
    return [
        {"card_id": ids[(index * 3 + offset) % len(ids)], "aspects_off": []}
        for offset in range(CONFIG["selection"]["min"])
    ]


def drive(client, index, deadline_seconds):
    session = client.post("/api/sessions").json()
    sid = session["id"]
    row = {"index": index, "error": None}
    try:
        selection = selection_for(index)
        client.put(
            f"/api/sessions/{sid}/selection", json={"cards": selection}
        ).raise_for_status()
        topic_id = CONFIG["topics"][index % len(CONFIG["topics"])]["id"]
        started = time.monotonic()
        client.post(
            f"/api/sessions/{sid}/runs",
            json={
                "topic_id": topic_id,
                "request_id": f"rehearsal-{index}-0",
            },
        ).raise_for_status()
        snapshot = wait(client, sid, deadline_seconds)
        row["seconds"] = round(time.monotonic() - started, 3)
        run = snapshot["run"]
        row.update(
            topic=run["topic_id"],
            images=len(run["personal"]),
            mode=run["mode"],
            error=run.get("error"),
            timings=run["timings"],
        )
    finally:
        client.delete(f"/api/sessions/{sid}")
    return row


def wait(client, sid, deadline_seconds):
    deadline = time.monotonic() + deadline_seconds
    snapshot = client.get(f"/api/sessions/{sid}").json()
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/sessions/{sid}").json()
        if snapshot["run"]["status"] == "done":
            return snapshot
        time.sleep(0.2)
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--url",
        default=None,
        help="Drive a running server instead of the in-process app",
    )
    args = parser.parse_args()
    rows = []
    with client_for(args.url) as client:
        for index in range(args.sessions):
            row = drive(client, index, args.timeout)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            while client.get("/api/health").json()["gpu_busy"]:
                time.sleep(0.1)
    times = sorted(row["seconds"] for row in rows if "seconds" in row)
    result = {
        "sessions": len(rows),
        "successes": sum(
            1 for row in rows if not row["error"] and row.get("images") == 4
        ),
        "p95_seconds": times[math.ceil(0.95 * len(times)) - 1] if times else None,
        "median_seconds": times[len(times) // 2] if times else None,
        "scope": "FAN live generation only; every stage loads in a fresh subprocess",
        "rows": rows,
    }
    write_json(OUTPUTS / "rehearsal.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False
        ),
        flush=True,
    )


def client_for(url):
    if url:
        import httpx

        return httpx.Client(base_url=url, timeout=30)
    from fastapi.testclient import TestClient

    from exhibit import app as module

    return TestClient(module.app)


if __name__ == "__main__":
    main()
