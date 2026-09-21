#!/usr/bin/env python3
"""Sequential localhost rehearsal of one whole revisioned session.

Each session drives the API of design §8.1 end to end: create, round 1, a
committed selection with explicit aspects, a run, a same-seed redraw after an
aspect gain change, a return to the identical content (which must be served from
the exact cache), a cancelled run and the delete. Successes, failures, the per
run generation time and the 120 s per-run deadline are recorded; peak VRAM is
read from the GPU worker's own event log when the server runs on this machine.

    exhibit/scripts/rehearsal.py --sessions 20                  # in-process app
    exhibit/scripts/rehearsal.py --sessions 20 --url http://127.0.0.1:7860
"""

import argparse
import json
import math
import time
from pathlib import Path

from exhibit.config import CONFIG, OUTPUTS, write_json
from exhibit.domain import ASPECTS

# One run per step; the labels are what the summary counts.
STEPS = ("initial", "regenerate", "identical")


def selection_for(card_ids, index, *, count=None):
    """A rotating, fully answered selection taken from the cards round 1 showed."""
    count = count or CONFIG["selection"]["min"]
    if len(card_ids) < count:
        raise ValueError(f"round 1 showed {len(card_ids)} cards, need {count}")
    chosen = [card_ids[(index + offset) % len(card_ids)] for offset in range(count)]
    return [
        {
            "card_id": card_id,
            "strength": 1 + (index + offset) % 2,
            # Every card names at least one aspect; an unanswered card cannot commit.
            "aspects": sorted(set(ASPECTS[: 1 + (index + offset) % len(ASPECTS)])),
        }
        for offset, card_id in enumerate(chosen)
    ]


def percentile(values, fraction):
    """Nearest-rank percentile over the measured values; None when nothing ran."""
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def peak_vram_mib(session_id, run_id, root=None):
    """The worker's own metrics event, when the server generated on this machine."""
    base = Path(root or OUTPUTS) / "sessions" / session_id / run_id / "personal"
    peaks = []
    try:
        logs = sorted(base.glob("generate-*/events.jsonl"))
    except OSError:
        return None
    for path in logs:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "metrics":
                value = event.get("peak_vram_mib")
                if isinstance(value, (int, float)):
                    peaks.append(float(value))
    return max(peaks) if peaks else None


def summarize(rows, *, deadline_seconds):
    """Aggregate sessions and runs without turning a failure into a measurement."""
    runs = [run for row in rows for run in row.get("runs", [])]
    finished = [run for run in runs if run.get("error") is None]
    times = [run.get("seconds") for run in finished]
    vram = [
        run["peak_vram_mib"] for run in runs if run.get("peak_vram_mib") is not None
    ]
    failures = [
        {
            "index": row["index"],
            "failed_step": row.get("failed_step"),
            "error": row.get("error"),
        }
        for row in rows
        if not row.get("ok")
    ]
    over = [
        {"index": run["index"], "step": run["step"], "seconds": run["seconds"]}
        for run in finished
        if run.get("seconds") is not None and run["seconds"] > deadline_seconds
    ]
    return {
        "sessions": len(rows),
        "successes": sum(1 for row in rows if row.get("ok")),
        "failures": failures,
        "runs": len(runs),
        "runs_measured": len(finished),
        "median_seconds": percentile(times, 0.5),
        "p95_seconds": percentile(times, 0.95),
        "max_seconds": max(times) if times else None,
        "deadline_seconds": deadline_seconds,
        "runs_over_deadline": over,
        "peak_vram_mib": max(vram) if vram else None,
        "exact_cache_runs": sum(1 for run in runs if run.get("mode") == "exact-cache"),
        "cancelled_runs": sum(1 for row in rows if row.get("cancelled")),
        "scope": "FAN live generation only; every run loads in a fresh subprocess",
    }


class Session:
    """The API calls of one rehearsal session, each raising on an unexpected reply."""

    def __init__(self, client, index, deadline_seconds):
        self.client = client
        self.index = index
        self.deadline_seconds = deadline_seconds
        self.id = None
        self.revision = 0
        self.runs = []

    def _json(self, response):
        response.raise_for_status()
        return response.json()

    def create(self):
        snapshot = self._json(self.client.post("/api/sessions"))
        self.id = snapshot["id"]
        self.revision = snapshot["revision"]
        return snapshot

    def put_selection(self, cards, gains, *, commit=True):
        snapshot = self._json(
            self.client.put(
                f"/api/sessions/{self.id}/selection",
                json={
                    "expected_revision": self.revision,
                    "cards": cards,
                    "aspect_gains": gains,
                    "commit": commit,
                },
            )
        )
        self.revision = snapshot["revision"]
        return snapshot

    def start_run(self, topic_id, request_id):
        snapshot = self._json(
            self.client.post(
                f"/api/sessions/{self.id}/runs",
                json={
                    "topic_id": topic_id,
                    "request_id": request_id,
                    "expected_revision": self.revision,
                },
            )
        )
        if not snapshot["run"]:
            raise AssertionError("the run was not started")
        return snapshot

    def wait(self, run_id, revision):
        """Poll by run id and revision so a stale run never ends this wait."""
        limit = time.monotonic() + self.deadline_seconds
        while True:
            snapshot = self._json(self.client.get(f"/api/sessions/{self.id}"))
            run = snapshot["run"]
            if not run or run["id"] != run_id:
                raise AssertionError("the run was replaced while waiting")
            if run["preference_revision"] != revision:
                raise AssertionError("the run belongs to another revision")
            if run["status"] == "done":
                return run
            if time.monotonic() > limit:
                raise TimeoutError(f"run {run_id} did not finish in time")
            time.sleep(0.2)

    def run(self, topic_id, step, *, expect_mode=None):
        started = time.monotonic()
        snapshot = self.start_run(topic_id, f"rehearsal-{self.index}-{step}")
        run_id = snapshot["run"]["id"]
        revision = snapshot["run"]["preference_revision"]
        record = {
            "index": self.index,
            "step": step,
            "run_id": run_id,
            "revision": revision,
            "topic_id": topic_id,
            "seconds": None,
            "mode": None,
            "images": 0,
            "within_deadline": None,
            "peak_vram_mib": None,
            "error": None,
        }
        self.runs.append(record)
        run = self.wait(run_id, revision)
        record.update(
            seconds=round(time.monotonic() - started, 3),
            mode=run["mode"],
            images=len(run["personal"]),
            error=run.get("error"),
            peak_vram_mib=peak_vram_mib(self.id, run_id),
        )
        record["within_deadline"] = record["seconds"] <= self.deadline_seconds
        if run.get("error"):
            raise AssertionError(f"run {step} failed: {run['error']}")
        if len(run["personal"]) != len(CONFIG["seeds"]):
            raise AssertionError(f"run {step} produced {len(run['personal'])} images")
        if expect_mode and run["mode"] != expect_mode:
            raise AssertionError(
                f"run {step} was {run['mode']}, expected {expect_mode}"
            )
        return run

    def cancel(self, topic_id):
        """Start one more run and cancel it while it is still generating."""
        snapshot = self.start_run(topic_id, f"rehearsal-{self.index}-cancel")
        if snapshot["run"]["status"] == "done":
            return False
        snapshot = self._json(self.client.post(f"/api/sessions/{self.id}/cancel"))
        if snapshot["run"] is not None:
            raise AssertionError("cancel left the unfinished run in place")
        return True

    def delete(self):
        self._json(self.client.delete(f"/api/sessions/{self.id}"))


def drive(client, index, deadline_seconds):
    """One session; a failure is recorded with the step it happened in."""
    session = Session(client, index, deadline_seconds)
    row = {
        "index": index,
        "session_id": None,
        "ok": False,
        "failed_step": None,
        "error": None,
        "cancelled": False,
        "runs": session.runs,
    }
    step = "create"
    try:
        snapshot = session.create()
        row["session_id"] = session.id
        cards = selection_for(snapshot["rounds"][0]["card_ids"], index)
        gains = {aspect: 1 for aspect in ASPECTS}
        topics = CONFIG["topics"]
        topic_id = topics[index % len(topics)]["id"]
        other_id = topics[(index + 1) % len(topics)]["id"]

        step = "selection"
        session.put_selection(cards, gains)
        step = "initial"
        session.run(topic_id, "initial", expect_mode="live")

        # A stronger aspect changes the references, so the same seeds are redrawn.
        step = "regenerate"
        session.put_selection(cards, {**gains, "color": 2})
        session.run(topic_id, "regenerate", expect_mode="live")

        # Back to the exact first content: the same hash, so the same images.
        step = "identical"
        session.put_selection(cards, gains)
        session.run(topic_id, "identical", expect_mode="exact-cache")

        step = "cancel"
        row["cancelled"] = session.cancel(other_id)
        row["ok"] = True
    except Exception as error:  # noqa: BLE001 - a session failure is a measurement
        row["failed_step"] = step
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        step = "delete"
        try:
            if session.id:
                session.delete()
        except Exception as error:  # noqa: BLE001 - the delete is part of the run
            if row["ok"]:
                row["ok"] = False
                row["failed_step"] = step
                row["error"] = f"{type(error).__name__}: {error}"
    return row


def client_for(url):
    if url:
        import httpx

        return httpx.Client(base_url=url, timeout=30)
    from fastapi.testclient import TestClient

    from exhibit import app as module

    return TestClient(module.app)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument(
        "--deadline",
        type=float,
        default=float(CONFIG["timeout_seconds"]),
        help="the per-run deadline every run must meet (default: demo.json)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="drive a running server instead of the in-process app",
    )
    parser.add_argument("--output", type=Path, default=OUTPUTS / "rehearsal.json")
    args = parser.parse_args(argv)

    rows = []
    with client_for(args.url) as client:
        for index in range(args.sessions):
            row = drive(client, index, args.deadline)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            while client.get("/api/health").json()["gpu_busy"]:
                time.sleep(0.1)
    result = {**summarize(rows, deadline_seconds=args.deadline), "rows": rows}
    write_json(args.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
