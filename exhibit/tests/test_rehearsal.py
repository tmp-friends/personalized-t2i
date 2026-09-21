"""The rehearsal's own bookkeeping: percentiles, the deadline, failures, cache."""

import importlib.util
import json

import pytest
from exhibit.config import CONFIG, ROOT
from exhibit.domain import ASPECTS


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rehearsal = load_script("rehearsal")
CARD_IDS = [f"girl-c{index}" for index in range(12)]


class Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self._payload}")
        return self

    def json(self):
        return self._payload


class FakeAPI:
    """A revisioned stand-in: one session, content-addressed cache, no images."""

    def __init__(self, *, fail_step=None, seconds=0.0, images=None):
        self.session = None
        self.cache = set()
        self.fail_step = fail_step
        self.seconds = seconds
        self.images = len(CONFIG["seeds"]) if images is None else images
        self.deleted = []

    # ------------------------------------------------------------- helpers
    def _content(self):
        state = self.session
        return json.dumps([state["selection"], state["aspect_gains"]], sort_keys=True)

    def _run_view(self):
        run = self.session["run"]
        if not run:
            return None
        return {key: value for key, value in run.items() if key != "key"}

    def _snapshot(self):
        state = self.session
        return {
            "id": state["id"],
            "revision": state["revision"],
            "committed": state["committed"],
            "selection": state["selection"],
            "aspect_gains": state["aspect_gains"],
            "rounds": state["rounds"],
            "run": self._run_view(),
        }

    # -------------------------------------------------------------- routes
    def request(self, method, path, payload=None):
        parts = [part for part in path.split("/") if part]
        if path == "/api/health":
            return Response(200, {"gpu_busy": False})
        if path == "/api/sessions" and method == "POST":
            self.session = {
                "id": "sess-1",
                "revision": 0,
                "selection": [],
                "aspect_gains": {aspect: 1 for aspect in ASPECTS},
                "committed": False,
                "rounds": [{"round_index": 0, "card_ids": list(CARD_IDS)}],
                "run": None,
            }
            return Response(200, self._snapshot())
        if not self.session or parts[2] != self.session["id"]:
            return Response(404, "no session")
        state = self.session
        tail = parts[3:]
        if not tail and method == "GET":
            run = state["run"]
            if run and run["status"] != "done":
                run["polls"] += 1
                if run["polls"] >= 1:
                    run["status"] = "done"
                    if run["mode"] == "live":
                        self.cache.add(run["key"])
            return Response(200, self._snapshot())
        if not tail and method == "DELETE":
            self.deleted.append(state["id"])
            self.session = None
            return Response(200, {"ok": True})
        if tail == ["selection"] and method == "PUT":
            if payload["expected_revision"] != state["revision"]:
                return Response(409, "stale revision")
            content = {
                "selection": payload["cards"],
                "aspect_gains": payload["aspect_gains"],
                "committed": payload["commit"],
            }
            if any(not entry["aspects"] for entry in payload["cards"]):
                return Response(422, "unanswered card")
            if content != {
                "selection": state["selection"],
                "aspect_gains": state["aspect_gains"],
                "committed": state["committed"],
            }:
                state.update(content)
                state["revision"] += 1
                state["run"] = None
            return Response(200, self._snapshot())
        if tail == ["runs"] and method == "POST":
            if payload["expected_revision"] != state["revision"]:
                return Response(409, "stale revision")
            if not state["committed"]:
                return Response(422, "not committed")
            key = (payload["topic_id"], self._content())
            step = payload["request_id"].rsplit("-", 1)[-1]
            cached = key in self.cache
            state["run"] = {
                "id": f"run-{payload['request_id']}",
                "topic_id": payload["topic_id"],
                "status": "done" if cached else "generating",
                "mode": "exact-cache" if cached else "live",
                "preference_revision": state["revision"],
                "personal": [{"id": index} for index in range(self.images)],
                "error": "worker_exit_1" if step == self.fail_step else None,
                "polls": 0,
                "key": key,
            }
            return Response(200, self._snapshot())
        if tail == ["cancel"] and method == "POST":
            if state["run"] and state["run"]["status"] != "done":
                state["run"] = None
            return Response(200, self._snapshot())
        return Response(404, "not found")

    def post(self, path, json=None):
        return self.request("POST", path, json)

    def put(self, path, json=None):
        return self.request("PUT", path, json)

    def get(self, path):
        return self.request("GET", path)

    def delete(self, path):
        return self.request("DELETE", path)


def test_percentile_is_nearest_rank_and_survives_an_empty_run():
    assert rehearsal.percentile([], 0.5) is None
    assert rehearsal.percentile([None, None], 0.95) is None
    assert rehearsal.percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert rehearsal.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    values = [float(index) for index in range(1, 21)]
    assert rehearsal.percentile(values, 0.95) == 19.0
    assert rehearsal.percentile(values, 0.5) == 10.0


def test_the_deadline_and_the_missing_measurements_are_reported_separately():
    rows = [
        {
            "index": 0,
            "ok": True,
            "runs": [
                {"index": 0, "step": "initial", "seconds": 30.0, "mode": "live"},
                {
                    "index": 0,
                    "step": "identical",
                    "seconds": 121.0,
                    "mode": "exact-cache",
                    "peak_vram_mib": 9000.0,
                },
            ],
        },
        {
            "index": 1,
            "ok": False,
            "failed_step": "initial",
            "error": "TimeoutError: run did not finish in time",
            "runs": [{"index": 1, "step": "initial", "seconds": None, "error": "boom"}],
        },
    ]
    summary = rehearsal.summarize(rows, deadline_seconds=120)
    assert summary["sessions"] == 2 and summary["successes"] == 1
    assert summary["runs"] == 3 and summary["runs_measured"] == 2
    # A failed run is never averaged into the timings.
    assert summary["median_seconds"] == 30.0 and summary["max_seconds"] == 121.0
    assert summary["runs_over_deadline"] == [
        {"index": 0, "step": "identical", "seconds": 121.0}
    ]
    assert summary["peak_vram_mib"] == 9000.0
    assert summary["exact_cache_runs"] == 1
    assert summary["failures"] == [
        {
            "index": 1,
            "failed_step": "initial",
            "error": "TimeoutError: run did not finish in time",
        }
    ]


def test_a_session_commits_explicit_aspects_and_returns_to_the_exact_cache():
    api = FakeAPI()
    row = rehearsal.drive(api, 0, 120.0)

    assert row["ok"] and row["error"] is None
    assert [run["step"] for run in row["runs"]] == list(rehearsal.STEPS)
    assert [run["mode"] for run in row["runs"]] == [
        "live",
        "live",
        "exact-cache",
    ]
    # Every run is bound to the revision it was started at, and they differ.
    revisions = [run["revision"] for run in row["runs"]]
    assert revisions == [1, 2, 3]
    assert all(run["within_deadline"] for run in row["runs"])
    assert row["cancelled"] is True
    assert api.deleted == ["sess-1"]
    assert api.session is None


def test_every_selected_card_answers_at_least_one_aspect():
    cards = rehearsal.selection_for(CARD_IDS, 3)
    assert len(cards) == CONFIG["selection"]["min"]
    assert len({card["card_id"] for card in cards}) == len(cards)
    for card in cards:
        assert card["aspects"] and not set(card["aspects"]) - set(ASPECTS)
        assert card["strength"] in (1, 2)
    with pytest.raises(ValueError):
        rehearsal.selection_for(CARD_IDS[:2], 0)


def test_a_failing_run_records_its_step_and_still_deletes_the_session():
    api = FakeAPI(fail_step="regenerate")
    row = rehearsal.drive(api, 1, 120.0)

    assert row["ok"] is False
    assert row["failed_step"] == "regenerate"
    assert "worker_exit_1" in row["error"]
    # The measurement of the failing run is kept, not dropped.
    assert [run["step"] for run in row["runs"]] == ["initial", "regenerate"]
    assert row["runs"][1]["error"] == "worker_exit_1"
    assert api.deleted == ["sess-1"]
    summary = rehearsal.summarize([row], deadline_seconds=120)
    assert summary["successes"] == 0 and summary["runs_measured"] == 1


def test_a_short_run_is_a_failure_rather_than_a_success():
    row = rehearsal.drive(FakeAPI(images=2), 0, 120.0)
    assert row["ok"] is False and row["failed_step"] == "initial"
    assert "2 images" in row["error"]


def test_peak_vram_reads_the_worker_metrics_event(tmp_path):
    base = tmp_path / "sessions/sess-1/run-1/personal/generate-abc"
    base.mkdir(parents=True)
    (base / "events.jsonl").write_text(
        json.dumps({"type": "image", "id": "personal-0"})
        + "\n{ not json\n"
        + json.dumps({"type": "metrics", "peak_vram_mib": 9512.5})
        + "\n"
    )
    assert rehearsal.peak_vram_mib("sess-1", "run-1", root=tmp_path) == 9512.5
    assert rehearsal.peak_vram_mib("sess-1", "missing", root=tmp_path) is None
