import threading
import time

import pytest
from exhibit.service import Conflict, Service


def test_single_session_and_reset_invalidates_old_id(tmp_path):
    s = Service(tmp_path)
    a = s.create_session()
    with pytest.raises(Conflict):
        s.create_session()
    s.reset(a["id"])
    with pytest.raises(KeyError):
        s.snapshot(a["id"])
    assert s.create_session()["id"] != a["id"]


def test_image_identity_is_preserved_and_invalid_answers_rejected(tmp_path):
    s = Service(tmp_path)
    a = s.create_session()
    p = a["pairs"][0]
    chosen = p["image_ids"][1]
    result = s.answer(a["id"], p["id"], chosen)
    assert result["choices"][0]["chosen_id"] == chosen
    assert result["choices"][0]["display_order"] == p["image_ids"]
    with pytest.raises(ValueError):
        s.answer(a["id"], p["id"], "foreign")


def test_insufficient_choices_and_idempotent_answers(tmp_path):
    s = Service(tmp_path)
    a = s.create_session()
    for p in a["pairs"]:
        s.answer(a["id"], p["id"], None)
    assert len(s.snapshot(a["id"])["choices"]) == 5
    with pytest.raises(ValueError):
        s.start_run(a["id"], "cat", {}, "request-1")
    p = a["pairs"][0]
    s.answer(a["id"], p["id"], p["image_ids"][0])
    assert len(s.snapshot(a["id"])["choices"]) == 5


def test_reset_during_job_blocks_next_until_worker_exit(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def runner(service, session, job):
        started.set()
        release.wait(2)
        service.publish(
            session["id"],
            job["id"],
            job["context"]["hash"],
            {"status": "done", "winner_id": "stale"},
        )

    s = Service(tmp_path, runner=runner)
    a = s.create_session()
    for p in a["pairs"]:
        s.answer(a["id"], p["id"], p["image_ids"][0])
    job = s.start_run(a["id"], "cat", {}, "one")["run"]
    assert started.wait(1)
    assert s.start_run(a["id"], "cat", {}, "one")["run"]["id"] == job["id"]
    with pytest.raises(Conflict):
        s.start_run(a["id"], "cat", {}, "two")
    s.reset(a["id"])
    with pytest.raises(Conflict):
        s.create_session()
    release.set()
    for _ in range(100):
        if not s.busy:
            break
        time.sleep(0.01)
    b = s.create_session()
    assert b["run"] is None
    assert not (tmp_path / "sessions" / a["id"]).exists()


def test_idle_reset_and_safe_artifact_resolution(tmp_path):
    s = Service(tmp_path)
    a = s.create_session()
    with pytest.raises((ValueError, FileNotFoundError)):
        s.artifact(a["id"], "../../secret")
    s.session["last_active"] = time.monotonic() - 91
    s.expire_idle()
    with pytest.raises(KeyError):
        s.snapshot(a["id"])


def test_cancel_run_invalidates_delayed_publication_and_retains_choices(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def runner(service, session, job):
        started.set()
        release.wait(2)
        service.publish(
            session["id"],
            job["id"],
            job["context"]["hash"],
            {"status": "done", "winner_id": "stale"},
        )

    s = Service(tmp_path, runner=runner)
    session = s.create_session()
    for pair in session["pairs"]:
        s.answer(session["id"], pair["id"], pair["image_ids"][0])
    s.start_run(session["id"], "cat", {}, "cancel-me")
    assert started.wait(1)
    s.cancel_run(session["id"])
    assert s.snapshot(session["id"])["run"] is None
    assert len(s.snapshot(session["id"])["choices"]) == 5
    release.set()
    for _ in range(100):
        if not s.busy:
            break
        time.sleep(0.01)
    assert s.snapshot(session["id"])["run"] is None


def test_partial_generation_survives_worker_failure_without_recommendation(
    tmp_path, monkeypatch
):
    import exhibit.service as module
    from exhibit.gpu import GPUError

    def stage(request, directory, cancel, deadline, on_event):
        if request["stage"] == "rewrite":
            on_event(
                {
                    "type": "rewrite",
                    "valid": True,
                    "prompt": "One cat sitting by a window. Warm colors.",
                }
            )
        elif request["stage"] == "generate":
            item = request["items"][0]
            Path(item["path"]).write_bytes(b"completed image")
            on_event(
                {
                    "type": "image",
                    **item,
                    "sha256": "fixture",
                    "context_hash": request["context_hash"],
                }
            )
            raise GPUError("worker_exit_oom")
        return {"returncode": 0, "wall_seconds": 0.01}

    from pathlib import Path

    monkeypatch.setattr(module, "run_stage", stage)
    service = Service(tmp_path)
    session = service.create_session()
    for pair in session["pairs"]:
        service.answer(session["id"], pair["id"], pair["image_ids"][0])
    service.start_run(session["id"], "cat", {}, "partial")
    for _ in range(200):
        if not service.busy:
            break
        time.sleep(0.01)
    run = service.snapshot(session["id"])["run"]
    assert len(run["personalized"]) == 1
    assert run["winner_id"] is None
    assert run["error"] == "worker_exit_oom"
