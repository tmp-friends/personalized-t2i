import threading
import time

import pytest
from exhibit.config import CONFIG
from exhibit.domain import CARDS
from exhibit.service import Conflict, Service

IDS = list(CARDS)


def cards(count=3):
    return [{"card_id": IDS[i], "aspects_off": []} for i in range(count)]


def settle(service, seconds=5):
    deadline = time.monotonic() + seconds
    while service.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not service.busy, "Coordinator did not release the GPU lease"


def start(service, sid, request_id="r0", topic="cat"):
    return service.start_run(sid, topic, request_id)


def prepared(service):
    session = service.create_session()
    service.set_selection(session["id"], cards())
    return session["id"]


def test_single_session_and_reset_invalidates_old_id(tmp_path):
    service = Service(tmp_path)
    first = service.create_session()
    assert len(first["card_order"]) == 16
    with pytest.raises(Conflict):
        service.create_session()
    service.reset(first["id"])
    with pytest.raises(KeyError):
        service.snapshot(first["id"])
    assert service.create_session()["id"] != first["id"]


def test_selection_is_validated_and_required_before_a_run(tmp_path, assets):
    service = Service(tmp_path)
    sid = service.create_session()["id"]
    with pytest.raises(ValueError):
        service.set_selection(sid, cards(2))
    with pytest.raises(ValueError):
        service.set_selection(sid, [{"card_id": "no-such-card"}] + cards(2))
    with pytest.raises(ValueError):
        start(service, sid)
    snapshot = service.set_selection(sid, cards(5))
    assert [entry["card_id"] for entry in snapshot["selection"]] == IDS[:5]


def test_a_run_shows_the_plain_and_personalized_images_side_by_side(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    started = start(service, sid)["run"]
    # The plain images come from the generic cache, so they are there at once.
    assert [image["id"] for image in started["plain"]] == [
        f"plain-{i}" for i in range(4)
    ]
    settle(service)
    run = service.snapshot(sid)["run"]
    assert run["status"] == "done" and run["mode"] == "live"
    assert "variants" not in run
    assert [image["id"] for image in run["personal"]] == [
        f"personal-{i}" for i in range(4)
    ]
    assert [image["seed"] for image in run["personal"]] == [
        image["seed"] for image in run["plain"]
    ]
    assert run["personalization"]["alpha"] == CONFIG["alpha"]
    assert all(
        ref["weight"] == len(ref["card_ids"]) for ref in run["personalization"]["refs"]
    )
    for image in run["plain"] + run["personal"]:
        assert image["url"] == f"/api/sessions/{sid}/images/{image['relative_path']}"
        assert service.artifact(sid, image["relative_path"]).is_file()


def test_returning_to_a_topic_reuses_the_exact_cache(tmp_path, assets, stage_stub):
    service = Service(tmp_path)
    sid = prepared(service)
    first = start(service, sid, "r0")["run"]["id"]
    settle(service)
    images = service.snapshot(sid)["run"]["personal"]
    start(service, sid, "r1", topic="tokyo")
    settle(service)
    start(service, sid, "r2")
    settle(service)
    run = service.snapshot(sid)["run"]
    assert run["id"] != first
    assert run["mode"] == "exact-cache" and run["status"] == "done"
    assert run["personal"] == images
    assert len(stage_stub) == 2


def test_request_id_is_idempotent_and_rejects_reuse_with_new_input(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    first = start(service, sid, "same")["run"]
    settle(service)
    again = start(service, sid, "same")["run"]
    assert again["id"] == first["id"]
    assert len(stage_stub) == 1
    with pytest.raises(Conflict):
        start(service, sid, "same", topic="tokyo")


def test_results_with_a_stale_personalization_hash_are_dropped(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    run = service.snapshot(sid)["run"]
    assert not service.publish(
        sid, run["id"], "not-the-hash", {"status": "done", "error": "stale"}
    )
    assert service.snapshot(sid)["run"]["error"] is None


def blocking_runner(started, release):
    def runner(service, session):
        started.set()
        release.wait(2)
        service.publish(
            session["id"],
            session["run"]["id"],
            session["run"]["personalization"]["hash"],
            {"status": "done", "error": "late"},
        )

    return runner


def test_cancelling_a_run_returns_the_visitor_to_their_selection(tmp_path, assets):
    started, release = threading.Event(), threading.Event()
    service = Service(tmp_path, runner=blocking_runner(started, release))
    sid = prepared(service)
    run_id = start(service, sid)["run"]["id"]
    assert started.wait(1)
    snapshot = service.cancel_run(sid)
    # The comparison never completed, so the run is discarded and the selection can change.
    assert snapshot["run"] is None
    assert snapshot["selection"]
    assert service.set_selection(sid, cards(4))["run"] is None
    # The still-exiting worker keeps its directory until the session ends.
    assert (tmp_path / "sessions" / sid / run_id).exists()
    release.set()
    settle(service)
    assert service.snapshot(sid)["run"] is None
    service.reset(sid)
    assert not (tmp_path / "sessions" / sid).exists()


def test_cancelling_a_finished_run_keeps_it(tmp_path, assets, stage_stub):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    run = service.cancel_run(sid)["run"]
    assert run["status"] == "done" and len(run["personal"]) == 4


def test_a_new_topic_starts_a_fresh_comparison(tmp_path, assets, stage_stub):
    service = Service(tmp_path)
    sid = prepared(service)
    first = start(service, sid, "r0")["run"]["id"]
    settle(service)
    run = start(service, sid, "r1", topic="tokyo")["run"]
    settle(service)
    assert run["id"] != first
    assert run["topic_id"] == "tokyo"
    run = service.snapshot(sid)["run"]
    assert len(run["personal"]) == 4


def test_a_topic_change_while_generating_is_refused(tmp_path, assets):
    started, release = threading.Event(), threading.Event()
    service = Service(tmp_path, runner=blocking_runner(started, release))
    sid = prepared(service)
    start(service, sid)
    assert started.wait(1)
    with pytest.raises(Conflict):
        start(service, sid, "other", topic="tokyo")
    release.set()
    settle(service)


def test_a_sample_is_replaced_by_a_live_run_on_the_same_topic(
    tmp_path, assets, asset_tree, stage_stub
):
    import json

    service = Service(tmp_path)
    sid = prepared(service)
    sample = json.loads((asset_tree["root"] / "samples.json").read_text())[0]
    sample_run = service.sample(sid, sample["id"])["run"]
    run = start(service, sid, "r0", topic=sample["topic_id"])["run"]
    settle(service)
    assert run["id"] != sample_run["id"]
    assert sample_run["mode"] == "sample" and run["mode"] == "live"
    assert len(service.snapshot(sid)["run"]["personal"]) == 4


def test_reset_during_a_job_blocks_the_next_session_until_the_worker_exits(
    tmp_path, assets
):
    started = threading.Event()
    release = threading.Event()

    def runner(service, session):
        started.set()
        release.wait(2)

    service = Service(tmp_path, runner=runner)
    sid = prepared(service)
    start(service, sid)
    assert started.wait(1)
    with pytest.raises(Conflict):
        start(service, sid, "another")
    service.reset(sid)
    with pytest.raises(Conflict):
        service.create_session()
    release.set()
    settle(service)
    nxt = service.create_session()
    assert nxt["run"] is None
    assert not (tmp_path / "sessions" / sid).exists()


def test_idle_reset_and_safe_artifact_resolution(tmp_path, assets):
    service = Service(tmp_path)
    session = service.create_session()
    with pytest.raises((ValueError, FileNotFoundError)):
        service.artifact(session["id"], "../../secret")
    with pytest.raises(FileNotFoundError):
        service.artifact(session["id"], "unknown/plain-0.png")
    service.session["last_active"] = time.monotonic() - CONFIG["idle_seconds"] - 1
    service.expire_idle()
    with pytest.raises(KeyError):
        service.snapshot(session["id"])


def test_sample_shows_both_sets_and_is_rejected_when_inconsistent(
    tmp_path, assets, asset_tree
):
    import json

    from exhibit.config import write_json

    service = Service(tmp_path)
    sid = service.create_session()["id"]
    samples = json.loads((asset_tree["root"] / "samples.json").read_text())
    run = service.sample(sid, samples[0]["id"])["run"]
    assert run["mode"] == "sample"
    assert len(run["plain"]) == 4 and len(run["personal"]) == 4
    samples[0]["images"][0]["seed"] = -999
    write_json(asset_tree["root"] / "samples.json", samples)
    service.reset(sid)
    sid = service.create_session()["id"]
    with pytest.raises(ValueError, match="inconsistent"):
        service.sample(sid, samples[0]["id"])
    assert service.snapshot(sid)["run"] is None
