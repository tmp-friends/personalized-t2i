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


def start(service, sid, request_id="r0", alpha="mid", weights=None, topic="cat"):
    return service.start_run(sid, topic, alpha, weights or {}, request_id)


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


def test_blind_pairs_hide_the_mapping_until_reveal(tmp_path, assets, stage_stub):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    run = service.snapshot(sid)["run"]
    assert run["status"] == "done"
    assert run["plain"] == []
    assert run["variants"][0]["images"] == []
    assert run["variants"][0]["personalization"] is None
    assert run["variants"][0]["done_count"] == 4
    assert run["blind"]["mapping"] is None and run["blind"]["score"] is None
    tokens = [item["token"] for pair in run["blind"]["pairs"] for item in pair["items"]]
    assert len(set(tokens)) == 8
    for pair in run["blind"]["pairs"]:
        assert pair["ready"]
        for item in pair["items"]:
            assert set(item) == {"token", "url"}
            assert item["url"].startswith(f"/api/sessions/{sid}/images/blind/")
            assert service.artifact(sid, f"blind/{item['token']}.png").is_file()

    picks = [pair["items"][0]["token"] for pair in run["blind"]["pairs"]]
    service.pick_blind(sid, 0, picks[0])
    service.pick_blind(sid, 1, "tie")
    assert service.snapshot(sid)["run"]["blind"]["answered"] == 2

    revealed = service.reveal(sid)["run"]
    assert revealed["blind"]["revealed"]
    assert set(revealed["blind"]["mapping"].values()) == {"plain", "personal"}
    assert len(revealed["blind"]["mapping"]) == 8
    score = revealed["blind"]["score"]
    assert score["answered"] == 2 and score["tie"] == 1
    assert score["personal"] + score["plain"] == 1
    assert [image["id"] for image in revealed["plain"]] == [
        f"plain-{i}" for i in range(4)
    ]
    assert len(revealed["variants"][0]["images"]) == 4
    assert (
        revealed["variants"][0]["personalization"]["alpha"] == CONFIG["alphas"]["mid"]
    )


def test_blind_picks_are_validated_and_frozen_after_reveal(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    pairs = service.snapshot(sid)["run"]["blind"]["pairs"]
    other = pairs[1]["items"][0]["token"]
    with pytest.raises(ValueError):
        service.pick_blind(sid, 0, other)
    with pytest.raises(ValueError):
        service.pick_blind(sid, 0, "guess")
    with pytest.raises(ValueError):
        service.pick_blind(sid, 9, "tie")
    service.pick_blind(sid, 0, pairs[0]["items"][1]["token"])
    service.reveal(sid)
    with pytest.raises(Conflict):
        service.pick_blind(sid, 0, "tie")


def test_variants_are_limited_and_identical_settings_reuse_the_exact_cache(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid, "r0")
    settle(service)
    service.reveal(sid)
    start(service, sid, "r1", alpha="strong")
    settle(service)
    start(service, sid, "r2", alpha="mid")
    settle(service)
    run = service.snapshot(sid)["run"]
    assert [v["mode"] for v in run["variants"]] == ["live", "live", "exact-cache"]
    assert [v["id"] for v in run["variants"]] == ["v0", "v1", "v2"]
    assert run["variants"][2]["images"] == run["variants"][0]["images"]
    assert len(stage_stub) == 2
    assert len(run["blind"]["pairs"]) == 4
    with pytest.raises(Conflict):
        start(service, sid, "r3", alpha="weak")
    assert len(service.snapshot(sid)["run"]["variants"]) == CONFIG["max_variants"]


def test_variants_wait_for_the_reveal(tmp_path, assets, stage_stub):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid, "r0")
    settle(service)
    with pytest.raises(Conflict):
        start(service, sid, "r1", alpha="strong")
    service.reveal(sid)
    assert start(service, sid, "r1", alpha="strong")["run"]["id"]


def test_request_id_is_idempotent_and_rejects_reuse_with_new_input(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    first = start(service, sid, "same")["run"]
    settle(service)
    again = start(service, sid, "same")["run"]
    assert again["id"] == first["id"]
    assert len(again["variants"]) == 1
    assert len(stage_stub) == 1
    service.reveal(sid)
    with pytest.raises(Conflict):
        start(service, sid, "same", alpha="strong")


def test_results_with_a_stale_personalization_hash_are_dropped(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    run = service.snapshot(sid)["run"]
    assert not service.publish(
        sid, run["id"], "v0", "not-the-hash", {"status": "done", "error": "stale"}
    )
    assert service.snapshot(sid)["run"]["error"] is None


def blocking_runner(started, release):
    def runner(service, session, run_id, variant):
        started.set()
        release.wait(2)
        service.publish(
            session["id"],
            run_id,
            variant["id"],
            variant["personalization"]["hash"],
            {"status": "done", "error": "late"},
        )

    return runner


def test_cancelling_the_blind_run_returns_the_visitor_to_their_selection(
    tmp_path, assets
):
    started, release = threading.Event(), threading.Event()
    service = Service(tmp_path, runner=blocking_runner(started, release))
    sid = prepared(service)
    run_id = start(service, sid)["run"]["id"]
    assert started.wait(1)
    snapshot = service.cancel_run(sid)
    # Nothing was shown, so the run is discarded and the selection can change.
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


def test_cancelling_an_adjustment_keeps_the_revealed_run(tmp_path, assets, stage_stub):
    service = Service(tmp_path)
    sid = prepared(service)
    start(service, sid, "r0")
    settle(service)
    service.reveal(sid)
    started, release = threading.Event(), threading.Event()
    service.runner = blocking_runner(started, release)
    start(service, sid, "r1", alpha="strong")
    assert started.wait(1)
    snapshot = service.cancel_run(sid)
    assert snapshot["run"] is not None
    assert snapshot["run"]["status"] == "done"
    assert snapshot["run"]["variants"][1]["error"] == "cancelled"
    assert len(snapshot["run"]["variants"][0]["images"]) == 4
    release.set()
    settle(service)
    assert service.snapshot(sid)["run"]["variants"][1]["error"] == "cancelled"


def test_a_new_topic_after_the_reveal_starts_a_fresh_comparison(
    tmp_path, assets, stage_stub
):
    service = Service(tmp_path)
    sid = prepared(service)
    first = start(service, sid, "r0")["run"]["id"]
    settle(service)
    service.reveal(sid)
    run = start(service, sid, "r1", topic="tokyo")["run"]
    settle(service)
    assert run["id"] != first
    assert run["topic_id"] == "tokyo"
    assert not run["blind"]["revealed"]
    run = service.snapshot(sid)["run"]
    assert [v["id"] for v in run["variants"]] == ["v0"]
    assert [pair["ready"] for pair in run["blind"]["pairs"]] == [True] * 4


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
    assert not run["blind"]["revealed"]
    assert service.snapshot(sid)["run"]["variants"][0]["mode"] == "live"


def test_direct_image_paths_cannot_de_blind_a_pair(tmp_path, assets, stage_stub):
    service = Service(tmp_path)
    sid = prepared(service)
    run = start(service, sid)["run"]
    settle(service)
    run_id = run["id"]
    for name in (f"{run_id}/plain-0.png", f"{run_id}/v0/v0-0.png"):
        assert (tmp_path / "sessions" / sid / name).is_file()
        with pytest.raises(FileNotFoundError):
            service.artifact(sid, name)
    token = service.snapshot(sid)["run"]["blind"]["pairs"][0]["items"][0]["token"]
    assert service.artifact(sid, f"blind/{token}.png").is_file()
    revealed = service.reveal(sid)["run"]
    assert service.artifact(sid, revealed["plain"][0]["relative_path"]).is_file()
    assert service.artifact(
        sid, revealed["variants"][0]["images"][0]["relative_path"]
    ).is_file()


def test_reset_during_a_job_blocks_the_next_session_until_the_worker_exits(
    tmp_path, assets
):
    started = threading.Event()
    release = threading.Event()

    def runner(service, session, run_id, variant):
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
        service.artifact(session["id"], "blind/unknown.png")
    service.session["last_active"] = time.monotonic() - CONFIG["idle_seconds"] - 1
    service.expire_idle()
    with pytest.raises(KeyError):
        service.snapshot(session["id"])


def test_sample_starts_revealed_and_is_rejected_when_inconsistent(
    tmp_path, assets, asset_tree
):
    import json

    from exhibit.config import write_json

    service = Service(tmp_path)
    sid = service.create_session()["id"]
    samples = json.loads((asset_tree["root"] / "samples.json").read_text())
    run = service.sample(sid, samples[0]["id"])["run"]
    assert run["blind"]["revealed"] and run["variants"][0]["mode"] == "sample"
    assert len(run["plain"]) == 4 and len(run["variants"][0]["images"]) == 4
    assert run["blind"]["mapping"]
    samples[0]["images"][0]["seed"] = -999
    write_json(asset_tree["root"] / "samples.json", samples)
    service.reset(sid)
    sid = service.create_session()["id"]
    with pytest.raises(ValueError, match="inconsistent"):
        service.sample(sid, samples[0]["id"])
    assert service.snapshot(sid)["run"] is None
