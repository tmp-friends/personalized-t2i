import copy
import json
import threading
import time
from pathlib import Path

import pytest
from conftest import PROVENANCE
from exhibit.config import CONFIG, write_json
from exhibit.elicitation import PreferenceError, RoundError
from exhibit.service import Conflict, Service

from exhibit import service as service_module

GAINS = {"color": 1, "lighting": 1, "texture": 1, "mood": 1}


def make(tmp_path, **kwargs):
    return Service(tmp_path, provenance=PROVENANCE, **kwargs)


def settle(service, seconds=5):
    deadline = time.monotonic() + seconds
    while service.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not service.busy, "Coordinator did not release the GPU lease"


def cards(snapshot, count=3, *, aspects=("color",), strength=1):
    return [
        {"card_id": card_id, "strength": strength, "aspects": list(aspects)}
        for card_id in snapshot["shown_ids"][:count]
    ]


def choose(service, sid, entries=None, *, revision=0, commit=True, gains=None):
    snapshot = service.snapshot(sid)
    return service.set_selection(
        sid,
        expected_revision=revision,
        cards=cards(snapshot) if entries is None else entries,
        aspect_gains=gains or GAINS,
        commit=commit,
    )


def prepared(service):
    sid = service.create_session()["id"]
    choose(service, sid)
    return sid


def start(service, sid, request_id="r0", topic="cat", revision=1):
    return service.start_run(sid, topic, request_id, revision)


def run_json(service, sid, run_id):
    path = Path(service.root) / "sessions" / sid / run_id / "personal" / "run.json"
    return json.loads(path.read_text())


# --------------------------------------------------------------------- state


def test_a_new_session_is_an_empty_draft_with_the_first_round(tmp_path, assets):
    service = make(tmp_path)
    snapshot = service.create_session()
    assert snapshot["revision"] == 0
    assert snapshot["catalog_id"] == CONFIG["catalog_id"]
    assert snapshot["committed"] is False
    assert snapshot["selection"] == []
    assert snapshot["aspect_gains"] == {key: 1.0 for key in GAINS}
    assert len(snapshot["rounds"]) == 1
    first = snapshot["rounds"][0]
    assert first["round_index"] == 0
    assert len(first["card_ids"]) == CONFIG["selection"]["round_size"]
    assert {pick["slot"] for pick in first["cards"]} == {"explore"}
    assert snapshot["shown_ids"] == first["card_ids"]
    assert snapshot["next_round_available"] and snapshot["run"] is None
    with pytest.raises(Conflict):
        service.create_session()


def test_selection_replaces_and_identical_content_keeps_the_revision(tmp_path, assets):
    service = make(tmp_path)
    sid = service.create_session()["id"]
    first = choose(service, sid, commit=False)
    assert first["revision"] == 1 and first["committed"] is False
    assert [entry["card_id"] for entry in first["selection"]] == sorted(
        entry["card_id"] for entry in first["selection"]
    )
    again = choose(service, sid, cards(first), revision=1, commit=False)
    assert again["revision"] == 1
    committed = choose(service, sid, cards(first), revision=1, commit=True)
    assert committed["revision"] == 2 and committed["committed"] is True
    # A replace, not a merge: two cards means two cards.
    replaced = choose(service, sid, cards(first, 2), revision=2, commit=False)
    assert len(replaced["selection"]) == 2 and replaced["revision"] == 3
    louder = choose(service, sid, cards(first, 2, strength=2), revision=3, commit=False)
    assert louder["revision"] == 4
    regained = choose(
        service,
        sid,
        cards(first, 2, strength=2),
        revision=4,
        commit=False,
        gains={**GAINS, "color": 2},
    )
    assert regained["revision"] == 5
    assert regained["aspect_gains"]["color"] == 2.0


def test_selection_refuses_stale_revisions_and_cards_that_were_never_shown(
    tmp_path, assets
):
    service = make(tmp_path)
    snapshot = service.create_session()
    sid = snapshot["id"]
    with pytest.raises(Conflict):
        choose(service, sid, revision=7)
    shown = set(snapshot["shown_ids"])
    unseen = next(
        card["id"]
        for card in service_module.active_catalog()["cards"]
        if card["id"] not in shown
    )
    with pytest.raises(PreferenceError):
        choose(
            service,
            sid,
            [{"card_id": unseen, "strength": 1, "aspects": ["color"]}],
            commit=False,
        )
    with pytest.raises(PreferenceError):
        choose(service, sid, cards(snapshot, strength=True))
    with pytest.raises(PreferenceError):
        choose(service, sid, gains={**GAINS, "color": 3})
    with pytest.raises(PreferenceError):
        choose(service, sid, cards(snapshot, aspects=()), commit=True)
    assert service.snapshot(sid)["revision"] == 0


def test_rounds_are_served_once_and_a_retry_repeats_nothing(tmp_path, assets):
    service = make(tmp_path)
    sid = service.create_session()["id"]
    first = service.snapshot(sid)["shown_ids"]
    snapshot = service.request_round(sid, "round-2", 0)
    second = snapshot["rounds"][1]
    assert len(snapshot["rounds"]) == 2
    assert not set(second["card_ids"]) & set(first)
    # Sixteen reviewed cards cannot fill a second round of twelve.
    assert second["shortfall_reason"] == "insufficient_unseen_cards"
    assert service.request_round(sid, "round-2", 0) == snapshot
    assert len(service.snapshot(sid)["rounds"]) == 2
    with pytest.raises(Conflict):
        service.request_round(sid, "round-2", 1)
    assert not service.snapshot(sid)["next_round_available"]
    with pytest.raises(RoundError):
        service.request_round(sid, "round-3", 0)


def test_no_more_than_three_rounds_are_served(tmp_path, assets, monkeypatch):
    monkeypatch.setitem(
        CONFIG, "selection", {"min": 3, "max": 10, "round_size": 4, "max_rounds": 3}
    )
    service = make(tmp_path)
    sid = service.create_session()["id"]
    service.request_round(sid, "r2", 0)
    snapshot = service.request_round(sid, "r3", 0)
    assert len(snapshot["rounds"]) == 3 and len(snapshot["shown_ids"]) == 12
    assert not snapshot["next_round_available"]
    with pytest.raises(RoundError):
        service.request_round(sid, "r4", 0)


def test_reset_drops_the_draft_round_history_and_request_ledger(tmp_path, assets):
    service = make(tmp_path)
    sid = prepared(service)
    service.request_round(sid, "round-2", 1)
    assert service.session["ledger"]
    service.reset(sid)
    with pytest.raises(KeyError):
        service.snapshot(sid)
    snapshot = service.create_session()
    assert snapshot["revision"] == 0 and snapshot["selection"] == []
    assert len(snapshot["rounds"]) == 1
    assert service.session["ledger"] == {}


# ---------------------------------------------------------------------- runs


def test_a_run_shows_the_plain_and_personalized_images_side_by_side(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    started = start(service, sid)["run"]
    # The plain images come from the generic cache, so they are there at once.
    assert [image["id"] for image in started["plain"]] == [
        f"plain-{i}" for i in range(4)
    ]
    assert started["preference_revision"] == 1
    settle(service)
    run = service.snapshot(sid)["run"]
    assert run["status"] == "done" and run["mode"] == "live"
    assert run["preference"]["selection"] == service.snapshot(sid)["selection"]
    assert run["policy_id"] == "legacy_exhibit"
    assert run["personalization_hash"] == run["personalization"]["hash"]
    assert run["personalization"]["effective_policy"]["alpha"] == 0.5
    assert "provenance" not in run["personalization"]
    assert [image["id"] for image in run["personal"]] == [
        f"personal-{i}" for i in range(4)
    ]
    assert all(
        image["policy_hash"] == run["policy_hash"]
        and image["personalization_hash"] == run["personalization_hash"]
        for image in run["personal"]
    )
    assert [image["seed"] for image in run["personal"]] == [
        image["seed"] for image in run["plain"]
    ]
    for image in run["plain"] + run["personal"]:
        assert image["url"] == f"/api/sessions/{sid}/images/{image['relative_path']}"
        assert service.artifact(sid, image["relative_path"]).is_file()
    # The immutable snapshot keeps the provenance the run was started with.
    stored = service.session["run"]["personalization"]["provenance"]
    assert stored["tokenizer_hash"] == PROVENANCE["tokenizer_hash"]


def test_a_run_needs_a_committed_selection_at_the_current_revision(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = service.create_session()["id"]
    choose(service, sid, commit=False)
    with pytest.raises(ValueError):
        start(service, sid)
    choose(service, sid, cards(service.snapshot(sid)), revision=1, commit=True)
    with pytest.raises(Conflict):
        start(service, sid, revision=1)
    with pytest.raises(ValueError):
        service.start_run(sid, "no-such-topic", "r1", 2)
    assert start(service, sid, "r2", revision=2)["run"]["status"] == "generating"
    settle(service)


def test_request_id_is_idempotent_and_rejects_reuse_with_new_input(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    first = start(service, sid, "same")["run"]
    again = start(service, sid, "same")["run"]
    assert again["id"] == first["id"]
    settle(service)
    assert start(service, sid, "same")["run"]["id"] == first["id"]
    assert len(stage_stub) == 1
    with pytest.raises(Conflict):
        start(service, sid, "same", topic="tokyo")
    with pytest.raises(Conflict):
        service.request_round(sid, "same", 1)


def test_a_used_request_id_cannot_revive_a_replaced_run(tmp_path, assets, stage_stub):
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid, "r0")
    settle(service)
    start(service, sid, "r1", topic="tokyo")
    settle(service)
    with pytest.raises(Conflict):
        start(service, sid, "r0")


def test_a_new_topic_starts_a_fresh_comparison(tmp_path, assets, stage_stub):
    service = make(tmp_path)
    sid = prepared(service)
    first = start(service, sid, "r0")["run"]["id"]
    settle(service)
    run = start(service, sid, "r1", topic="tokyo")["run"]
    settle(service)
    assert run["id"] != first and run["topic_id"] == "tokyo"
    assert len(service.snapshot(sid)["run"]["personal"]) == 4


def test_a_run_never_starts_without_computable_provenance(
    tmp_path, assets, stage_stub, monkeypatch
):
    def broken():
        raise ValueError("pinned runtime file is not cached")

    monkeypatch.setattr(service_module, "web_provenance", broken)
    service = Service(tmp_path)
    sid = prepared(service)
    with pytest.raises(ValueError, match="由来"):
        start(service, sid)
    assert service.snapshot(sid)["run"] is None and not stage_stub


# ------------------------------------------------------- revisions and events


def test_changing_the_selection_is_refused_until_the_worker_has_exited(
    tmp_path, assets
):
    started, release = threading.Event(), threading.Event()
    service = make(tmp_path, runner=blocking_runner(started, release))
    sid = prepared(service)
    run_id = start(service, sid)["run"]["id"]
    assert started.wait(1)
    with pytest.raises(Conflict):
        choose(service, sid, revision=1, commit=False)
    with pytest.raises(Conflict):
        service.request_round(sid, "round-2", 1)
    with pytest.raises(Conflict):
        service.set_feedback(sid, run_id, 1, "plain")
    snapshot = service.cancel_run(sid)
    assert snapshot["run"] is None
    # The run object is gone but its worker is still exiting.
    assert service.busy
    with pytest.raises(Conflict):
        choose(service, sid, revision=1, commit=False)
    release.set()
    settle(service)
    assert (tmp_path / "sessions" / sid / run_id).exists()
    assert choose(service, sid, revision=1, commit=False)["revision"] == 2


def test_reselecting_after_a_finished_run_detaches_it(tmp_path, assets, stage_stub):
    service = make(tmp_path)
    sid = prepared(service)
    run = start(service, sid)["run"]
    settle(service)
    images = [
        Path(service.artifact(sid, image["relative_path"]))
        for image in service.snapshot(sid)["run"]["personal"]
    ]
    snapshot = choose(
        service,
        sid,
        cards(service.snapshot(sid), aspects=("color", "mood")),
        revision=1,
    )
    assert snapshot["revision"] == 2 and snapshot["run"] is None
    # Old images stay on disk as this session's cache.
    assert all(path.is_file() for path in images)
    fresh = start(service, sid, "r1", revision=2)["run"]
    settle(service)
    assert fresh["id"] != run["id"]
    assert service.snapshot(sid)["run"]["preference_revision"] == 2


def test_a_late_worker_event_cannot_publish_into_the_new_run(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    stale = copy.deepcopy(service.session["run"])
    choose(service, sid, cards(service.snapshot(sid), aspects=("mood",)), revision=1)
    start(service, sid, "r1", revision=2)
    settle(service)
    current = copy.deepcopy(service.session["run"])
    late = {"status": "done", "error": "late", "message": "stale"}
    assert not service.publish(
        sid,
        stale["id"],
        stale["preference_revision"],
        stale["personalization"]["hash"],
        late,
    )
    assert not service.publish(
        sid,
        current["id"],
        stale["preference_revision"],
        current["personalization"]["hash"],
        late,
    )
    assert not service.publish(
        sid,
        current["id"],
        current["preference_revision"],
        stale["personalization"]["hash"],
        late,
    )
    run = service.snapshot(sid)["run"]
    assert run["error"] is None and run["message"] != "stale"
    assert service.publish(
        sid,
        current["id"],
        current["preference_revision"],
        current["personalization"]["hash"],
        {"message": "ok"},
    )


def test_an_image_event_from_another_policy_aborts_the_run(
    tmp_path, assets, monkeypatch
):
    def stage(request, directory, cancel, deadline, on_event):
        for item in request["items"]:
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["id"].encode())
            personalization = item["personalization"]
            on_event(
                {
                    "type": "image",
                    "id": item["id"],
                    "path": item["path"],
                    "sha256": "0" * 64,
                    "seed": item["seed"],
                    "prompt": item["prompt"],
                    "policy_id": "official_encoder",
                    "policy_hash": personalization["policy_hash"],
                    "effective_policy": personalization["effective_policy"],
                    "personalization_hash": personalization["hash"],
                }
            )
        return {"returncode": 0, "wall_seconds": 0.01}

    monkeypatch.setattr(service_module, "run_stage", stage)
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    run = service.snapshot(sid)["run"]
    assert run["status"] == "done" and run["personal"] == []
    assert "Personalization mismatch" in run["error"]


def test_an_image_event_without_the_new_fields_aborts_the_run(
    tmp_path, assets, monkeypatch
):
    def stage(request, directory, cancel, deadline, on_event):
        for item in request["items"]:
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["id"].encode())
            on_event(
                {
                    "type": "image",
                    "id": item["id"],
                    "path": item["path"],
                    "sha256": "0" * 64,
                    "seed": item["seed"],
                    "prompt": item["prompt"],
                    "personalization_hash": item["personalization"]["hash"],
                }
            )
        return {"returncode": 0, "wall_seconds": 0.01}

    monkeypatch.setattr(service_module, "run_stage", stage)
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    assert "Personalization mismatch" in service.snapshot(sid)["run"]["error"]


# --------------------------------------------------------------------- cache


def test_a_new_revision_with_the_same_content_still_hits_the_cache(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    original = cards(service.snapshot(sid))
    start(service, sid, "r0")
    settle(service)
    images = service.snapshot(sid)["run"]["personal"]
    choose(service, sid, cards(service.snapshot(sid), aspects=("mood",)), revision=1)
    snapshot = choose(service, sid, original, revision=2)
    # The revision moved on, the content did not.
    assert snapshot["revision"] == 3
    start(service, sid, "r1", revision=3)
    settle(service)
    run = service.snapshot(sid)["run"]
    assert run["mode"] == "exact-cache" and run["preference_revision"] == 3
    assert run["personal"] == images
    assert len(stage_stub) == 1


def test_returning_to_the_same_content_reuses_the_exact_cache(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid, "r0")
    settle(service)
    images = service.snapshot(sid)["run"]["personal"]
    start(service, sid, "r1", topic="tokyo")
    settle(service)
    start(service, sid, "r2")
    settle(service)
    run = service.snapshot(sid)["run"]
    assert run["mode"] == "exact-cache" and run["status"] == "done"
    assert run["personal"] == images
    assert len(stage_stub) == 2


def test_another_policy_or_catalog_misses_the_cache_and_the_old_content_hits(
    tmp_path, assets, asset_tree, stage_stub, monkeypatch
):
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid, "r0")
    settle(service)
    registered = service_module.FAN_POLICIES
    policies = copy.deepcopy(registered)
    policies["policies"]["legacy_exhibit"]["alpha"] = 0.4
    monkeypatch.setattr(service_module, "FAN_POLICIES", policies)
    start(service, sid, "r1")
    settle(service)
    assert service.snapshot(sid)["run"]["mode"] == "live"
    assert len(stage_stub) == 2
    monkeypatch.setattr(service_module, "FAN_POLICIES", registered)

    root = asset_tree["root"]
    manifest = json.loads((root / "manifest.json").read_text())
    selected = {entry["card_id"] for entry in service.snapshot(sid)["selection"]}
    card_id = next(
        key
        for key, image in manifest["images"].items()
        if key not in selected and image["path"].startswith("cards/")
    )
    original = (root / manifest["images"][card_id]["path"]).read_bytes()
    original_sha = manifest["images"][card_id]["sha256"]
    (root / manifest["images"][card_id]["path"]).write_bytes(original + b" revised")
    manifest["images"][card_id]["sha256"] = service_module.file_hash(
        root / manifest["images"][card_id]["path"]
    )
    write_json(root / "manifest.json", manifest)
    start(service, sid, "r2")
    settle(service)
    assert service.snapshot(sid)["run"]["mode"] == "live"
    assert len(stage_stub) == 3

    (root / manifest["images"][card_id]["path"]).write_bytes(original)
    manifest["images"][card_id]["sha256"] = original_sha
    write_json(root / "manifest.json", manifest)
    start(service, sid, "r3")
    settle(service)
    assert service.snapshot(sid)["run"]["mode"] == "exact-cache"
    assert len(stage_stub) == 3


def test_a_cache_entry_without_the_policy_fields_is_not_reused(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid, "r0")
    settle(service)
    service.session["cache"] = {
        key: {"images": entry["images"], "timings": entry["timings"]}
        for key, entry in service.session["cache"].items()
    }
    start(service, sid, "r1")
    settle(service)
    assert service.snapshot(sid)["run"]["mode"] == "live"
    assert len(stage_stub) == 2


# ------------------------------------------------------------------ feedback


def test_feedback_answers_only_the_current_finished_run_once(
    tmp_path, assets, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    run_id = start(service, sid)["run"]["id"]
    settle(service)
    # The record on disk is written before `done` is published, yet must say so.
    finished = run_json(service, sid, run_id)
    assert finished["status"] == "done" and finished["error"] is None
    assert len(finished["personal"]) == 4
    with pytest.raises(ValueError):
        service.set_feedback(sid, run_id, 1, "best")
    with pytest.raises(Conflict):
        service.set_feedback(sid, run_id, 0, "personal")
    snapshot = service.set_feedback(sid, run_id, 1, "personal")
    assert snapshot["run"]["feedback"]["preference"] == "personal"
    repeated = service.set_feedback(sid, run_id, 1, "personal")
    assert repeated["run"]["feedback"] == snapshot["run"]["feedback"]
    replaced = service.set_feedback(sid, run_id, 1, "tie")
    assert replaced["run"]["feedback"]["preference"] == "tie"
    stored = run_json(service, sid, run_id)
    assert stored["feedback"]["preference"] == "tie"
    assert stored["feedback"]["preference_revision"] == 1
    # The answer never touches the preference itself.
    assert service.snapshot(sid)["revision"] == 1
    assert service.snapshot(sid)["selection"] == snapshot["selection"]
    with pytest.raises(Conflict):
        service.set_feedback(sid, "another-run", 1, "tie")


def test_feedback_for_another_revision_is_refused(tmp_path, assets, stage_stub):
    service = make(tmp_path)
    sid = prepared(service)
    run_id = start(service, sid)["run"]["id"]
    settle(service)
    choose(service, sid, cards(service.snapshot(sid), aspects=("mood",)), revision=1)
    start(service, sid, "r1", revision=2)
    settle(service)
    new_run = service.snapshot(sid)["run"]["id"]
    with pytest.raises(Conflict):
        service.set_feedback(sid, new_run, 1, "plain")
    with pytest.raises(Conflict):
        service.set_feedback(sid, run_id, 2, "plain")
    assert service.set_feedback(sid, new_run, 2, "plain")["run"]["feedback"]


# ------------------------------------------------------------------- samples


def test_a_sample_never_becomes_the_visitor_preference(
    tmp_path, assets, asset_tree, stage_stub
):
    service = make(tmp_path)
    sid = prepared(service)
    before = service.snapshot(sid)
    sample = json.loads((asset_tree["root"] / "samples.json").read_text())[0]
    snapshot = service.sample(sid, sample["id"])
    run = snapshot["run"]
    assert run["mode"] == "sample" and run["preference_revision"] is None
    assert run["preference"]["source"] == "sample"
    assert len(run["plain"]) == 4 and len(run["personal"]) == 4
    assert snapshot["revision"] == before["revision"]
    assert snapshot["selection"] == before["selection"]
    assert snapshot["rounds"] == before["rounds"]
    with pytest.raises(Conflict):
        service.set_feedback(sid, run["id"], snapshot["revision"], "personal")
    live = start(service, sid, "r0")["run"]
    settle(service)
    assert live["id"] != run["id"] and live["mode"] == "live"
    assert len(service.snapshot(sid)["run"]["personal"]) == 4


def test_a_sample_is_rejected_while_generating_and_when_inconsistent(
    tmp_path, assets, asset_tree
):
    started, release = threading.Event(), threading.Event()
    service = make(tmp_path, runner=blocking_runner(started, release))
    sid = prepared(service)
    samples = json.loads((asset_tree["root"] / "samples.json").read_text())
    start(service, sid)
    assert started.wait(1)
    with pytest.raises(Conflict):
        service.sample(sid, samples[0]["id"])
    release.set()
    settle(service)
    samples[0]["images"][0]["seed"] = -999
    write_json(asset_tree["root"] / "samples.json", samples)
    with pytest.raises(ValueError, match="inconsistent"):
        service.sample(sid, samples[0]["id"])


# ----------------------------------------------------------------- lifecycle


def blocking_runner(started, release):
    def runner(service, session):
        started.set()
        release.wait(2)
        run = session["run"]
        service.publish(
            session["id"],
            run["id"],
            run["preference_revision"],
            run["personalization"]["hash"],
            {"status": "done", "error": "late"},
        )

    return runner


def test_cancelling_a_run_returns_the_visitor_to_their_selection(tmp_path, assets):
    started, release = threading.Event(), threading.Event()
    service = make(tmp_path, runner=blocking_runner(started, release))
    sid = prepared(service)
    run_id = start(service, sid)["run"]["id"]
    assert started.wait(1)
    snapshot = service.cancel_run(sid)
    # The comparison never completed, so the run is discarded.
    assert snapshot["run"] is None and snapshot["selection"]
    # The still-exiting worker keeps its directory until the session ends.
    assert (tmp_path / "sessions" / sid / run_id).exists()
    release.set()
    settle(service)
    assert service.snapshot(sid)["run"] is None
    service.reset(sid)
    assert not (tmp_path / "sessions" / sid).exists()


def test_cancelling_a_finished_run_keeps_it(tmp_path, assets, stage_stub):
    service = make(tmp_path)
    sid = prepared(service)
    start(service, sid)
    settle(service)
    run = service.cancel_run(sid)["run"]
    assert run["status"] == "done" and len(run["personal"]) == 4


def test_reset_during_a_job_blocks_the_next_session_until_the_worker_exits(
    tmp_path, assets
):
    started, release = threading.Event(), threading.Event()

    def runner(service, session):
        started.set()
        release.wait(2)

    service = make(tmp_path, runner=runner)
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
    service = make(tmp_path)
    session = service.create_session()
    with pytest.raises((ValueError, FileNotFoundError)):
        service.artifact(session["id"], "../../secret")
    with pytest.raises(FileNotFoundError):
        service.artifact(session["id"], "unknown/plain-0.png")
    service.session["last_active"] = time.monotonic() - CONFIG["idle_seconds"] - 1
    service.expire_idle()
    with pytest.raises(KeyError):
        service.snapshot(session["id"])
