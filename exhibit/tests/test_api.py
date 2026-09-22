import pytest
from conftest import PROVENANCE
from exhibit.catalog import build_catalog
from exhibit.config import CONFIG
from exhibit.service import Service
from fastapi.testclient import TestClient

from exhibit import app as module

GAINS = {"color": 1, "lighting": 1, "texture": 1, "mood": 1}


@pytest.fixture
def client(tmp_path, monkeypatch, assets):
    monkeypatch.setattr(module, "service", Service(tmp_path, provenance=PROVENANCE))
    with TestClient(module.app) as test_client:
        yield test_client


def cards(snapshot, count=3, *, aspects=("color",), strength=1):
    return [
        {"card_id": card_id, "strength": strength, "aspects": list(aspects)}
        for card_id in snapshot["shown_ids"][:count]
    ]


def select(client, sid, entries, *, revision=0, commit=True, gains=None):
    return client.put(
        f"/api/sessions/{sid}/selection",
        json={
            "expected_revision": revision,
            "cards": entries,
            "aspect_gains": gains or GAINS,
            "commit": commit,
        },
    )


def prepared(client):
    session = client.post("/api/sessions").json()
    sid = session["id"]
    assert select(client, sid, cards(session)).status_code == 200
    return sid


def wait(client, sid):
    for _ in range(500):
        snapshot = client.get(f"/api/sessions/{sid}").json()
        if snapshot["run"] and snapshot["run"]["status"] == "done":
            return snapshot
    raise AssertionError("run never finished")


def test_api_flow_rejects_cross_origin_and_invalid_ids(client):
    assert client.get("/").status_code == 200
    assert (
        client.post(
            "/api/sessions", headers={"Origin": "https://evil.invalid"}
        ).status_code
        == 403
    )
    session = client.post("/api/sessions").json()
    sid = session["id"]
    assert session["revision"] == 0 and session["committed"] is False
    assert len(session["rounds"][0]["card_ids"]) == CONFIG["selection"]["round_size"]
    assert client.post("/api/sessions").status_code == 409
    assert select(client, sid, cards(session, 2)).status_code == 422
    assert (
        select(
            client, sid, [{"card_id": "nope", "strength": 1, "aspects": ["color"]}]
        ).status_code
        == 422
    )
    assert select(client, sid, cards(session)).status_code == 200
    assert client.get(f"/api/sessions/{sid}/images/run.json").status_code == 404
    assert client.post(f"/api/sessions/{sid}/touch").json() == {"ok": True}
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.post("/api/sessions").status_code == 200


def test_selection_payloads_are_validated_strictly(client):
    session = client.post("/api/sessions").json()
    sid = session["id"]
    unknown = cards(session)
    unknown[0]["note"] = "extra"
    assert select(client, sid, unknown).status_code == 422
    assert select(client, sid, cards(session, strength=True)).status_code == 422
    assert select(client, sid, cards(session, strength=1.0)).status_code == 422
    assert select(client, sid, cards(session, strength=3)).status_code == 422
    assert (
        select(client, sid, cards(session), gains={**GAINS, "color": True}).status_code
        == 422
    )
    assert (
        select(client, sid, cards(session), gains={**GAINS, "color": 3}).status_code
        == 422
    )
    assert select(client, sid, cards(session), gains={"color": 1}).status_code == 422
    assert (
        client.put(
            f"/api/sessions/{sid}/selection",
            json={
                "expected_revision": 0,
                "cards": cards(session),
                "aspect_gains": GAINS,
                "commit": True,
                "alpha": 0.9,
            },
        ).status_code
        == 422
    )
    # A card the session was never shown cannot be liked.
    shown = set(session["shown_ids"])
    config = client.get("/api/config").json()
    unseen = next(card["id"] for card in config["cards"] if card["id"] not in shown)
    assert (
        select(
            client, sid, [{"card_id": unseen, "strength": 1, "aspects": ["color"]}]
        ).status_code
        == 422
    )
    assert select(client, sid, cards(session), revision=5).status_code == 409
    assert client.get(f"/api/sessions/{sid}").json()["revision"] == 0


def test_rounds_are_replayed_and_limited(client):
    session = client.post("/api/sessions").json()
    sid = session["id"]
    first = client.post(
        f"/api/sessions/{sid}/rounds", json={"request_id": "q1", "expected_revision": 0}
    )
    assert first.status_code == 200 and len(first.json()["rounds"]) == 2
    replay = client.post(
        f"/api/sessions/{sid}/rounds", json={"request_id": "q1", "expected_revision": 0}
    )
    assert replay.status_code == 200 and replay.json() == first.json()
    assert (
        client.post(
            f"/api/sessions/{sid}/rounds",
            json={"request_id": "q1", "expected_revision": 1},
        ).status_code
        == 409
    )
    third = client.post(
        f"/api/sessions/{sid}/rounds", json={"request_id": "q2", "expected_revision": 0}
    )
    assert third.status_code == 200 and len(third.json()["rounds"]) == 3
    # Three rounds is the configured maximum, whatever is left unseen.
    assert (
        client.post(
            f"/api/sessions/{sid}/rounds",
            json={"request_id": "q3", "expected_revision": 0},
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/sessions/{sid}/rounds",
            json={"request_id": "q4", "expected_revision": 0, "round": 3},
        ).status_code
        == 422
    )


def test_run_comparison_over_http(client, stage_stub):
    sid = prepared(client)
    response = client.post(
        f"/api/sessions/{sid}/runs",
        json={"topic_id": "cat", "request_id": "run-1", "expected_revision": 1},
    )
    assert response.status_code == 200
    run = wait(client, sid)["run"]
    from exhibit.config import FAN_POLICIES

    assert "blind" not in run and run["mode"] == "live"
    assert run["preference_revision"] == 1
    assert run["policy_id"] == FAN_POLICIES["default_policy_id"] and run["policy_hash"]
    assert len(run["plain"]) == 4 and len(run["personal"]) == 4
    image = client.get(run["plain"][0]["url"])
    assert image.status_code == 200
    assert image.headers["cache-control"] == "no-store"
    assert client.get(run["personal"][0]["url"]).status_code == 200
    for gone in ("blind", "reveal"):
        assert client.post(f"/api/sessions/{sid}/{gone}").status_code in (404, 405)
    refs = run["personalization"]["refs"]
    assert {card for ref in refs for card in ref["card_ids"]} == {
        entry["card_id"]
        for entry in client.get(f"/api/sessions/{sid}").json()["selection"]
    }
    assert all(
        set(ref) == {"ref_id", "text", "weight", "card_ids", "aspects"} for ref in refs
    )
    # Encoder settings are server-owned; extra fields are rejected like any other.
    assert (
        client.post(
            f"/api/sessions/{sid}/runs",
            json={
                "topic_id": "cat",
                "alpha": "strong",
                "request_id": "run-2",
                "expected_revision": 1,
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/sessions/{sid}/runs",
            json={"topic_id": "cat", "request_id": "run-3", "expected_revision": 0},
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/sessions/{sid}/runs",
            json={"topic_id": "nope", "request_id": "run-4", "expected_revision": 1},
        ).status_code
        == 422
    )
    assert client.post(f"/api/sessions/{sid}/cancel").status_code == 200


def test_feedback_over_http(client, stage_stub):
    sid = prepared(client)
    client.post(
        f"/api/sessions/{sid}/runs",
        json={"topic_id": "cat", "request_id": "run-1", "expected_revision": 1},
    )
    run_id = wait(client, sid)["run"]["id"]
    url = f"/api/sessions/{sid}/runs/{run_id}/feedback"
    assert (
        client.put(
            url, json={"expected_revision": 1, "preference": "maybe"}
        ).status_code
        == 422
    )
    assert (
        client.put(url, json={"expected_revision": 0, "preference": "tie"}).status_code
        == 409
    )
    answered = client.put(url, json={"expected_revision": 1, "preference": "personal"})
    assert answered.status_code == 200
    assert answered.json()["run"]["feedback"]["preference"] == "personal"
    replaced = client.put(url, json={"expected_revision": 1, "preference": "tie"})
    assert replaced.json()["run"]["feedback"]["preference"] == "tie"
    assert client.get(f"/api/sessions/{sid}").json()["revision"] == 1


def test_health_and_config_are_available_without_loading_gpu(client):
    health = client.get("/api/health").json()
    assert not health["gpu_busy"] and health["mode"] == "fan-live"
    config = client.get("/api/config").json()
    assert config["schema_version"] == CONFIG["schema_version"]
    assert config["catalog_id"] == "catalog-v2" and config["catalog_hash"]
    assert len(config["topics"]) == 6
    assert len(config["cards"]) == 64
    assert set(config["cards"][0]["axis_levels"]) == set(config["aspects"])
    assert set(config["aspects"]) == {"color", "lighting", "texture", "mood"}
    assert config["selection"] == {
        "min": 3,
        "max": 10,
        "round_size": 12,
        "max_rounds": 3,
    }
    from exhibit.config import FAN_POLICIES

    # The displayed alpha comes from the resolved policy, not from the root config.
    assert config["policy"]["policy_id"] == FAN_POLICIES["default_policy_id"]
    assert config["alpha"] == config["policy"]["alpha"] == 0.5
    assert config["policy"]["profiling"] == {"mode": "all"}
    assert "alphas" not in config and "max_variants" not in config
    assert len(config["samples"]) == 6
    assert config["ready"]


def test_config_only_offers_reviewed_cards(client, asset_tree):
    import json

    review = json.loads(asset_tree["review"].read_text())
    kept = next(iter(review))
    for card_id in review:
        if card_id != kept:
            review[card_id] = {"reviewed": False, "note": ""}
    asset_tree["review"].write_text(json.dumps(review))
    config = client.get("/api/config").json()
    assert [card["id"] for card in config["cards"]] == [kept]
    assert not config["ready"]


def partly_reviewed_v2(root, review_path):
    """40 of 64 cards reviewed, leaning hard on `girl` and the lower levels."""
    from conftest import write_catalog_bundle
    from exhibit.config import ROOT, read_json
    from exhibit.domain import ASPECTS

    cards = build_catalog(read_json(ROOT / "configs/catalog-v2.json"))
    reviewed = set()
    for subject, quota in (
        ("girl", 16),
        ("student", 12),
        ("traveler", 8),
        ("barista", 4),
    ):
        rows = sorted(
            (card for card in cards if card["subject_id"] == subject),
            key=lambda card: (sum(card["axis_levels"][a] for a in ASPECTS), card["id"]),
        )
        reviewed.update(card["id"] for card in rows[:quota])
    write_catalog_bundle(root, review_path, reviewed=reviewed)
    return reviewed


def test_a_partly_reviewed_v2_catalog_still_opens_and_serves_rounds(
    client, assets, tmp_path, monkeypatch
):
    """Design §6.1 runs on the cards that passed, not on a complete 64."""
    from exhibit import catalog as catalog_module

    review = tmp_path / "partly-reviewed.json"
    reviewed = partly_reviewed_v2(assets, review)
    monkeypatch.setattr(catalog_module, "REVIEW", review)

    config = client.get("/api/config").json()
    assert config["catalog_id"] == "catalog-v2"
    assert len(config["cards"]) == len(reviewed) == 40
    assert config["ready"] is True

    session = client.post("/api/sessions").json()
    assert set(session["shown_ids"]) <= reviewed
    assert len(session["rounds"][0]["cards"]) == CONFIG["selection"]["round_size"]


def build_tech_script():
    import importlib.util

    from exhibit.config import ROOT

    path = ROOT / "scripts/build_tech.py"
    spec = importlib.util.spec_from_file_location("test_build_tech_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_tech_page_is_served_from_the_shipped_asset():
    from exhibit.config import ASSETS

    assert (ASSETS / "tech.html").is_file()
    response = TestClient(module.app).get("/tech")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "FAN の仕組み" in response.text
    assert "言わないこと" not in response.text


def test_tech_page_is_rebuilt_from_the_current_configs():
    """The shipped page is exactly what build_tech.py renders from the configs."""
    import re

    from exhibit.config import ASSETS, FAN_POLICIES

    script = build_tech_script()
    page = script.render()
    assert (ASSETS / "tech.html").read_text() == page, (
        "assets/tech.html is stale: run exhibit/scripts/build_tech.py"
    )
    generation = CONFIG["generation"]
    policy = FAN_POLICIES["policies"][FAN_POLICIES["default_policy_id"]]
    layer_coverage = [
        f"{n}層のうち {script.layer_range(script.pa_layers(n, policy))} 層目の"
        f"{len(script.pa_layers(n, policy))}層"
        for n in script.ENCODER_LAYERS.values()
    ]
    for value in (
        f"{generation['width']}×{generation['height']}",
        f"<dt>steps</dt><dd>{generation['steps']}</dd>",
        f"<dt>CFG</dt><dd>{generation['guidance_scale']}</dd>",
        f"<code>alpha</code> = {policy['alpha']}",
        f"<code>pooled_mode</code> = {policy['pooled_mode']}",
        *layer_coverage,
        CONFIG["fan"]["decoders"]["bigG.pth"][:12],
        "data:image/png;base64,",
        generation["revision"][:12],
        generation["vae"]["model"],
    ):
        assert value in page, value
    # Self-contained: fonts are inlined and nothing is loaded from the network.
    assert "<link" not in page.replace('<link rel="icon" href="data:', "")
    assert "@import" not in page
    assert not re.search(r'src="https?://', page)
    assert not re.search(r"url\((?!data:)", page)


def test_the_welcome_pair_is_one_sample_beside_the_plain_image_of_its_seed(
    client, assets, monkeypatch
):
    from exhibit import app as app_module
    from exhibit.config import read_json

    hero = client.get("/api/config").json()["hero"]
    sample_id, index = app_module.HERO
    root = assets
    sample = next(s for s in read_json(root / "samples.json") if s["id"] == sample_id)
    plain = read_json(root / "manifest.json")["images"][f"{sample['topic_id']}-{index}"]
    assert hero == {
        "sample_id": sample_id,
        "topic_id": sample["topic_id"],
        "plain_url": "/assets/" + plain["path"],
        "personal_url": "/assets/" + sample["images"][index]["path"],
    }
    assert plain["seed"] == sample["images"][index]["seed"]

    # A pair that cannot be matched is not offered; the page falls back on its own.
    monkeypatch.setattr(app_module, "HERO", ("no-such-sample", 0))
    assert client.get("/api/config").json()["hero"] is None
