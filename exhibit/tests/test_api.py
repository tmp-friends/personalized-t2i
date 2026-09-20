import pytest
from exhibit.config import CONFIG
from exhibit.domain import CARDS
from exhibit.service import Service
from fastapi.testclient import TestClient

from exhibit import app as module

IDS = list(CARDS)


@pytest.fixture
def client(tmp_path, monkeypatch, assets):
    monkeypatch.setattr(module, "service", Service(tmp_path))
    with TestClient(module.app) as test_client:
        yield test_client


def cards(count=3):
    return [{"card_id": IDS[i], "aspects_off": []} for i in range(count)]


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
    assert len(session["card_order"]) == 16
    assert client.post("/api/sessions").status_code == 409
    assert (
        client.put(
            f"/api/sessions/{sid}/selection", json={"cards": cards(2)}
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"/api/sessions/{sid}/selection",
            json={"cards": [{"card_id": "nope"}] + cards(2)},
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"/api/sessions/{sid}/selection", json={"cards": cards()}
        ).status_code
        == 200
    )
    assert client.get(f"/api/sessions/{sid}/images/run.json").status_code == 404
    assert client.post(f"/api/sessions/{sid}/touch").json() == {"ok": True}
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.post("/api/sessions").status_code == 200


def test_blind_run_reveal_and_adjustment_over_http(client, stage_stub):
    sid = client.post("/api/sessions").json()["id"]
    client.put(f"/api/sessions/{sid}/selection", json={"cards": cards()})
    response = client.post(
        f"/api/sessions/{sid}/runs",
        json={
            "topic_id": "cat",
            "alpha": "mid",
            "weights": {IDS[0]: "emphasis"},
            "request_id": "run-1",
        },
    )
    assert response.status_code == 200
    for _ in range(500):
        snapshot = client.get(f"/api/sessions/{sid}").json()
        if snapshot["run"]["status"] == "done":
            break
    run = snapshot["run"]
    assert run["plain"] == [] and run["blind"]["mapping"] is None
    first = run["blind"]["pairs"][0]["items"][0]
    image = client.get(first["url"])
    assert image.status_code == 200
    assert image.headers["cache-control"] == "no-store"
    assert (
        client.post(
            f"/api/sessions/{sid}/blind",
            json={"pair_index": 1, "pick": first["token"]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/sessions/{sid}/blind", json={"pair_index": 0, "pick": "tie"}
        ).status_code
        == 200
    )
    run_id = run["id"]
    for guess in (f"{run_id}/plain-0.png", f"{run_id}/v0/v0-0.png"):
        assert client.get(f"/api/sessions/{sid}/images/{guess}").status_code == 404
    revealed = client.post(f"/api/sessions/{sid}/reveal").json()["run"]
    assert revealed["blind"]["score"]["tie"] == 1
    assert len(revealed["plain"]) == 4
    assert client.get(revealed["plain"][0]["url"]).status_code == 200
    assert client.get(revealed["variants"][0]["images"][0]["url"]).status_code == 200
    assert (
        client.get(f"/api/sessions/{sid}/images/{run_id}/plain-0.png").status_code
        == 200
    )
    assert (
        client.post(
            f"/api/sessions/{sid}/blind", json={"pair_index": 0, "pick": "tie"}
        ).status_code
        == 409
    )
    adjusted = client.post(
        f"/api/sessions/{sid}/runs",
        json={
            "topic_id": "cat",
            "alpha": "strong",
            "weights": {IDS[1]: "exclude"},
            "request_id": "run-2",
        },
    )
    assert adjusted.status_code == 200
    for _ in range(500):
        snapshot = client.get(f"/api/sessions/{sid}").json()
        if snapshot["run"]["status"] == "done":
            break
    variants = snapshot["run"]["variants"]
    assert len(variants) == 2
    assert variants[1]["alpha_key"] == "strong"
    refs = variants[1]["personalization"]["refs"]
    assert {card for ref in refs for card in ref["card_ids"]} == {IDS[0], IDS[2]}
    assert all(set(ref) == {"text", "weight", "aspect", "card_ids"} for ref in refs)
    assert client.post(f"/api/sessions/{sid}/cancel").status_code == 200


def test_health_and_config_are_available_without_loading_gpu(client):
    health = client.get("/api/health").json()
    assert not health["gpu_busy"] and health["mode"] == "fan-live"
    config = client.get("/api/config").json()
    assert len(config["topics"]) == 6
    assert len(config["cards"]) == 16
    assert set(config["aspects"]) == {"color", "lighting", "texture", "mood"}
    assert config["selection"] == CONFIG["selection"]
    assert config["alphas"] == CONFIG["alphas"]
    assert config["max_variants"] == CONFIG["max_variants"]
    assert len(config["samples"]) == 6
    assert config["ready"]


def test_config_only_offers_reviewed_cards(client, monkeypatch):
    monkeypatch.setattr(module, "reviewed_ids", lambda: {IDS[0]})
    config = client.get("/api/config").json()
    assert [card["id"] for card in config["cards"]] == [IDS[0]]
