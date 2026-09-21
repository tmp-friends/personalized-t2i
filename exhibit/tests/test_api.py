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


def test_run_comparison_over_http(client, stage_stub):
    sid = client.post("/api/sessions").json()["id"]
    client.put(f"/api/sessions/{sid}/selection", json={"cards": cards()})
    response = client.post(
        f"/api/sessions/{sid}/runs",
        json={
            "topic_id": "cat",
            "request_id": "run-1",
        },
    )
    assert response.status_code == 200
    for _ in range(500):
        snapshot = client.get(f"/api/sessions/{sid}").json()
        if snapshot["run"]["status"] == "done":
            break
    run = snapshot["run"]
    assert "blind" not in run and run["mode"] == "live"
    assert len(run["plain"]) == 4
    image = client.get(run["plain"][0]["url"])
    assert image.status_code == 200
    assert image.headers["cache-control"] == "no-store"
    assert "variants" not in run and len(run["personal"]) == 4
    assert client.get(run["personal"][0]["url"]).status_code == 200
    for gone in ("blind", "reveal"):
        assert client.post(f"/api/sessions/{sid}/{gone}").status_code in (404, 405)
    refs = run["personalization"]["refs"]
    assert {card for ref in refs for card in ref["card_ids"]} == set(IDS[:3])
    assert all(set(ref) == {"text", "weight", "aspect", "card_ids"} for ref in refs)
    # The adjust parameters are gone; extra fields are rejected like any other.
    assert (
        client.post(
            f"/api/sessions/{sid}/runs",
            json={"topic_id": "cat", "alpha": "strong", "request_id": "run-2"},
        ).status_code
        == 422
    )
    assert client.post(f"/api/sessions/{sid}/cancel").status_code == 200


def test_health_and_config_are_available_without_loading_gpu(client):
    health = client.get("/api/health").json()
    assert not health["gpu_busy"] and health["mode"] == "fan-live"
    config = client.get("/api/config").json()
    assert len(config["topics"]) == 6
    assert len(config["cards"]) == 16
    assert set(config["aspects"]) == {"color", "lighting", "texture", "mood"}
    assert config["selection"] == CONFIG["selection"]
    assert config["alpha"] == CONFIG["alpha"]
    assert "alphas" not in config and "max_variants" not in config
    assert len(config["samples"]) == 6
    assert config["ready"]


def test_config_only_offers_reviewed_cards(client, monkeypatch):
    monkeypatch.setattr(module, "reviewed_ids", lambda: {IDS[0]})
    config = client.get("/api/config").json()
    assert [card["id"] for card in config["cards"]] == [IDS[0]]
