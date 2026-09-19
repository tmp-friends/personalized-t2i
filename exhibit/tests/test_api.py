from exhibit.service import Service
from fastapi.testclient import TestClient

from exhibit import app as module


def test_api_flow_rejects_cross_origin_and_invalid_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "service", Service(tmp_path))
    with TestClient(module.app) as client:
        assert client.get("/").status_code == 200
        assert (
            client.post(
                "/api/sessions", headers={"Origin": "https://evil.invalid"}
            ).status_code
            == 403
        )
        session = client.post("/api/sessions").json()
        sid = session["id"]
        assert client.post("/api/sessions").status_code == 409
        assert (
            client.post(
                f"/api/sessions/{sid}/choices",
                json={"pair_id": "p1", "chosen_id": "p2-a"},
            ).status_code
            == 422
        )
        assert client.get(f"/api/sessions/{sid}/images/context.json").status_code == 404
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
        assert client.get(f"/api/sessions/{sid}").status_code == 404
        assert client.post("/api/sessions").status_code == 200


def test_health_and_config_are_available_without_loading_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "service", Service(tmp_path))
    with TestClient(module.app) as client:
        health = client.get("/api/health").json()
        assert not health["gpu_busy"]
        config = client.get("/api/config").json()
        assert len(config["topics"]) == 6
        assert len(config["pairs"]) == 5
