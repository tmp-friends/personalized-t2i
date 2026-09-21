"""Exercise process death through the real lease, event reader and coordinator."""

import os
import sys
import time

import pytest
from exhibit.domain import CARDS
from exhibit.service import Service

from exhibit import gpu

IDS = list(CARDS)

WORKER = """
import hashlib
import json
import os
import resource
import signal
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
assert request["stage"] == "generate", request["stage"]
marker = Path(os.environ["RECOVERY_TEST_MARKER"])
for item in request["items"]:
    Path(item["path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(item["path"]).write_bytes(item["id"].encode())
    print(json.dumps({"type": "image", "id": item["id"], "path": item["path"],
                      "seed": item["seed"], "prompt": item["prompt"],
                      "sha256": hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(),
                      "personalization_hash": item["personalization"]["hash"]}), flush=True)
    if not marker.exists():
        marker.write_text(str(os.getpid()))
        if os.environ["RECOVERY_TEST_FAILURE"] == "kill":
            os.kill(os.getpid(), signal.SIGKILL)
        # This child alone is capped; do not exhaust host RAM or GPU memory.
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
        bytearray(256 * 1024**2)
"""


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [("memory", "worker_exit_1"), ("kill", "worker_exit_-9")],
)
def test_worker_death_preserves_image_and_next_session_can_generate(
    tmp_path, monkeypatch, assets, failure, expected_error
):
    # Only substitute the expensive model executable. IPC, subprocess lifecycle,
    # flock, publication, artifact access and session cleanup remain real.
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    marker = tmp_path / "failed-pid"
    real_run_process = gpu.run_process

    def model_process(command, *args, env=None, **kwargs):
        return real_run_process(
            [sys.executable, str(worker), command[-1]],
            *args,
            env={
                **env,
                "RECOVERY_TEST_MARKER": str(marker),
                "RECOVERY_TEST_FAILURE": failure,
            },
            **kwargs,
        )

    monkeypatch.setattr(gpu, "run_process", model_process)
    monkeypatch.setattr(gpu, "OUTPUTS", tmp_path)
    service = Service(tmp_path)

    def generate(request_id):
        session = service.create_session()
        sid = session["id"]
        service.set_selection(
            sid, [{"card_id": IDS[i], "aspects_off": []} for i in range(3)]
        )
        service.start_run(sid, "cat", "mid", {}, request_id)
        deadline = time.monotonic() + 15
        while service.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not service.busy, "Coordinator did not release the failed worker"
        return sid, service.reveal(sid)["run"]

    sid, run = generate("recovery")
    assert run["status"] == "done"
    assert run["error"] == expected_error
    assert len(run["plain"]) == 4
    variant = run["variants"][0]
    assert len(variant["images"]) == 1
    assert variant["error"] == expected_error
    # The partial result is still a usable blind pair; the rest never became ready.
    assert [pair["ready"] for pair in run["blind"]["pairs"]] == [
        True,
        False,
        False,
        False,
    ]
    completed = variant["images"][0]
    artifact = service.artifact(sid, completed["relative_path"])
    assert artifact.read_bytes() == b"v0-0"
    token = run["blind"]["pairs"][0]["items"][0]["token"]
    assert service.artifact(sid, f"blind/{token}.png").is_file()
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    if failure == "memory":
        assert any(
            "MemoryError" in log.read_text()
            for log in (tmp_path / "sessions" / sid).rglob("stderr.log")
        )

    service.reset(sid)
    assert not artifact.exists()
    with pytest.raises(KeyError):
        service.artifact(sid, completed["relative_path"])
    next_sid, next_run = generate("recovery-2")
    assert next_run["status"] == "done"
    assert next_run["error"] is None
    assert len(next_run["variants"][0]["images"]) == 4
    service.reset(next_sid)
