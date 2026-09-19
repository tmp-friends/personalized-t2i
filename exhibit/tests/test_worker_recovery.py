"""Exercise process death through the real lease, event reader and coordinator."""

import os
import sys
import time

import pytest
from exhibit.config import ASSETS
from exhibit.service import Service

from exhibit import gpu


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [("memory", "worker_exit_1"), ("kill", "worker_exit_-9")],
)
def test_worker_death_preserves_image_and_next_session_can_generate(
    tmp_path, monkeypatch, failure, expected_error
):
    # Only substitute the expensive model executable. IPC, subprocess lifecycle,
    # flock, publication, artifact access and session cleanup remain real.
    worker = tmp_path / "worker.py"
    worker.write_text("""
import hashlib
import json
import os
import resource
import shutil
import signal
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
if request["stage"] == "rewrite":
    print(json.dumps({"type": "rewrite", "valid": True,
                      "prompt": "One cat sitting by a window. Warm colors."}), flush=True)
elif request["stage"] == "generate":
    marker = Path(os.environ["RECOVERY_TEST_MARKER"])
    for item in request["items"]:
        shutil.copyfile(os.environ["RECOVERY_TEST_IMAGE"], item["path"])
        print(json.dumps({"type": "image", **item,
                          "sha256": hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(),
                          "context_hash": request["context_hash"]}), flush=True)
        if not marker.exists():
            marker.write_text(str(os.getpid()))
            if os.environ["RECOVERY_TEST_FAILURE"] == "kill":
                os.kill(os.getpid(), signal.SIGKILL)
            # This child alone is capped; do not exhaust host RAM or GPU memory.
            resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
            bytearray(256 * 1024**2)
else:
    raise AssertionError("Unexpected stage: " + request["stage"])
""")
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
                "RECOVERY_TEST_IMAGE": str(ASSETS / "generic/cat-0.png"),
            },
            **kwargs,
        )

    monkeypatch.setattr(gpu, "run_process", model_process)
    monkeypatch.setattr(gpu, "OUTPUTS", tmp_path)
    service = Service(tmp_path)

    def generate():
        session = service.create_session()
        for pair in session["pairs"]:
            service.answer(session["id"], pair["id"], pair["image_ids"][0])
        service.start_run(session["id"], "cat", {}, "recovery")
        deadline = time.monotonic() + 10
        while service.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not service.busy, "Coordinator did not release the failed worker"
        return session["id"], service.snapshot(session["id"])["run"]

    sid, run = generate()
    assert run["status"] == "done"
    assert run["error"] == expected_error
    assert len(run["generic"]) == 4
    assert len(run["personalized"]) == 1
    assert run["winner_id"] is None
    assert run["judgments"] == []
    completed = run["personalized"][0]
    artifact = service.artifact(sid, completed["relative_path"])
    assert artifact.read_bytes() == (ASSETS / "generic/cat-0.png").read_bytes()
    assert service.select(sid, completed["id"])["run"]["selected_id"] == completed["id"]
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
    next_sid, next_run = generate()
    assert next_run["status"] == "done"
    assert next_run.get("error") is None
    assert len(next_run["personalized"]) == 4
    assert next_run["winner_id"] is None
    service.reset(next_sid)
