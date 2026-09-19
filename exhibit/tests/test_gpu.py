import sys
import threading
import time

import pytest
from exhibit.gpu import GPUError, run_process


def test_progress_is_received_before_exit_and_child_is_reaped(tmp_path):
    events = []
    r = run_process(
        [
            sys.executable,
            "-c",
            'import json; print(json.dumps({"type":"image","id":"a"}), flush=True)',
        ],
        tmp_path,
        threading.Event(),
        time.monotonic() + 5,
        events.append,
    )
    assert events == [{"type": "image", "id": "a"}]
    assert r["returncode"] == 0


def test_timeout_kills_worker_before_return(tmp_path):
    t = time.monotonic()
    with pytest.raises(GPUError, match="timeout"):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path,
            threading.Event(),
            t + 0.2,
            lambda e: None,
        )
    assert time.monotonic() - t < 3


def test_cancelled_job_never_starts(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GPUError, match="cancel"):
        run_process(
            [sys.executable, "-c", "raise RuntimeError()"],
            tmp_path,
            cancel,
            time.monotonic() + 3,
            lambda e: None,
        )


def test_gpu_lease_rejects_concurrent_processes(tmp_path, monkeypatch):
    import fcntl

    from exhibit import gpu

    monkeypatch.setattr(gpu, "OUTPUTS", tmp_path)
    with (tmp_path / "gpu.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(GPUError, match="busy"):
            gpu.run_stage(
                {"stage": "rewrite"},
                tmp_path,
                threading.Event(),
                time.monotonic() + 1,
                lambda e: None,
            )
