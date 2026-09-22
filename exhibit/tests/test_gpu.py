import sys
import threading
import time

import pytest
from exhibit.gpu import GPUError, gpu_lease, run_process


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
                {"stage": "generate"},
                tmp_path,
                threading.Event(),
                time.monotonic() + 1,
                lambda e: None,
            )


def test_gpu_lease_is_the_shared_cancel_and_deadline_boundary(tmp_path, monkeypatch):
    from exhibit import gpu

    monkeypatch.setattr(gpu, "OUTPUTS", tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    with (
        pytest.raises(GPUError, match="cancelled"),
        gpu_lease(cancelled, time.monotonic() + 1),
    ):
        pytest.fail("cancelled lease entered")

    with (
        pytest.raises(GPUError, match="timeout"),
        gpu_lease(threading.Event(), time.monotonic() - 1),
    ):
        pytest.fail("expired lease entered")


def test_gpu_lease_releases_after_context_exit(tmp_path, monkeypatch):
    from exhibit import gpu

    monkeypatch.setattr(gpu, "OUTPUTS", tmp_path)
    cancel = threading.Event()
    deadline = time.monotonic() + 1
    with gpu_lease(cancel, deadline):
        pass
    with gpu_lease(cancel, deadline):
        pass


def test_evaluation_controller_competes_for_the_same_gpu_lock(tmp_path, monkeypatch):
    import fcntl
    import importlib.util

    from exhibit.config import FAN_POLICIES, ROOT
    from exhibit.fan_adapter import resolve_policy, thaw_policy

    from exhibit import gpu

    monkeypatch.setattr(gpu, "OUTPUTS", tmp_path)
    script = ROOT / "scripts/evaluate_fan.py"
    spec = importlib.util.spec_from_file_location("evaluate_fan_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "encoding.json"
    config.write_text(
        __import__("json").dumps(
            {
                "output": str(tmp_path / "report.json"),
                "prompt": "target",
                "refs": [{"text": "warm", "weight": 1.0}],
                "policies": [
                    {
                        "policy_id": "legacy_exhibit",
                        "effective_policy": thaw_policy(
                            resolve_policy("legacy_exhibit", FAN_POLICIES)
                        ),
                    }
                ],
            }
        )
    )

    with (tmp_path / "gpu.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(GPUError, match="busy"):
            module.run_encoding_controller(
                config,
                threading.Event(),
                time.monotonic() + 1,
                lambda event: None,
            )
