"""One lease across preparation and serving; release only after child exit."""

import fcntl
import json
import os
import signal
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import FAN_UPSTREAM, GPU_PYTHON, OUTPUTS, ROOT, write_json


class GPUError(RuntimeError):
    pass


@contextmanager
def gpu_lease(cancel, deadline):
    """Hold the process-wide GPU lock after honoring cancellation and timeout."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if cancel.is_set():
        raise GPUError("cancelled")
    if time.monotonic() >= deadline:
        raise GPUError("timeout")
    with (OUTPUTS / "gpu.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GPUError("GPU busy") from error
        try:
            if cancel.is_set():
                raise GPUError("cancelled")
            if time.monotonic() >= deadline:
                raise GPUError("timeout")
            yield
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def run_process(command, directory, cancel, deadline, on_event, *, env=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if cancel.is_set():
        raise GPUError("cancelled")
    if time.monotonic() >= deadline:
        raise GPUError("timeout")
    started = time.monotonic()
    with (
        (directory / "events.jsonl").open("w") as stdout,
        (directory / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env=env, start_new_session=True
        )
        try:
            with (directory / "events.jsonl").open() as events:
                while True:
                    for line in events:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict) and "type" in event:
                            on_event(event)
                    code = process.poll()
                    if code is not None:
                        # Drain writes between the previous read and process exit.
                        for line in events:
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(event, dict) and "type" in event:
                                on_event(event)
                        if code:
                            raise GPUError(f"worker_exit_{code}")
                        return {
                            "returncode": code,
                            "wall_seconds": round(time.monotonic() - started, 3),
                        }
                    if cancel.is_set():
                        raise GPUError("cancelled")
                    if time.monotonic() >= deadline:
                        raise GPUError("timeout")
                    time.sleep(0.05)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_stage(request, directory, cancel, deadline, on_event):
    with gpu_lease(cancel, deadline):
        directory = Path(directory) / (request["stage"] + "-" + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=True)
        request_file = directory / "request.json"
        write_json(request_file, request)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(FAN_UPSTREAM)]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        return run_process(
            [GPU_PYTHON, "-m", "exhibit.workers", str(request_file)],
            directory,
            cancel,
            deadline,
            on_event,
            env=env,
        )
