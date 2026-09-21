#!/usr/bin/env python3
"""Run FAN evaluation workers under the same physical GPU lease as generation."""

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from exhibit.config import FAN_UPSTREAM, GPU_PYTHON, ROOT
from exhibit.evaluation_worker import validate_policy_specs
from exhibit.gpu import gpu_lease, run_process


def run_encoding_controller(config_path, cancel, deadline, on_event):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate_policy_specs(config.get("policies"))
    if not isinstance(config.get("output"), str) or not config["output"]:
        raise ValueError("output is required")
    output = Path(config["output"]).resolve()
    upstream = Path(config.get("upstream", str(FAN_UPSTREAM))).resolve()
    directory = output.parent / ("encoding-" + uuid.uuid4().hex[:8])
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(upstream)]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with gpu_lease(cancel, deadline):
        return run_process(
            [
                GPU_PYTHON,
                "-m",
                "exhibit.evaluation_worker",
                "encoding",
                "--config",
                str(config_path),
            ],
            directory,
            cancel,
            deadline,
            on_event,
            env=environment,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    encoding = subparsers.add_parser("encoding")
    encoding.add_argument("--config", required=True)
    encoding.add_argument("--timeout", type=float, default=1800)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cancel = threading.Event()
    if args.command == "encoding":
        result = run_encoding_controller(
            args.config,
            cancel,
            time.monotonic() + args.timeout,
            lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
        )
        print(json.dumps({"type": "controller", **result}), flush=True)


if __name__ == "__main__":
    main()
