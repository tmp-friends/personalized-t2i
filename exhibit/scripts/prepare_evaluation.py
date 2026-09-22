#!/usr/bin/env python3
"""Explicitly prepare and hash the frozen offline CLIP evaluator."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from exhibit.evaluation import load_evaluation_config, prepare_evaluator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT / "configs/fan-evaluation.json")
    )
    parser.add_argument("--output")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_evaluation_config(args.config, "encoding")
    output = Path(args.output or config["preparation_manifest"]).resolve()
    result = prepare_evaluator(
        config["evaluator"],
        output,
        local_files_only=args.local_files_only,
    )
    print(
        json.dumps(
            {
                "type": "evaluator_prepared",
                "path": str(output),
                "repo_id": result["repo_id"],
                "resolved_revision": result["resolved_revision"],
                "evaluator_hash": result["evaluator_hash"],
                "file_count": len(result["files"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
