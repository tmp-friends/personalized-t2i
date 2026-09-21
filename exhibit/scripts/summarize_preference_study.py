#!/usr/bin/env python3
"""Aggregate the blind preference study: participant first, then participants.

Win 1 / tie 0.5 / loss 0 is averaged inside one participant before anything is
averaged across participants, so a participant with many seeds stays one unit.
With no answers the report says 未実施 and no win rate is produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from exhibit import study as study_lib


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="the study directory")
    args = parser.parse_args(argv)

    directory = Path(args.study)
    if not (directory / "status.json").is_file():
        raise SystemExit(
            f"{directory} is not a study directory; run build_preference_study.py first"
        )
    participants_path = directory / "participants.json"
    participants = (
        json.loads(participants_path.read_text())
        if participants_path.is_file()
        else None
    )
    summary = study_lib.summarize_study(directory, participants=participants)
    study_lib.write_json(directory / "summary.json", summary)
    study_lib.write_text(directory / "summary.md", study_lib.summary_markdown(summary))
    print(
        json.dumps(
            {
                "type": "study_summary",
                "study_id": summary["study_id"],
                "study_kind": summary["study_kind"],
                "status": summary["status"],
                "participants_with_answers": summary["participants_with_answers"],
                "conclusions": {
                    name: item["conclusion"]
                    for name, item in summary["comparisons"].items()
                },
                "default_policy_change_eligible": summary["default_policy_change"][
                    "eligible"
                ],
                "summary": str(directory / "summary.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
