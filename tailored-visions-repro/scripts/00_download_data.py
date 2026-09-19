#!/usr/bin/env python
"""Stage 0: fetch and unpack the PIP dataset.

The archive is ~11 MB and expands to one JSONL per user (3,116 files, ~122 MB).

    python scripts/00_download_data.py

``result_url`` fields point at a CDN that has been offline since 2024, so the
images the paper's Image-Align metric needs cannot be downloaded; nothing here
tries to.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DRIVE_FILE_ID = "14VGi9ZQVxn7IX4iRWqLPtT6dhcnH7mUw"
EXPECTED_USERS = 3116


def download(dest_zip: Path) -> None:
    if dest_zip.exists():
        print(f"[download] {dest_zip} already present")
        return
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is required: pip install gdown")
    print(f"[download] fetching PIP dataset -> {dest_zip}")
    gdown.download(id=DRIVE_FILE_ID, output=str(dest_zip), quiet=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    user_dir = data_dir / "user_data"
    if user_dir.exists() and not args.force:
        n = len(list(user_dir.glob("*.jsonl")))
        print(f"[download] {user_dir} already has {n} user files; pass --force to redo")
        return 0
    if args.force and user_dir.exists():
        shutil.rmtree(user_dir)

    zip_path = data_dir / "PIP-dataset.zip"
    download(zip_path)

    print(f"[download] extracting -> {user_dir}")
    user_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(user_dir)
    # The archive may contain a single top-level folder; flatten it.
    entries = list(user_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        for path in entries[0].iterdir():
            shutil.move(str(path), user_dir / path.name)
        entries[0].rmdir()

    n = len(list(user_dir.glob("*.jsonl")))
    print(f"[download] {n} user files")
    if n != EXPECTED_USERS:
        print(f"  note: expected {EXPECTED_USERS} files (the paper reports 3115 users)")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tv.data import dataset_stats

    print(f"[download] {dataset_stats(user_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
