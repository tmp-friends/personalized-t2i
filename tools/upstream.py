from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO, Sequence


class PreparationError(RuntimeError):
    """Raised when an official upstream cannot be materialized safely."""


def _run(
    *args: str,
    cwd: Path | None = None,
    stdout: BinaryIO | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=stdout is None,
        stdout=stdout or subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _head(upstream: Path) -> str:
    if not (upstream / ".git").exists():
        raise PreparationError(
            f"{upstream} is not initialized; "
            "run git submodule update --init --recursive"
        )
    return _run("git", "-C", str(upstream), "rev-parse", "HEAD").stdout.strip()


def _patches(method_root: Path) -> list[Path]:
    series = method_root / "patches/series"
    if not series.exists():
        return []
    names = [line.strip() for line in series.read_text().splitlines()]
    return [
        series.parent / name
        for name in names
        if name and not name.startswith("#")
    ]


def _fingerprint(expected_sha: str, patches: list[Path]) -> str:
    digest = hashlib.sha256(expected_sha.encode())
    for patch in patches:
        digest.update(patch.name.encode())
        digest.update(patch.read_bytes())
    return digest.hexdigest()


def _extract(upstream: Path, sha: str, destination: Path) -> None:
    archive_path = destination.parent / "upstream.tar"
    with archive_path.open("wb") as archive:
        _run("git", "-C", str(upstream), "archive", sha, stdout=archive)
    try:
        with tarfile.open(archive_path) as archive:
            base = destination.resolve()
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if target != base and base not in target.parents:
                    raise PreparationError(f"unsafe archive path: {member.name}")
            extract_options = {"filter": "data"} if "filter" in inspect.signature(archive.extractall).parameters else {}
            archive.extractall(destination, **extract_options)
    finally:
        archive_path.unlink(missing_ok=True)


def _apply(destination: Path, patches: list[Path]) -> None:
    _run("git", "init", "-q", cwd=destination)
    try:
        for patch in patches:
            _run("git", "apply", "--check", str(patch), cwd=destination)
            _run("git", "apply", str(patch), cwd=destination)
    finally:
        shutil.rmtree(destination / ".git", ignore_errors=True)


def prepare(method_root: Path, expected_sha: str) -> Path:
    method_root = method_root.resolve()
    upstream = method_root / "upstream"
    actual_sha = _head(upstream)
    if actual_sha != expected_sha:
        raise PreparationError(
            f"expected commit {expected_sha}, found {actual_sha}"
        )

    patches = _patches(method_root)
    for patch in patches:
        if not patch.is_file():
            raise PreparationError(f"missing patch listed in series: {patch}")

    fingerprint = _fingerprint(expected_sha, patches)
    work = method_root / ".work"
    current = work / "upstream"
    stamp = current / ".prepared.json"
    if stamp.is_file() and json.loads(stamp.read_text()).get("fingerprint") == fingerprint:
        return current

    work.mkdir(parents=True, exist_ok=True)
    next_dir = Path(tempfile.mkdtemp(prefix="upstream-next-", dir=work))
    try:
        _extract(upstream, expected_sha, next_dir)
        _apply(next_dir, patches)
        (next_dir / ".prepared.json").write_text(
            json.dumps(
                {"commit": expected_sha, "fingerprint": fingerprint}, indent=2
            )
            + "\n"
        )
        previous = work / "upstream.previous"
        if previous.exists():
            shutil.rmtree(previous)
        if current.exists():
            os.replace(current, previous)
        os.replace(next_dir, current)
        shutil.rmtree(previous, ignore_errors=True)
    except Exception:
        shutil.rmtree(next_dir, ignore_errors=True)
        raise
    return current


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("method_root", type=Path)
    parser.add_argument("expected_sha")
    args = parser.parse_args(argv)
    print(prepare(args.method_root, args.expected_sha))
    return 0
