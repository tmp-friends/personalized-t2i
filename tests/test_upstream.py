from __future__ import annotations

import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

from tools.upstream import PreparationError, prepare


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


class PrepareUpstreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.method = self.root / "method"
        self.method.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_upstream(self) -> str:
        upstream = self.method / "upstream"
        upstream.mkdir()
        git(upstream, "init", "-b", "main")
        git(upstream, "config", "user.email", "test@example.com")
        git(upstream, "config", "user.name", "Test")
        (upstream / "value.txt").write_text("upstream\n")
        git(upstream, "add", "value.txt")
        git(upstream, "commit", "-m", "initial")
        return git(upstream, "rev-parse", "HEAD")

    def test_missing_submodule_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(
            PreparationError, "git submodule update --init --recursive"
        ):
            prepare(self.method, "0" * 40)

    def test_wrong_sha_fails_before_creating_output(self) -> None:
        self.make_upstream()
        with self.assertRaisesRegex(PreparationError, "expected commit"):
            prepare(self.method, "0" * 40)
        self.assertFalse((self.method / ".work/upstream").exists())

    def write_patch(self, body: str) -> None:
        patches = self.method / "patches"
        patches.mkdir()
        (patches / "series").write_text("0001-change.patch\n")
        (patches / "0001-change.patch").write_text(body)

    def test_patch_is_applied_and_second_run_is_idempotent(self) -> None:
        sha = self.make_upstream()
        self.write_patch(
            "diff --git a/value.txt b/value.txt\n"
            "--- a/value.txt\n+++ b/value.txt\n"
            "@@ -1 +1 @@\n-upstream\n+patched\n"
        )
        first = prepare(self.method, sha)
        first_stat = first.stat().st_mtime_ns
        self.assertEqual((first / "value.txt").read_text(), "patched\n")
        second = prepare(self.method, sha)
        self.assertEqual(second, first)
        self.assertEqual(second.stat().st_mtime_ns, first_stat)

    def test_bad_patch_keeps_previous_materialization(self) -> None:
        sha = self.make_upstream()
        current = prepare(self.method, sha)
        (current / "sentinel.txt").write_text("keep\n")
        self.write_patch("not a patch\n")
        with self.assertRaises(subprocess.CalledProcessError):
            prepare(self.method, sha)
        self.assertEqual((current / "sentinel.txt").read_text(), "keep\n")

    def test_archive_extraction_emits_no_deprecation_warning(self) -> None:
        sha = self.make_upstream()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            output = prepare(self.method, sha)
        self.assertEqual((output / "value.txt").read_text(), "upstream\n")
