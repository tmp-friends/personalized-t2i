import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FanLayoutTests(unittest.TestCase):
    def test_common_paths_exist(self) -> None:
        for relative in [
            "upstream",
            "patches/series",
            "scripts/generate.py",
            "scripts/smoke_clip.py",
            "docs/summary.html",
        ]:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_upstream_is_clean(self) -> None:
        status = subprocess.run(
            ["git", "-C", str(ROOT / "upstream"), "status", "--short"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        self.assertEqual(status, "")

    def test_generate_cli_exposes_help(self) -> None:
        local_python = ROOT / ".venv/bin/python"
        python = str(local_python) if local_python.exists() else sys.executable
        subprocess.run(
            [python, str(ROOT / "scripts/generate.py"), "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


if __name__ == "__main__":
    unittest.main()
