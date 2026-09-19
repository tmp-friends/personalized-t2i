import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PremierLayoutTests(unittest.TestCase):
    def test_common_paths_exist(self) -> None:
        for relative in [
            "upstream",
            "patches/series",
            "src/premier_repro",
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

    def test_package_imports(self) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        import premier_repro

        self.assertIsNotNone(premier_repro)

    def test_official_clis_expose_help_without_loading_models(self) -> None:
        python = ROOT / ".venv/bin/python"
        self.assertTrue(python.exists(), "run uv sync --project premier-repro")
        for script in ["run_official.py", "train_official_user.py"]:
            result = subprocess.run(
                [str(python), str(ROOT / "scripts" / script), "--help"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
