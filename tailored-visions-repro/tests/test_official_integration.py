import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OfficialIntegrationTests(unittest.TestCase):
    def test_outer_integration_paths_exist(self) -> None:
        for relative in [
            "scripts/prepare_upstream.py",
            "scripts/official_paths.py",
            "scripts/run_official.sh",
            "scripts/serve_local_llm.py",
            "docs/OFFICIAL_SETUP.md",
        ]:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_runner_help_is_local_and_nonzero(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/run_official.sh"), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
