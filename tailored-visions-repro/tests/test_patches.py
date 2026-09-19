import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "d0f4454ca08c68c5d30f08a01ff4a23a8b33b610"


class TailoredPatchTests(unittest.TestCase):
    def test_series_applies_to_expected_upstream(self) -> None:
        head = subprocess.run(
            ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        self.assertEqual(head, EXPECTED)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT / "upstream"), tmp],
                check=True,
            )
            subprocess.run(["git", "-C", tmp, "checkout", "--detach", EXPECTED], check=True)
            names = [
                line.strip()
                for line in (ROOT / "patches/series").read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
            for name in names:
                patch = ROOT / "patches" / name
                subprocess.run(["git", "-C", tmp, "apply", "--check", str(patch)], check=True)
                subprocess.run(["git", "-C", tmp, "apply", str(patch)], check=True)
            apiuse = (Path(tmp) / "apiuse.py").read_text()
            self.assertIn("from openai import OpenAI", apiuse)
            self.assertNotIn("openai.ChatCompletion.create", apiuse)


if __name__ == "__main__":
    unittest.main()
