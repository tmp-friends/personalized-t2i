import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TailoredLayoutTests(unittest.TestCase):
    def test_package_imports_from_src(self) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        import tailored_visions_repro

        self.assertIsNotNone(tailored_visions_repro)

    def test_active_code_has_no_legacy_paths(self) -> None:
        files = list((ROOT / "scripts").glob("*.py"))
        files.extend((ROOT / "src").rglob("*.py") if (ROOT / "src").exists() else [])
        files.append(ROOT / "run_all.sh")
        obsolete = re.compile(r"\b(?:from|import) tv\b|results/|data/user_data")
        for path in files:
            self.assertIsNone(obsolete.search(path.read_text()), str(path.relative_to(ROOT)))

    def test_common_paths_exist(self) -> None:
        for relative in ["src/tailored_visions_repro", "docs/summary.html"]:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_downloader_defaults_to_ignored_raw_data(self) -> None:
        source = (ROOT / "scripts/00_download_data.py").read_text()
        self.assertIn('default="data/raw"', source)


if __name__ == "__main__":
    unittest.main()
