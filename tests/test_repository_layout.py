import configparser
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "fan-repro": "https://github.com/Burf/FAN.git",
    "premier-repro": "https://github.com/120L020904/Premier.git",
    "tailored-visions-repro": "https://github.com/zzjchen/Tailored-Visions.git",
}


class RepositoryLayoutTests(unittest.TestCase):
    def test_method_contract(self) -> None:
        required = [
            "README.md",
            "pyproject.toml",
            "uv.lock",
            "upstream",
            "patches/series",
            "scripts",
            "docs/summary.html",
        ]
        for method in METHODS:
            base = ROOT / method
            for relative in required:
                self.assertTrue((base / relative).exists(), f"{method}/{relative}")

    def test_submodule_urls_are_https(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(ROOT / ".gitmodules")
        urls = {
            section.split('"')[1]: parser[section]["url"]
            for section in parser.sections()
        }
        for path, url in METHODS.items():
            self.assertEqual(urls[f"{path}/upstream"], url)

    def test_submodules_are_clean(self) -> None:
        for method in METHODS:
            status = subprocess.run(
                ["git", "-C", str(ROOT / method / "upstream"), "status", "--short"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout
            self.assertEqual(status, "", method)


if __name__ == "__main__":
    unittest.main()
