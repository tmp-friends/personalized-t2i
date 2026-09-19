from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from tools.upstream import prepare

EXPECTED_SHA = "42473476a189b6b0127890a98e92b7edf49c0d59"


if __name__ == "__main__":
    print(prepare(ROOT, EXPECTED_SHA))
