from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from tools.upstream import prepare


EXPECTED_SHA = "d0f4454ca08c68c5d30f08a01ff4a23a8b33b610"


if __name__ == "__main__":
    print(prepare(ROOT, EXPECTED_SHA))
