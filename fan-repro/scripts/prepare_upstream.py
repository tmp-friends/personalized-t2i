from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from tools.upstream import prepare


EXPECTED_SHA = "9d0b76843f6437718195accac9cf3f050a25d26b"


if __name__ == "__main__":
    print(prepare(ROOT, EXPECTED_SHA))
