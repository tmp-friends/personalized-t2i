import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
CONFIG = json.loads((ROOT / "configs/demo.json").read_text())
FAN_POLICIES = json.loads((ROOT / "configs/fan-policies.json").read_text())
ASSETS = ROOT / "assets"
OUTPUTS = Path(os.environ.get("EXHIBIT_OUTPUTS", str(ROOT / "outputs"))).resolve()
# The FAN encoder needs its own environment; the pinned path lives with the contract.
GPU_PYTHON = os.environ.get("EXHIBIT_GPU_PYTHON", str(REPO / CONFIG["fan"]["python"]))
FAN_UPSTREAM = REPO / CONFIG["fan"]["upstream"]


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
