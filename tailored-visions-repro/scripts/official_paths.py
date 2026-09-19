from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHED_UPSTREAM = ROOT / ".work/upstream"
DATA = ROOT / "data/raw/user_data"
OUTPUTS = ROOT / "outputs/official"


def require_upstream() -> Path:
    if not (PATCHED_UPSTREAM / "main.py").is_file():
        raise RuntimeError("run `python scripts/prepare_upstream.py` first")
    return PATCHED_UPSTREAM
