"""Stage 1: train preference adapters + training-user embeddings.

  python scripts/02_train_stage1.py --config configs/default.yaml [key=value ...]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from premier.config import load_config  # noqa: E402
from premier.train.stage1 import train_stage1  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    train_stage1(load_config(a.config, a.overrides))
