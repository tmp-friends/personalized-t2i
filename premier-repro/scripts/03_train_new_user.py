"""Stage 2: learn embeddings for new (test) users from k preferred images.

  python scripts/03_train_new_user.py --config configs/default.yaml --ckpt outputs/stage1_default/checkpoint-003000 \
        [--users u1 u2 | --n-users 20] [--k-shot 8] [--mode linear|direct] [--steps 300] [--out outputs/stage2_default]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from premier_repro.config import load_config  # noqa: E402
from premier_repro.data.cache import FeatureCache  # noqa: E402
from premier_repro.data.prefbench import load_manifest  # noqa: E402
from premier_repro.model.loading import load_transformer  # noqa: E402
from premier_repro.model.premier import PremierModel  # noqa: E402
from premier_repro.train.common import load_bank  # noqa: E402
from premier_repro.train.stage2 import train_new_user  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--users", nargs="*")
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--k-shot", type=int, default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    ck = Path(a.ckpt)
    out = Path(a.out) if a.out else ck.parent / f"stage2_{a.mode or cfg.stage2.mode}_k{a.k_shot or cfg.stage2.k_shot}"
    out.mkdir(parents=True, exist_ok=True)

    man = load_manifest(cfg.paths.manifest)
    users = a.users or man["test_users"][: (a.n_users or cfg.eval.n_users)]
    cache = FeatureCache(cfg.paths.cache_dir, cfg.data.resolution, cfg.data.max_len)
    premier = PremierModel.load(ck, cfg.device)
    bank = load_bank(ck, cfg.device)
    transformer = load_transformer(cfg.model.transformer_id, cfg.model.quant, cfg.device,
                                   gradient_checkpointing=cfg.model.gradient_checkpointing)
    print(f"{len(users)} users -> {out}")
    for i, u in enumerate(users):
        if a.skip_existing and (out / "users" / u / "user.safetensors").exists():
            continue
        p = train_new_user(cfg, u, man, cache, premier, bank, transformer, out, k_shot=a.k_shot, mode=a.mode,
                           steps=a.steps, seed=cfg.seed + i)
        print(f"[{i+1}/{len(users)}] {u}: saved {p}")


if __name__ == "__main__":
    main()
