"""Encode images (VAE latents) and prompts (T5 / CLIP) for the selected users.

  python scripts/01_cache_features.py --config configs/default.yaml [key=value ...]

Caches: preferred images + prompts of the first `data.n_train_users` training users,
preferred AND dispreferred images/prompts of the first `data.n_test_users` test users,
the preview prompts and the empty prompt.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from premier_repro.config import load_config  # noqa: E402
from premier_repro.data.cache import FeatureCache, cache_latents, cache_texts  # noqa: E402
from premier_repro.data.prefbench import load_manifest  # noqa: E402
from premier_repro.model.loading import TextEncoders, free_cuda, load_vae  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-missing", action="store_true", help="ignore images that are not extracted yet")
    ap.add_argument("--train-neg", action="store_true", help="also cache dispreferred images of training users")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    man = load_manifest(cfg.paths.manifest)
    image_root = Path(man["image_root"])
    train_users = man["train_users"][: cfg.data.n_train_users]
    test_users = man["test_users"][: cfg.data.n_test_users]
    pairs = []
    for u in train_users:
        pairs += man["users"][u]["pos"]
        if args.train_neg:
            pairs += man["users"][u]["neg"]
    for u in test_users:
        pairs += man["users"][u]["pos"] + man["users"][u]["neg"]
    images = sorted({p["image"] for p in pairs})
    prompts = sorted({p["prompt"] for p in pairs} | set(cfg.train.sample_prompts))
    present = [p for p in images if (image_root / p).exists()]
    print(f"{len(train_users)} train users, {len(test_users)} test users -> {len(images)} images "
          f"({len(present)} present on disk), {len(prompts)} prompts")
    if len(present) < len(images) and not args.skip_missing:
        raise SystemExit(f"{len(images) - len(present)} images missing; run scripts/00_prepare_prefbench.py --extract "
                         f"or pass --skip-missing")

    cache = FeatureCache(cfg.paths.cache_dir, cfg.data.resolution, cfg.data.max_len)
    vae = load_vae(cfg.model.aux_id, cfg.device)
    n = cache_latents(cache, image_root, present, vae, batch_size=8)
    print(f"latents: {n} new, {len(present)} total at {cfg.data.resolution}px")
    del vae
    free_cuda()
    enc = TextEncoders(cfg.model.aux_id, cfg.device, max_len=cfg.data.max_len)
    n = cache_texts(cache, prompts, enc, batch_size=16)
    print(f"texts: {n} new (max_len={cfg.data.max_len})")
    enc.unload()


if __name__ == "__main__":
    main()
