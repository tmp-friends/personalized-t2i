"""Generate images with / without a user preference embedding.

  python scripts/04_generate.py --config configs/default.yaml --ckpt outputs/stage1_default/checkpoint-003000 \
      --prompts "a cat sitting on a windowsill" "a city street at night" \
      [--train-users 3]                 # rows = base + first N training users
      [--users <train uid> ...]          # rows = base + given training users
      [--user-files outputs/.../users/<uid>/user.safetensors ...]   # rows = base + stage-2 users
      [--seed 0 --steps 28 --guidance 3.5 --res 512 --delta-scale 1.0 --no-shared --no-distinct]
      --out outputs/gen/demo
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from premier.config import load_config  # noqa: E402
from premier.data.cache import FeatureCache, cache_texts  # noqa: E402
from premier.infer.sampler import generate, image_grid, to_pil  # noqa: E402
from premier.model.loading import TextEncoders, free_cuda, load_transformer, load_vae  # noqa: E402
from premier.model.premier import PremierModel  # noqa: E402
from premier.train.common import load_bank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--train-users", type=int, default=0)
    ap.add_argument("--users", nargs="*", default=[])
    ap.add_argument("--user-files", nargs="*", default=[])
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--no-shared", action="store_true")
    ap.add_argument("--no-distinct", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dev = cfg.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = a.steps or cfg.eval.steps
    guidance = a.guidance if a.guidance is not None else cfg.eval.guidance
    res = a.res or cfg.data.resolution

    cache = FeatureCache(cfg.paths.cache_dir, res, cfg.data.max_len)
    missing = [p for p in a.prompts if not cache.has_text(p)]
    if missing or not cache.empty_text_path.exists():
        enc = TextEncoders(cfg.model.aux_id, dev, max_len=cfg.data.max_len)
        cache_texts(cache, missing, enc)
        enc.unload()
    empty_t5 = cache.load_empty_text()[0][None].to(dev)

    ck = Path(a.ckpt)
    premier = PremierModel.load(ck, dev).eval()
    bank = load_bank(ck, dev)
    rows: list[tuple[str, torch.Tensor | None]] = [] if a.no_base else [("base (FLUX.1-dev)", None)]
    for u in bank.user_ids[: a.train_users]:
        rows.append((f"train user {u}", bank.emb[bank.index[u]].detach()[None]))
    for u in a.users:
        rows.append((f"train user {u}", bank.emb[bank.index[u]].detach()[None]))
    for f in a.user_files:
        meta = json.load(open(Path(f).parent / "meta.json"))
        rows.append((f"new user {meta['user_id']} ({meta['mode']}, k={meta['k_shot']})", load_file(f)["emb"].to(dev)))

    transformer = load_transformer(cfg.model.transformer_id, cfg.model.quant, dev)
    vae = load_vae(cfg.model.aux_id, dev)
    pils = []
    for label, ue in rows:
        for i, p in enumerate(a.prompts):
            t5, pooled = cache.load_text(p)
            img = generate(transformer, vae, t5[None], pooled[None], res, res, steps=steps, guidance=guidance,
                           seed=a.seed, premier=premier, user_emb=ue, empty_t5=empty_t5, use_shared=not a.no_shared,
                           use_distinct=not a.no_distinct, delta_scale=a.delta_scale)
            pil = to_pil(img)[0]
            pil.save(out / f"row{len(pils)//len(a.prompts):02d}_p{i:02d}.png")
            pils.append(pil)
            print(f"{label} | {p}")
    grid = image_grid(pils, cols=len(a.prompts))
    grid.save(out / "grid.jpg", quality=92)
    json.dump({"rows": [r[0] for r in rows], "cols": a.prompts, "seed": a.seed, "steps": steps, "guidance": guidance,
               "delta_scale": a.delta_scale}, open(out / "grid.json", "w"), indent=1)
    print("saved", out / "grid.jpg")


if __name__ == "__main__":
    main()
