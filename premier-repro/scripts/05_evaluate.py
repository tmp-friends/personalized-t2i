"""Evaluate stage-2 users: generate base-model and Premier images for held-out preferred
prompts and compute CLIP T2I, LPIPS (vs. the preferred image), a SigLIP preference
proxy and, optionally, the ViPer proxy score / win rate.

  python scripts/05_evaluate.py --config configs/default.yaml --ckpt outputs/stage1_default/checkpoint-003000 \
      --stage2 outputs/stage1_default/stage2_linear_k8 [--viper] [--n-users 20] [--max-prompts 4]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from premier_repro.config import load_config  # noqa: E402
from premier_repro.data.cache import FeatureCache, cache_texts  # noqa: E402
from premier_repro.data.prefbench import load_manifest  # noqa: E402
from premier_repro.infer.sampler import generate, image_grid, to_pil  # noqa: E402
from premier_repro.model.loading import TextEncoders, free_cuda, load_transformer, load_vae  # noqa: E402
from premier_repro.model.premier import PremierModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--viper", action="store_true")
    ap.add_argument("--viper-4bit", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-generation", action="store_true", help="reuse images already generated")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dev = cfg.device
    stage2 = Path(a.stage2)
    out = Path(a.out) if a.out else stage2 / "eval"
    out.mkdir(parents=True, exist_ok=True)
    max_prompts = a.max_prompts or cfg.eval.max_eval_prompts
    seed = cfg.eval.seed
    res = cfg.data.resolution

    man = load_manifest(cfg.paths.manifest)
    image_root = Path(man["image_root"])
    user_dirs = sorted(p for p in (stage2 / "users").iterdir() if (p / "user.safetensors").exists())
    if a.n_users:
        user_dirs = user_dirs[: a.n_users]
    cache = FeatureCache(cfg.paths.cache_dir, res, cfg.data.max_len)

    # ---- eval items: held-out preferred pairs (not used for stage 2)
    items = []
    for ud in user_dirs:
        meta = json.load(open(ud / "meta.json"))
        uid = meta["user_id"]
        used = set(meta["train_images"])
        held = [p for p in man["users"][uid]["pos"] if p["image"] not in used][:max_prompts]
        for p in held:
            items.append({"user": uid, "user_dir": str(ud), "prompt": p["prompt"], "image": p["image"]})
    prompts = sorted({it["prompt"] for it in items})
    missing = [p for p in prompts if not cache.has_text(p)]
    if missing:
        enc = TextEncoders(cfg.model.aux_id, dev, max_len=cfg.data.max_len)
        cache_texts(cache, missing, enc)
        enc.unload()
    print(f"{len(user_dirs)} users, {len(items)} eval items")

    # ---- generation
    gen_dir = out / "images"
    gen_dir.mkdir(exist_ok=True)
    if not a.skip_generation:
        premier = PremierModel.load(Path(a.ckpt), dev).eval()
        transformer = load_transformer(cfg.model.transformer_id, cfg.model.quant, dev)
        vae = load_vae(cfg.model.aux_id, dev)
        empty_t5 = cache.load_empty_text()[0][None].to(dev)
        for i, it in enumerate(items):
            t5, pooled = cache.load_text(it["prompt"])
            ue = load_file(str(Path(it["user_dir"]) / "user.safetensors"))["emb"].to(dev)
            for tag, u in (("base", None), ("premier", ue)):
                path = gen_dir / f"{it['user']}_{i:04d}_{tag}.png"
                if path.exists():
                    continue
                img = generate(transformer, vae, t5[None], pooled[None], res, res, steps=cfg.eval.steps,
                               guidance=cfg.eval.guidance, seed=seed + i, premier=premier, user_emb=u, empty_t5=empty_t5)
                to_pil(img)[0].save(path)
            print(f"[{i+1}/{len(items)}] {it['user']} | {it['prompt'][:60]}")
        del transformer, vae, premier
        free_cuda()

    # ---- metrics
    from premier_repro.eval.metrics import ClipScorer, LpipsScorer, SiglipPrefScorer
    clip, lp, sig = ClipScorer(dev), LpipsScorer(dev), SiglipPrefScorer(dev)
    viper = None
    if a.viper:
        from premier_repro.eval.metrics import ViperProxy
        viper = ViperProxy(dev, load_in_4bit=a.viper_4bit)
    rows = []
    for i, it in enumerate(items):
        uid = it["user"]
        base = Image.open(gen_dir / f"{uid}_{i:04d}_base.png").convert("RGB")
        prem = Image.open(gen_dir / f"{uid}_{i:04d}_premier.png").convert("RGB")
        ref = Image.open(image_root / it["image"]).convert("RGB")
        meta = json.load(open(Path(it["user_dir"]) / "meta.json"))
        used = set(meta["train_images"])
        liked = [Image.open(image_root / p["image"]).convert("RGB") for p in man["users"][uid]["pos"] if p["image"] not in used]
        disliked = [Image.open(image_root / p["image"]).convert("RGB") for p in man["users"][uid]["neg"]]
        c_base, c_prem = clip.score([base, prem], [it["prompt"], it["prompt"]])
        s_base, s_prem = sig.score([base, prem], liked, disliked)
        row = dict(user=uid, idx=i, prompt=it["prompt"], clip_base=c_base, clip_premier=c_prem,
                   lpips_base=lp.score(base, ref), lpips_premier=lp.score(prem, ref),
                   siglip_base=s_base, siglip_premier=s_prem, lpips_ref_clip=clip.score([ref], [it["prompt"]])[0],
                   siglip_ref=sig.score([ref], liked, disliked)[0])
        if viper is not None:
            n = min(8, len(liked), len(disliked))
            row["viper_base"] = viper.score(base, liked[:n], disliked[:n])
            row["viper_premier"] = viper.score(prem, liked[:n], disliked[:n])
            row["viper_ref"] = viper.score(ref, liked[:n], disliked[:n])
        rows.append(row)
        print(f"[{i+1}/{len(items)}] {uid} clip {c_base:.3f}->{c_prem:.3f} lpips {row['lpips_base']:.3f}->{row['lpips_premier']:.3f} "
              f"siglip {s_base:.3f}->{s_prem:.3f}" + (f" viper {row['viper_base']:.3f}->{row['viper_premier']:.3f}" if viper else ""))

    with open(out / "per_item.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    summary["n_items"] = len(rows)
    summary["n_users"] = len(user_dirs)
    summary["siglip_win_rate"] = sum(r["siglip_premier"] > r["siglip_base"] for r in rows) / len(rows)
    summary["lpips_win_rate"] = sum(r["lpips_premier"] < r["lpips_base"] for r in rows) / len(rows)
    if viper is not None:
        summary["viper_win_rate"] = sum(r["viper_premier"] > r["viper_base"] for r in rows) / len(rows)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    # ---- grids per user: rows = preferred / base / premier, cols = prompts
    for ud in user_dirs:
        uid = json.load(open(ud / "meta.json"))["user_id"]
        its = [(i, it) for i, it in enumerate(items) if it["user"] == uid]
        if not its:
            continue
        pils = [Image.open(image_root / it["image"]).convert("RGB").resize((res, res)) for _, it in its]
        pils += [Image.open(gen_dir / f"{uid}_{i:04d}_base.png") for i, _ in its]
        pils += [Image.open(gen_dir / f"{uid}_{i:04d}_premier.png") for i, _ in its]
        image_grid(pils, cols=len(its)).save(out / f"grid_{uid}.jpg", quality=90)


if __name__ == "__main__":
    main()
