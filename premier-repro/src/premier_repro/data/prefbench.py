"""PrefBench (wenyii/PrefBench, DiffusionDB split) -> Premier manifest.

PrefBench JSON: list of records, each
  {id, image_file (target, preferred), text (its prompt),
   positive_img[], negative_img[], prompt_list[] (index aligned with pos/neg)}
Every user has 2 records that share (most of) the reference images, so we
merge them into one user entry with de-duplicated preferred / dispreferred
(image, prompt) pairs.

Manifest layout (data/raw/prefbench/manifest.json):
  {
    "source": "wenyii/PrefBench:diffusiondb",
    "image_root": "data/raw/prefbench/images",
    "train_users": [uid, ...], "test_users": [uid, ...],
    "users": { uid: {"split": "train"|"test",
                     "pos": [{"image": "diffusiondb/x.png", "prompt": "..."}],
                     "neg": [{"image": ..., "prompt": ...}],
                     "attributes": {"pos": str|None, "neg": str|None}} }
  }
"""
from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path


def merge_users(records: list[dict]) -> dict[str, dict]:
    users: dict[str, dict] = defaultdict(lambda: {"pos": {}, "neg": {}})
    for r in records:
        u = users[str(r["id"])]
        # target image is a preferred image with its own caption
        u["pos"].setdefault(r["image_file"], r["text"])
        for img, p in zip(r["positive_img"], r["prompt_list"]):
            u["pos"].setdefault(img, p)
        for img, p in zip(r["negative_img"], r["prompt_list"]):
            u["neg"].setdefault(img, p)
    out = {}
    for uid, u in users.items():
        out[uid] = {
            "pos": [{"image": k, "prompt": v} for k, v in u["pos"].items()],
            "neg": [{"image": k, "prompt": v} for k, v in u["neg"].items()],
        }
    return out


def load_attributes(pkl_path: Path) -> dict[str, tuple[str | None, str | None]]:
    """Return uid -> (neg_attributes, pos_attributes) from the PKL grouping."""
    if not pkl_path.exists():
        return {}
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    attrs = {}
    for uid, items in data.items():
        if items and isinstance(items[-1], list) and len(items[-1]) == 2:
            attrs[str(uid)] = (items[-1][0], items[-1][1])
    return attrs


def build_manifest(
    json_path: Path,
    out_path: Path,
    n_train: int,
    n_test: int,
    min_pos: int = 8,
    min_neg: int = 6,
    seed: int = 0,
    pkl_path: Path | None = None,
    image_root: str = "data/raw/prefbench/images",
) -> dict:
    records = json.load(open(json_path))
    users = merge_users(records)
    eligible = sorted(u for u, d in users.items() if len(d["pos"]) >= min_pos and len(d["neg"]) >= min_neg)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if len(eligible) < n_train + n_test:
        raise ValueError(f"only {len(eligible)} eligible users, need {n_train + n_test}")
    train = eligible[:n_train]
    test = eligible[n_train : n_train + n_test]
    attrs = load_attributes(pkl_path) if pkl_path else {}
    man_users = {}
    for split, uids in (("train", train), ("test", test)):
        for uid in uids:
            d = users[uid]
            # deterministic per-user ordering (so k-shot subsets are stable)
            urng = random.Random(f"{seed}-{uid}")
            pos = list(d["pos"]); urng.shuffle(pos)
            neg = list(d["neg"]); urng.shuffle(neg)
            a = attrs.get(uid, (None, None))
            man_users[uid] = {"split": split, "pos": pos, "neg": neg, "attributes": {"neg": a[0], "pos": a[1]}}
    manifest = {
        "source": f"wenyii/PrefBench:{json_path.stem}",
        "image_root": image_root,
        "seed": seed,
        "train_users": train,
        "test_users": test,
        "users": man_users,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out_path, "w"), indent=1)
    return manifest


def manifest_image_list(manifest: dict) -> list[str]:
    imgs = set()
    for u in manifest["users"].values():
        imgs.update(x["image"] for x in u["pos"])
        imgs.update(x["image"] for x in u["neg"])
    return sorted(imgs)


def load_manifest(path: str | Path) -> dict:
    return json.load(open(path))
