"""Build the PrefBench manifest and (optionally) stream-extract the needed images.

Usage:
  python scripts/00_prepare_prefbench.py --n-train 1000 --n-test 100 [--extract]

The HF dataset ships images as concatenated tar parts (24 GB for diffusiondb).
tar has no index, so we stream the whole archive once and only keep the
members listed in the manifest (a few thousand PNGs).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from premier_repro.data.prefbench import build_manifest, manifest_image_list  # noqa: E402

HF = "https://huggingface.co/datasets/wenyii/PrefBench/resolve/main"
PARTS = {"diffusiondb": 6, "coco": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="diffusiondb", choices=list(PARTS))
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--min-neg", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "prefbench"))
    ap.add_argument("--extract", action="store_true", help="stream tar parts and extract manifest images")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    json_path = data_dir / "json" / f"{args.split}.json"
    if not json_path.exists():
        json_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "--retry", "5", "-o", str(json_path), f"{HF}/json/{args.split}.json"], check=True)
    pkl_path = data_dir / f"{args.split}.pkl"
    manifest_path = data_dir / "manifest.json"
    man = build_manifest(
        json_path, manifest_path, args.n_train, args.n_test, args.min_pos, args.min_neg, args.seed,
        pkl_path if pkl_path.exists() else None, image_root=str(Path(args.data_dir) / "images"),
    )
    images = manifest_image_list(man)
    n_pos = sum(len(u["pos"]) for u in man["users"].values())
    n_neg = sum(len(u["neg"]) for u in man["users"].values())
    print(f"manifest: {len(man['train_users'])} train users, {len(man['test_users'])} test users, "
          f"{n_pos} preferred + {n_neg} dispreferred pairs, {len(images)} distinct images -> {manifest_path}")
    list_path = data_dir / "image_list.txt"
    list_path.write_text("\n".join(images) + "\n")

    if args.extract:
        img_root = data_dir / "images"
        img_root.mkdir(parents=True, exist_ok=True)
        missing = [p for p in images if not (img_root / p).exists()]
        if not missing:
            print("all images already extracted")
            return
        print(f"extracting {len(missing)} images by streaming {PARTS[args.split]} tar parts (~24 GB)...")
        urls = [f"{HF}/{args.split}.tar.part-{i:05d}" for i in range(PARTS[args.split])]
        cmd = (
            "set -o pipefail; (" + " && ".join(f"curl -sL --retry 10 --retry-all-errors '{u}'" for u in urls) + ")"
            f" | tar -xf - -C '{img_root}' -T '{list_path}' --ignore-failed-read"
        )
        rc = subprocess.run(["bash", "-c", cmd]).returncode
        got = sum((img_root / p).exists() for p in images)
        print(f"tar rc={rc}; {got}/{len(images)} images present")


if __name__ == "__main__":
    main()
