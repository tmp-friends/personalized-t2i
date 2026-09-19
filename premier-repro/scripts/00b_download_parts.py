"""Download the PrefBench tar parts with huggingface_hub (resumable, xet-accelerated),
extract only the manifest images, then delete the parts.

  python scripts/00b_download_parts.py --split diffusiondb [--keep-parts]
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
PARTS = {"diffusiondb": 6, "coco": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="diffusiondb")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "prefbench"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    parts_dir = data_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{args.split}.tar.part-{i:05d}" for i in range(PARTS[args.split])]

    def dl(name):
        p = hf_hub_download("wenyii/PrefBench", name, repo_type="dataset", local_dir=str(parts_dir))
        print("downloaded", p, flush=True)
        return p

    with ThreadPoolExecutor(args.workers) as ex:
        paths = list(ex.map(dl, names))
    img_root = data_dir / "images"
    img_root.mkdir(exist_ok=True)
    list_path = data_dir / "image_list.txt"
    cmd = f"cat {' '.join(repr(p) for p in paths)} | tar -xf - -C '{img_root}' -T '{list_path}'"
    print(cmd, flush=True)
    rc = subprocess.run(["bash", "-c", cmd]).returncode
    images = [l.strip() for l in open(list_path) if l.strip()]
    got = sum((img_root / p).exists() for p in images)
    print(f"tar rc={rc}; {got}/{len(images)} images present", flush=True)
    if not args.keep_parts and got == len(images):
        for p in paths:
            Path(p).unlink()
        print("parts deleted", flush=True)


if __name__ == "__main__":
    main()
