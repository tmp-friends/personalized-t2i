"""Generate a small, style-consistent "preferred image" set with plain FLUX.1-dev
(used to sanity-check new-user training: does the learned user embedding pick up the style?).

  python scripts/make_style_set.py --style "watercolor painting, soft pastel colors, paper texture" --name watercolor --out data/examples
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from premier_repro.model.loading import TextEncoders, load_transformer, load_vae, free_cuda
from premier_repro.infer.sampler import generate, to_pil

SUBJECTS = ["a cat sitting on a windowsill", "a lighthouse on a rocky coast", "a bowl of fruit on a wooden table",
            "a woman reading a book in a cafe", "a mountain village in winter", "a vintage bicycle leaning on a wall",
            "a fox in a forest", "a sailboat at sunset"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/examples"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    subjects = SUBJECTS[: a.n]
    prompts = [f"{s}, {a.style}" for s in subjects]
    enc = TextEncoders(max_len=256)
    t5, pooled = enc.encode(prompts)
    empty = enc.encode([""])
    enc.unload(); free_cuda()
    tr = load_transformer(quant="quanto_int8")
    vae = load_vae()
    items = []
    for i, (s, p) in enumerate(zip(subjects, prompts)):
        img = generate(tr, vae, t5[i:i+1], pooled[i:i+1], a.size, a.size, steps=a.steps, guidance=3.5, seed=a.seed + i)
        f = out / f"{i:02d}.png"
        to_pil(img)[0].save(f)
        # caption WITHOUT the style words: the style must be learned by the user embedding, not read from the prompt
        items.append({"image": f.name, "caption": s, "generation_prompt": p})
        print("saved", f, flush=True)
    json.dump(items, open(out / "items.json", "w"), indent=1)
    print("done", out / "items.json")

if __name__ == "__main__":
    main()
