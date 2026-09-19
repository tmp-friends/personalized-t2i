#!/usr/bin/env python
"""Generate images with the *official* Premier implementation and the released weights.

Rows = users (`--users`), columns = prompts (`--prompts`), same seed per column so the
base model and every user can be compared side by side (outputs <out>/grid.jpg).

  --users  none            : plain FLUX.1-dev (no preference)
           train:<0-999>   : one of the 1000 training-user embeddings
           test:<id>       : released test user, directly trained embedding   (users/user_embedding_<id>.safetensors)
           linear:<id>     : released test user, linear-combination embedding (users_linear/user_combination_<id>.safetensors)
           file:<path>     : a .safetensors produced by train_new_user.py
  --memory fp8 (default) | int8 | bf16 | offload      (see premier_local.py)

Examples
  python run_premier.py --users none train:0 train:1 --prompts "a cat sitting on a windowsill" "a city street at night" --out outputs/demo
  python run_premier.py --users none test:3685 linear:3685 --prompts "a portrait of a woman" --out outputs/user3685
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from premier_local import (WEIGHTS, apply_memory_mode, cuda_gb, image_grid, load_adapter_config, load_components,
                           load_train_bank, load_user, log, make_pipeline, memoize_encode_prompt)
from scripts.pipeline.flux_adapter import generate_xverse  # official
from scripts.pipeline.mod_adapters import load_modulation_adapter  # official


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", nargs="+", default=["none", "train:0", "train:1"])
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=512, help="T5 sequence length (official default 512)")
    ap.add_argument("--memory", default="fp8", choices=["fp8", "int8", "bf16", "offload"])
    ap.add_argument("--t5", default="auto", choices=["auto", "gpu", "offload"], help="keep T5 on GPU or stream it from CPU RAM")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=str(WEIGHTS))
    a = ap.parse_args()
    weights = Path(a.weights)
    dtype = torch.bfloat16
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    adapter_config = load_adapter_config(weights)
    pipe = apply_memory_mode(make_pipeline(load_components(dtype=dtype)), a.memory, a.device, dtype, t5=a.t5)
    memoize_encode_prompt(pipe)
    log(f"pipeline ready ({a.memory}); cuda {cuda_gb()}")
    mod_adapter = load_modulation_adapter(adapter_config, dtype, a.device, ckpt_dir=str(weights), is_training=False)
    mod_adapter.eval()
    need_bank = any(s.split(":")[0] in ("train", "linear", "file") for s in a.users)
    bank = load_train_bank(a.device, dtype, weights) if need_bank else None
    rows = [load_user(s, bank, a.device, dtype, weights) for s in a.users]
    log(f"adapter + user embeddings ready; cuda {cuda_gb()}")

    pils, records = [], []
    for r, (label, ue) in enumerate(rows):
        for c, prompt in enumerate(a.prompts):
            gen = torch.Generator(a.device).manual_seed(a.seed + c)
            t0 = time.time()
            if ue is None:
                img = pipe(prompt=prompt, prompt_2=prompt, height=a.size, width=a.size, num_inference_steps=a.steps,
                           guidance_scale=a.guidance, generator=gen, max_sequence_length=a.max_seq_len).images[0]
            else:
                img = generate_xverse(pipeline=pipe, mod_adapter=mod_adapter, user_preference_embedding=ue,
                                      prompt=prompt, prompt_2=prompt, height=a.size, width=a.size,
                                      num_inference_steps=a.steps, guidance_scale=a.guidance, generator=gen,
                                      max_sequence_length=a.max_seq_len, model_config=adapter_config).images[0]
            path = out / f"r{r:02d}_{a.users[r].replace(':', '-').replace('/', '_')}_p{c:02d}.png"
            img.save(path)
            pils.append(img)
            records.append({"row": r, "user": a.users[r], "label": label, "col": c, "prompt": prompt, "file": path.name})
            log(f"{label} | {prompt[:60]} | {time.time() - t0:.1f}s | cuda {cuda_gb()}")
    image_grid(pils, cols=len(a.prompts)).save(out / "grid.jpg", quality=92)
    json.dump({"rows": [r[0] for r in rows], "cols": a.prompts, "seed": a.seed, "steps": a.steps,
               "guidance": a.guidance, "size": a.size, "memory": a.memory, "t5": a.t5, "items": records},
              open(out / "grid.json", "w"), indent=1, ensure_ascii=False)
    log(f"saved {out / 'grid.jpg'}")


if __name__ == "__main__":
    main()
