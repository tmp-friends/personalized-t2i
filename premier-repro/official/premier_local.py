"""Shared helpers to run the official Premier code (github.com/120L020904/Premier) on this
machine: local FLUX.1-dev pieces, released weights (hf.co/pino10010/Premier) and the
memory handling needed for a single 24 GB GPU.

The official code assumes `FluxPipeline.from_pretrained(...).to("cuda")` in bf16
(~33 GB).  Here the frozen FLUX transformer (and T5) are stored in 8 bit:

  memory="fp8"  : diffusers layerwise casting (float8 storage, bf16 compute)  - inference only
  memory="int8" : optimum-quanto int8 weights (dequantised on the fly)        - inference + training
  memory="bf16" : no compression (needs >= 40 GB)
  memory="offload": sequential CPU offload (slow, tiny VRAM, inference only)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Premier"))

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast  # noqa: E402

FLUX_DEV = "black-forest-labs/FLUX.1-dev"
FLUX_AUX = "black-forest-labs/FLUX.1-schnell"   # VAE / CLIP-L / T5-XXL: identical weights to dev, fully cached here
WEIGHTS = HERE / "weights" / "pino10010_Premier"
# scheduler_config.json of FLUX.1-dev (only the transformer of the dev repo is cached locally)
DEV_SCHEDULER = dict(base_image_seq_len=256, max_image_seq_len=4096, base_shift=0.5, max_shift=1.15,
                     num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True)
TOKEN_NUM = 30
USER_DIM = 1024
N_TRAIN_USERS = 1000


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cuda_gb() -> str:
    return f"{torch.cuda.memory_allocated() / 2**30:.1f} GB (peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)"


def load_adapter_config(weights: Path = WEIGHTS) -> dict:
    return yaml.safe_load(open(Path(weights) / "adapter_config.yaml"))


# ----------------------------------------------------------------------------- FLUX components
def load_components(transformer_id: str = FLUX_DEV, aux_id: str = FLUX_AUX, dtype=torch.bfloat16,
                    with_text_encoders: bool = True, with_vae: bool = True, with_transformer: bool = True) -> dict:
    c = {}
    if with_transformer:
        log(f"loading FLUX.1-dev transformer ({transformer_id}) ...")
        c["transformer"] = FluxTransformer2DModel.from_pretrained(transformer_id, subfolder="transformer", torch_dtype=dtype)
    if with_text_encoders:
        log(f"loading CLIP-L / T5-XXL ({aux_id}) ...")
        c["text_encoder"] = CLIPTextModel.from_pretrained(aux_id, subfolder="text_encoder", torch_dtype=dtype)
        c["tokenizer"] = CLIPTokenizer.from_pretrained(aux_id, subfolder="tokenizer")
        c["text_encoder_2"] = T5EncoderModel.from_pretrained(aux_id, subfolder="text_encoder_2", torch_dtype=dtype)
        c["tokenizer_2"] = T5TokenizerFast.from_pretrained(aux_id, subfolder="tokenizer_2")
    if with_vae:
        c["vae"] = AutoencoderKL.from_pretrained(aux_id, subfolder="vae", torch_dtype=dtype)
    c["scheduler"] = FlowMatchEulerDiscreteScheduler(**DEV_SCHEDULER)
    for k in ("transformer", "text_encoder", "text_encoder_2", "vae"):
        if k in c:
            c[k].requires_grad_(False).eval()
    return c


def make_pipeline(c: dict) -> FluxPipeline:
    return FluxPipeline(scheduler=c["scheduler"], vae=c.get("vae"), text_encoder=c.get("text_encoder"),
                        tokenizer=c.get("tokenizer"), text_encoder_2=c.get("text_encoder_2"),
                        tokenizer_2=c.get("tokenizer_2"), transformer=c.get("transformer"))


def quantize_int8(module: nn.Module, device: str, blocks_only: bool = True):
    """optimum-quanto int8 weights, quantised block by block on the GPU (no bf16 copy of the
    whole 24 GB model on the GPU at any time).  Works without nvcc / CUDA toolkit."""
    from optimum.quanto import freeze, qint8, quantize
    if blocks_only and hasattr(module, "transformer_blocks"):
        parts = list(module.transformer_blocks) + list(module.single_transformer_blocks)
    elif blocks_only and hasattr(module, "encoder") and hasattr(module.encoder, "block"):   # T5
        parts = list(module.encoder.block)
    else:
        parts = [module]
    for p in parts:
        p.to(device)
        quantize(p, weights=qint8)
        freeze(p)
    module.to(device)
    return module


def fp8_cast(module: nn.Module, compute_dtype=torch.bfloat16, skip_pattern=("pos_embed", "norm", "shared", "embed", "wo")):
    """float8 storage / bf16 compute via diffusers layerwise casting (inference only).
    T5 casts activations to `wo.weight.dtype` before calling `wo`, so `wo` must stay in bf16.
    For the FLUX transformer pass skip_pattern=("pos_embed",) so that the large AdaLN projections
    (`norm1.linear`, 3.2B params) are cast too; real norm layers carry no / tiny weights there."""
    from diffusers.hooks import apply_layerwise_casting
    from diffusers.models.normalization import RMSNorm
    apply_layerwise_casting(module, storage_dtype=torch.float8_e4m3fn, compute_dtype=compute_dtype,
                            skip_modules_pattern=tuple(skip_pattern),
                            skip_modules_classes=(nn.LayerNorm, RMSNorm, nn.Embedding))
    return module


def free_gpu_gb(device: str = "cuda") -> float:
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return free / 2**30


def apply_memory_mode(pipe: FluxPipeline, memory: str, device: str, dtype=torch.bfloat16, t5: str = "auto"):
    """memory: fp8 | int8 | bf16 | offload.   t5: gpu | offload | auto (offload when < 20 GB free).
    With t5="offload" the T5 encoder stays in CPU RAM and is streamed layer by layer through the
    GPU by accelerate (a few seconds per prompt); use it when another process occupies the GPU."""
    tr, t5m = pipe.transformer, pipe.text_encoder_2
    if t5 == "auto":
        t5 = "offload" if free_gpu_gb(device) < 20 else "gpu"
    if memory == "offload":
        pipe.enable_sequential_cpu_offload(device=device)
        return pipe
    if memory == "fp8":
        from diffusers.hooks import apply_layerwise_casting
        # diffusers' default skip pattern keeps every module whose name contains "norm" in bf16 - for
        # FLUX that includes the big AdaLN projections (3.2B params, +3 GB), so we only skip real norm layers
        fp8_cast(tr, dtype, skip_pattern=("pos_embed",))
        if t5m is not None and t5 == "gpu":
            fp8_cast(t5m, dtype)
    elif memory == "int8":
        quantize_int8(tr, device)
        if t5m is not None and t5 == "gpu":
            quantize_int8(t5m, device)
    elif memory != "bf16":
        raise ValueError(memory)
    for name in ("transformer", "vae", "text_encoder"):
        m = getattr(pipe, name, None)
        if m is not None:
            m.to(device)
    if t5m is not None:
        if t5 == "gpu":
            t5m.to(device)
        else:
            from accelerate import cpu_offload
            cpu_offload(t5m, execution_device=torch.device(device))
    log(f"memory mode {memory}, T5 {t5}; free GPU {free_gpu_gb(device):.1f} GB")
    return pipe


def memoize_encode_prompt(pipe: FluxPipeline):
    """Cache pipe.encode_prompt results (the official generate_xverse re-encodes the empty prompt
    for every image)."""
    orig = pipe.encode_prompt
    cache = {}

    def encode_prompt(prompt=None, prompt_2=None, device=None, num_images_per_prompt=1, prompt_embeds=None,
                      pooled_prompt_embeds=None, max_sequence_length=512, lora_scale=None):
        if prompt_embeds is not None:
            return orig(prompt=prompt, prompt_2=prompt_2, device=device, num_images_per_prompt=num_images_per_prompt,
                        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                        max_sequence_length=max_sequence_length, lora_scale=lora_scale)
        key = (str(prompt), str(prompt_2), num_images_per_prompt, max_sequence_length)
        if key not in cache:
            cache[key] = orig(prompt=prompt, prompt_2=prompt_2, device=device, num_images_per_prompt=num_images_per_prompt,
                              max_sequence_length=max_sequence_length, lora_scale=lora_scale)
        return cache[key]

    pipe.encode_prompt = encode_prompt
    return pipe


# ----------------------------------------------------------------------------- released user embeddings
def load_train_bank(device, dtype, weights: Path = WEIGHTS) -> nn.Embedding:
    """user_embedding.safetensors: nn.Embedding(1000, 30*1024) of the 1000 training users."""
    emb = nn.Embedding(N_TRAIN_USERS, TOKEN_NUM * USER_DIM)
    emb.load_state_dict(load_file(str(Path(weights) / "user_embedding.safetensors")))
    return emb.to(device=device, dtype=dtype).requires_grad_(False)


def load_user(spec: str, bank: nn.Embedding | None, device, dtype, weights: Path = WEIGHTS) -> tuple[str, torch.Tensor | None]:
    """spec: none | train:<0-999> | test:<id> | linear:<id> | file:<path.safetensors>
    returns (label, user_preference_embedding [1, 30, 1024] or None)"""
    weights = Path(weights)
    kind, _, key = spec.partition(":")
    if kind == "none":
        return "base FLUX.1-dev", None
    if kind == "train":
        i = int(key)
        return f"train user #{i}", bank.weight[i].detach().view(1, TOKEN_NUM, USER_DIM)
    if kind == "test":
        sd = load_file(str(weights / "users" / f"user_embedding_{key}.safetensors"))
        w = sd["weight"] if "weight" in sd else next(iter(sd.values()))
        return f"test user {key} (direct)", w.reshape(1, TOKEN_NUM, USER_DIM).to(device=device, dtype=dtype)
    if kind == "linear":
        from scripts.train_flux.train_user_embedding_linear import EmbeddingLinearCombination
        comb = EmbeddingLinearCombination(embedding_num=N_TRAIN_USERS, combination_size=1, use_softmax=False)
        comb.load_state_dict(load_file(str(weights / "users_linear" / f"user_combination_{key}.safetensors")))
        comb = comb.to(device=device, dtype=dtype)
        with torch.no_grad():
            w = comb(bank, input_ids=torch.tensor([0], device=device))
        return f"test user {key} (linear comb.)", w.reshape(1, TOKEN_NUM, USER_DIM)
    if kind == "file":
        p = Path(key)
        sd = load_file(str(p))
        if "combination_weights" in sd:          # EmbeddingLinearCombination state dict
            from scripts.train_flux.train_user_embedding_linear import EmbeddingLinearCombination
            comb = EmbeddingLinearCombination(embedding_num=N_TRAIN_USERS, combination_size=1, use_softmax=False)
            comb.load_state_dict(sd)
            comb = comb.to(device=device, dtype=dtype)
            with torch.no_grad():
                w = comb(bank, input_ids=torch.tensor([0], device=device))
        else:
            w = sd["weight"] if "weight" in sd else next(iter(sd.values()))
        return f"user file {p.stem}", w.reshape(1, TOKEN_NUM, USER_DIM).to(device=device, dtype=dtype)
    raise ValueError(spec)


def image_grid(pils, cols: int, pad: int = 4):
    from PIL import Image
    w, h = pils[0].size
    rows = (len(pils) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (255, 255, 255))
    for i, im in enumerate(pils):
        grid.paste(im, ((i % cols) * (w + pad), (i // cols) * (h + pad)))
    return grid
