"""Loading FLUX.1-dev pieces on a single 24 GB GPU.

* transformer  : black-forest-labs/FLUX.1-dev (12B).  bf16 weights are 24 GB, so the
                 DiT blocks are quantised to float8 (optimum-quanto) block by block
                 -> ~12 GB.  Weights are frozen; gradients still flow through them to
                 the (fp32) preference adapters / user embeddings.
* VAE, CLIP-L, T5-XXL: identical weights are shared by FLUX.1-dev and FLUX.1-schnell,
                 and only the schnell repo is fully cached here, so we load them from
                 there (override with `aux_model_id`).
"""
from __future__ import annotations

import gc
import math

import torch
import torch.nn as nn

FLUX_DEV = "black-forest-labs/FLUX.1-dev"
FLUX_AUX = "black-forest-labs/FLUX.1-schnell"


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- transformer
def load_transformer(model_id: str = FLUX_DEV, quant: str | None = "quanto_fp8", device: str = "cuda",
                     dtype: torch.dtype = torch.bfloat16, gradient_checkpointing: bool = False):
    from diffusers import FluxTransformer2DModel

    if quant == "bnb_nf4":
        from diffusers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        t = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype,
                                                   quantization_config=qcfg, device_map=device)
    else:
        t = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
        if quant in ("quanto_fp8", "quanto_int8"):
            from optimum.quanto import freeze, qfloat8, qint8, quantize
            qtype = qfloat8 if quant == "quanto_fp8" else qint8
            # quantise block by block on the GPU so that the bf16 model never has to fit there at once
            for blk in list(t.transformer_blocks) + list(t.single_transformer_blocks):
                blk.to(device)
                quantize(blk, weights=qtype)
                freeze(blk)
        elif quant not in (None, "none"):
            raise ValueError(f"unknown quant {quant}")
        t.to(device)
    t.requires_grad_(False)
    t.eval()
    if gradient_checkpointing:
        t.enable_gradient_checkpointing()
    free_cuda()
    return t


# ----------------------------------------------------------------------------- text encoders
class TextEncoders:
    """CLIP-L pooled embedding (for the modulation vector y) + T5-XXL token embeddings."""

    def __init__(self, model_id: str = FLUX_AUX, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 max_len: int = 256):
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        self.device, self.dtype, self.max_len = device, dtype, max_len
        self.clip_tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.clip = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
        self.t5_tok = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")
        self.t5 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (t5_embeds [B, L, 4096], clip_pooled [B, 768]) — same recipe as FluxPipeline."""
        ids = self.t5_tok(prompts, padding="max_length", max_length=self.max_len, truncation=True,
                          return_length=False, return_overflowing_tokens=False, return_tensors="pt").input_ids
        t5 = self.t5(ids.to(self.device), output_hidden_states=False)[0].to(self.dtype)
        cids = self.clip_tok(prompts, padding="max_length", max_length=77, truncation=True,
                             return_overflowing_tokens=False, return_length=False, return_tensors="pt").input_ids
        pooled = self.clip(cids.to(self.device), output_hidden_states=False).pooler_output.to(self.dtype)
        return t5, pooled

    def unload(self):
        del self.clip, self.t5
        free_cuda()


# ----------------------------------------------------------------------------- VAE
def load_vae(model_id: str = FLUX_AUX, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def vae_encode_dist(vae, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pixels in [-1, 1] -> (mean, std) of the *raw* latent distribution (unscaled)."""
    dist = vae.encode(pixels.to(vae.device, vae.dtype)).latent_dist
    return dist.mean, dist.std


def raw_to_model_latents(raw: torch.Tensor, vae) -> torch.Tensor:
    return (raw - vae.config.shift_factor) * vae.config.scaling_factor


def model_to_raw_latents(lat: torch.Tensor, vae) -> torch.Tensor:
    return lat / vae.config.scaling_factor + vae.config.shift_factor


@torch.no_grad()
def vae_decode(vae, latents_model: torch.Tensor) -> torch.Tensor:
    """model-space latents [B,16,H/8,W/8] -> images in [0,1]."""
    raw = model_to_raw_latents(latents_model.to(vae.device, vae.dtype), vae)
    img = vae.decode(raw, return_dict=False)[0]
    return ((img.float() + 1) / 2).clamp(0, 1)


# ----------------------------------------------------------------------------- latent packing (FluxPipeline helpers)
def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)(W/2), 4C]"""
    b, c, h, w = latents.shape
    x = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """inverse of pack_latents; h, w are the (full) latent height/width."""
    b, n, d = packed.shape
    c = d // 4
    x = packed.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h, w)


def latent_image_ids(h: int, w: int, device, dtype) -> torch.Tensor:
    """positional ids for the packed latent grid; h, w are the full latent dims."""
    ids = torch.zeros(h // 2, w // 2, 3)
    ids[..., 1] += torch.arange(h // 2)[:, None]
    ids[..., 2] += torch.arange(w // 2)[None, :]
    return ids.reshape(-1, 3).to(device=device, dtype=dtype)


def text_ids(seq_len: int, device, dtype) -> torch.Tensor:
    return torch.zeros(seq_len, 3, device=device, dtype=dtype)


# ----------------------------------------------------------------------------- flow-matching time shift (FLUX dynamic shifting)
def flux_shift(image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15,
               base_seq_len: int = 256, max_seq_len: int = 4096) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = image_seq_len * m + (base_shift - m * base_seq_len)
    return math.exp(mu)


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def count_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))
