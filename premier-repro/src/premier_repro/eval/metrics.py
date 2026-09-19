"""Evaluation metrics used in the paper (+ a cheap SigLIP preference proxy).

  ClipScorer      : CLIP T2I cosine similarity (openai/clip-vit-large-patch14)
  LpipsScorer     : LPIPS (AlexNet) between the generated image and the user's preferred image of the same prompt
  SiglipPrefScorer: cos(x, mean(liked)) - cos(x, mean(disliked)) with SigLIP-so400m image embeddings (fast proxy)
  ViperProxy      : the ViPer proxy metric (IDEFICS2-8B + LoRA "EPFL-VILAB/Metric-ViPer"), score in [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


class ClipScorer:
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPProcessor
        self.dev = device
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        inp = self.proc(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True).to(self.dev)
        img = F.normalize(self.model.get_image_features(pixel_values=inp["pixel_values"].half()), dim=-1)
        txt = F.normalize(self.model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]), dim=-1)
        return (img * txt).sum(-1).float().cpu().tolist()


class LpipsScorer:
    def __init__(self, device="cuda", net="alex", size=256):
        import lpips
        self.dev, self.size = device, size
        self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    def _t(self, im: Image.Image) -> torch.Tensor:
        import numpy as np
        im = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1
        return x[None].to(self.dev)

    @torch.no_grad()
    def score(self, a: Image.Image, b: Image.Image) -> float:
        return float(self.fn(self._t(a), self._t(b)).item())


class SiglipPrefScorer:
    def __init__(self, device="cuda", model_id="google/siglip-so400m-patch14-384"):
        from transformers import SiglipModel, SiglipProcessor
        self.dev = device
        self.model = SiglipModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        self.proc = SiglipProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        out = []
        for i in range(0, len(images), 16):
            px = self.proc(images=images[i:i + 16], return_tensors="pt")["pixel_values"].to(self.dev).half()
            out.append(F.normalize(self.model.get_image_features(pixel_values=px).float(), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def score(self, images: list[Image.Image], liked: list[Image.Image], disliked: list[Image.Image]) -> list[float]:
        x = self.embed(images)
        pos = F.normalize(self.embed(liked).mean(0), dim=-1)
        s = x @ pos
        if disliked:
            neg = F.normalize(self.embed(disliked).mean(0), dim=-1)
            s = s - x @ neg
        return s.cpu().tolist()


class ViperProxy:
    """ViPer proxy metric (Salehi et al. 2024): P('+' | liked/disliked context, query)."""

    def __init__(self, device="cuda", base="HuggingFaceM4/idefics2-8b", adapter="EPFL-VILAB/Metric-ViPer",
                 load_in_4bit: bool = False):
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.dev = device
        self.proc = AutoProcessor.from_pretrained(base, size={"longest_edge": 448, "shortest_edge": 378},
                                                  do_image_splitting=False)
        kw = dict(torch_dtype=torch.bfloat16)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForVision2Seq.from_pretrained(base, device_map=device, **kw)
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.plus_id, self.minus_id = 648, 387  # token ids of "+" and "-" (from ViPer/metric.py)

    @torch.no_grad()
    def score(self, query: Image.Image, liked: list[Image.Image], disliked: list[Image.Image]) -> float:
        n = min(len(liked), len(disliked))
        ctx, prompt = [], ""
        for i in range(n):
            ctx += [disliked[i], liked[i]]
            prompt += "User:<image>Score for this image?<end_of_utterance>\nAssistant: -<end_of_utterance>\n"
            prompt += "User:<image>Score for this image?<end_of_utterance>\nAssistant: +<end_of_utterance>\n"
        ctx.append(query)
        prompt += "User:<image>Score for this image?<end_of_utterance>\nAssistant: "
        inp = self.proc(text=prompt, images=ctx, return_tensors="pt").to(self.dev)
        logits = self.model(**inp).logits[:, -1]
        p = torch.exp(logits[:, self.plus_id]) / (torch.exp(logits[:, self.plus_id]) + torch.exp(logits[:, self.minus_id]))
        return float(p.item())
