"""CPU/GPU smoke test: FAN on the CLIP-L text encoder only (no diffusion model needed)."""
import sys, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from transformers import CLIPModel, CLIPProcessor
from fan import FAN
import transformers, diffusers
print("transformers", transformers.__version__, "diffusers", diffusers.__version__, "torch", torch.__version__)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
fan = FAN(model, processor, decoder="weight/L.pth")
tgt = "A photograph of an astronaut riding a horse"
ref = ["A retro-futuristic space exploration movie poster with bold, vibrant colors"]
with torch.no_grad():
    base = fan(tgt, None, pooling=False)                      # plain encoding
    pers0 = fan(tgt, ref, weight=[1.0], alpha=0.0, pooling=False)
    pers4 = fan(tgt, ref, weight=[1.0], alpha=0.4, pooling=False)
    f_t = fan.get_text_feature(tgt)
    f_r = fan.get_text_feature(ref[0])
    f_p = fan.get_text_feature(tgt, ref, alpha=0.4)
    # profiling path: many refs + sample_size
    refs = ref * 5 + ["A watercolor painting of a cat", "Minimal line drawing, fine ink lines"] * 5
    prof = fan(tgt, refs, weight=[1.0]*len(refs), alpha=0.4, sample_size=0.3, pooling=True)
cos = torch.nn.functional.cosine_similarity
print("shape", tuple(base.shape), tuple(pers4.shape))
print("alpha=0 vs plain: max|diff| =", (pers0 - base).abs().max().item())
print("alpha=0.4 vs plain: mean token cos =", cos(pers4, base, dim=-1).mean().item())
print("feature cos(target, ref)      =", cos(f_t, f_r).item())
print("feature cos(personal, target) =", cos(f_p, f_t).item())
print("feature cos(personal, ref)    =", cos(f_p, f_r).item())
print("profiling path ok:", tuple(prof[0].shape), tuple(prof[1].shape))
