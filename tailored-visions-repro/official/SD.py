import os

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

# [2026 patch] The paper used SD v1-5, whose `runwayml` repo was removed from the
# Hub in Aug 2024. The default is now SDXL; set TV_SD_MODEL to go back to v1-5
# ("stable-diffusion-v1-5/stable-diffusion-v1-5") or to any other t2i repo.
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_MODEL = os.environ.get("TV_SD_MODEL", SDXL_MODEL)

# SDXL renders at 1024x1024 against v1-5's 512x512, so a batch of 4 costs ~4x the
# activation memory. The local rewriting LLM keeps ~9GB resident (SETUP.md §3),
# which leaves ~15GB here -- enough, but only with VAE slicing on.
DEFAULT_BATCH = int(os.environ.get("TV_SD_BATCH", "4"))


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_pipeline(model_id=None, device=None, dtype=torch.float16):
    """[2026 patch] Load any t2i pipeline by repo id, SD v1-5 or SDXL alike.

    Upstream hardcoded ``StableDiffusionPipeline``, which cannot load SDXL.
    ``AutoPipelineForText2Image`` picks the class from the repo's
    ``model_index.json`` instead. Two knobs differ per repo and are probed
    rather than assumed:

    * ``variant="fp16"`` -- halves the download, but not every repo ships it.
    * ``safety_checker`` -- an SD v1.x argument; the SDXL pipeline has no such
      parameter and rejects it.
    """
    model_id = model_id or DEFAULT_MODEL
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pipe = None
    errors = []
    no_checker = {"safety_checker": None, "requires_safety_checker": False}
    for variant in ({"variant": "fp16"}, {}):
        for checker in (no_checker, {}):
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, torch_dtype=dtype, **variant, **checker)
                break
            except (TypeError, ValueError, OSError) as exc:
                errors.append(f"{variant or 'no variant'} / {'no checker' if checker else 'default checker'}: {exc}")
        if pipe is not None:
            break
    if pipe is None:
        raise RuntimeError(f"could not load {model_id}:\n  " + "\n  ".join(errors))

    # Decoding a batch of 1024x1024 latents in one go is the peak allocation of
    # an SDXL run; slicing it is free in quality and removes the spike.
    pipe.enable_vae_slicing()

    free, _ = torch.cuda.mem_get_info() if device == "cuda" else (0, 0)
    if device == "cuda" and free < 12 * 1024 ** 3:
        # Not enough room to hold the weights and run a batch -- stream the
        # components from CPU instead of failing with OOM. Slower, but works
        # alongside the rewriting LLM server.
        print(f"[SD] only {free / 1024 ** 3:.1f}GB free on the GPU; enabling model CPU offload")
        pipe.enable_model_cpu_offload()
        # Moving an offloaded pipeline with .to() re-pins every component to the
        # GPU and defeats the offload, so flag it for callers that do that.
        pipe._tv_offloaded = True
        return pipe
    pipe._tv_offloaded = False
    return pipe.to(device)


@torch.no_grad()
def text2img(pipe,prompt,batch_size=None,save=False,save_path=None,**kwargs):
    if batch_size is None:
        batch_size = DEFAULT_BATCH
    image = pipe(prompt,num_images_per_prompt=batch_size,**kwargs).images
    if save and save_path is not None:
        grid=image_grid(image,1,batch_size)
        grid.save(save_path)
    return image

if __name__=="__main__":
    pipe = load_pipeline()
    prompt="an astronaut riding a horse"
    text2img(pipe,prompt,save=True,save_path="astronaut_rides_horse.png")
