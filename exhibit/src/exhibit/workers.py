"""Offline inference entry point. No model is loaded into the web process."""

import json
import resource
import sys
import time
from pathlib import Path

from .config import CONFIG, FAN_UPSTREAM
from .domain import fan_settings, file_hash


def emit(kind, **data):
    print(json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def load_pipeline(settings):
    """Pinned single-file checkpoint; only architecture files come from the config."""
    import diffusers
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    if settings.get("checkpoint"):
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            settings["model"],
            filename=settings["checkpoint"],
            revision=settings["revision"],
            local_files_only=True,
        )
        config = settings["pipeline_config"]
        config_path = str(
            Path(
                hf_hub_download(
                    config["model"],
                    filename="model_index.json",
                    revision=config["revision"],
                    local_files_only=True,
                )
            ).parent
        )
        # Only architecture/tokenizer files come from this config; all neural
        # weights are loaded from the pinned single-file checkpoint.
        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint,
            config=config_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            settings["model"],
            revision=settings["revision"],
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        ).to("cuda")
    vae = settings.get("vae")
    if vae:
        # The checkpoint's own VAE decodes washed out in fp16; this one does not.
        pipe.vae = AutoencoderKL.from_pretrained(
            vae["model"],
            revision=vae["revision"],
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")
    name = settings.get("scheduler", "EulerAncestralDiscreteScheduler")
    scheduler = getattr(diffusers, name, None)
    if scheduler is None:
        raise ValueError(f"Unknown scheduler: {name}")
    pipe.scheduler = scheduler.from_config(
        pipe.scheduler.config, **settings.get("scheduler_kwargs", {})
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_encoder(pipe, upstream=None):
    """The published ClassTokenDecoder weights, bound to the pipeline's own encoders.

    ``fan.wrapper.personalized_t2i_encoder`` dispatches on ``pipeline.name_or_path``,
    which ``from_single_file`` leaves empty, so the SDXL branch is wired by hand.
    """
    from fan import FAN
    from fan.wrapper import stable_diffusion_xl

    weights = Path(upstream or FAN_UPSTREAM) / "weight"
    large = FAN(pipe.text_encoder, pipe.tokenizer, decoder=str(weights / "L.pth"))
    bigG = FAN(pipe.text_encoder_2, pipe.tokenizer_2, decoder=str(weights / "bigG.pth"))
    return stable_diffusion_xl(large, bigG)


# Tuning knobs the measured FAN call accepts straight from configs/demo.json.
PASSTHROUGH = ("use_attn_mask", "skip_pa")


def fan_block(fan=None):
    """The exact encoding settings behind every emitted image."""
    fan = fan or CONFIG["fan"]
    return {"commit": fan["commit"], "pooled": "plain", **fan_settings({"fan": fan})}


def encode(encoder, prompt, personalization=None, fan=None):
    """The single adaptation point for the measured FAN call: nothing else casts."""
    import torch

    fan = fan or CONFIG["fan"]
    refs = weights = None
    alpha = None
    if personalization:
        refs = [ref["text"] for ref in personalization["refs"]]
        weights = [float(ref["weight"]) for ref in personalization["refs"]]
        alpha = personalization["alpha"]
    with torch.no_grad():
        cond, pooled = encoder(
            prompt,
            refs,
            weight=weights,
            alpha=alpha,
            skip=fan["skip"],
            sample_size=fan["sample_size"],
            **{key: fan[key] for key in PASSTHROUGH if key in fan},
        )
    return cond.to(torch.float16), pooled.to(torch.float16)


def generate(request):
    import torch

    settings = request.get("settings", CONFIG["generation"])
    started = time.monotonic()
    pipe = load_pipeline(settings)
    encoder = build_encoder(pipe, request.get("upstream"))
    fan = request.get("fan") or CONFIG["fan"]
    emit(
        "loaded",
        scheduler={
            "name": settings.get("scheduler", "EulerAncestralDiscreteScheduler"),
            "kwargs": settings.get("scheduler_kwargs", {}),
            "config": dict(pipe.scheduler.config),
        },
        vae=settings.get("vae"),
        fan=fan_block(fan),
        load_seconds=round(time.monotonic() - started, 3),
    )
    plain = {}

    def plain_encode(prompt):
        """Reference-free encoding is bit-identical to `pipe.encode_prompt`."""
        if prompt not in plain:
            plain[prompt] = encode(encoder, prompt, fan=fan)
        return plain[prompt]

    # The negative side is never personalized: only ref/weight/alpha may differ.
    negative_cond, negative_pooled = plain_encode(settings["negative_prompt"])
    for item in request["items"]:
        personalization = item.get("personalization")
        t = time.monotonic()
        # Documented deviation from upstream: the personalized pooled embedding
        # comes from a mis-detected padding token, so the plain pooled is used.
        cond, pooled = plain_encode(item["prompt"])
        if personalization:
            cond = encode(encoder, item["prompt"], personalization, fan)[0]
        image = pipe(
            prompt_embeds=cond,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_cond,
            negative_pooled_prompt_embeds=negative_pooled,
            num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance_scale"],
            width=settings["width"],
            height=settings["height"],
            generator=torch.Generator(device="cuda").manual_seed(item["seed"]),
        ).images[0]
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.png")
        image.save(temporary)
        temporary.replace(path)
        emit(
            "image",
            id=item["id"],
            path=str(path),
            sha256=file_hash(path),
            seed=item["seed"],
            prompt=item["prompt"],
            negative_prompt=settings["negative_prompt"],
            settings=settings,
            fan=fan_block(fan),
            pooled="plain",
            personalization_hash=personalization["hash"] if personalization else None,
            seconds=round(time.monotonic() - t, 3),
        )


def main():
    import torch

    started = time.monotonic()
    request = json.loads(Path(sys.argv[1]).read_text())
    torch.set_num_threads(4)
    {"generate": generate}[request["stage"]](request)
    torch.cuda.synchronize()
    emit(
        "metrics",
        stage=request["stage"],
        worker_seconds=round(time.monotonic() - started, 3),
        peak_vram_mib=round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        peak_rss_mib=round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
    )


if __name__ == "__main__":
    main()
