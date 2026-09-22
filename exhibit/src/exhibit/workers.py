"""Offline inference entry point. No model is loaded into the web process."""

import json
import math
import resource
import sys
import time
from pathlib import Path

from .config import CONFIG, FAN_POLICIES, FAN_UPSTREAM, write_json
from .domain import digest, file_hash
from .fan_adapter import (
    encode_conditioning,
    freeze_policy,
    profiling_argument,
    resolve_policy,
    thaw_policy,
)
from .fan_mask import install_mask_fix


def emit(kind, **data):
    print(json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def _json_safe(value):
    """Schedulers report `-inf`/`nan`; strict JSON sinks need them spelled out."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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
            use_safetensors=True,
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
    import importlib

    from fan import FAN
    from fan.wrapper import stable_diffusion_xl

    weights = Path(upstream or FAN_UPSTREAM) / "weight"
    large = FAN(pipe.text_encoder, pipe.tokenizer, decoder=str(weights / "L.pth"))
    bigG = FAN(pipe.text_encoder_2, pipe.tokenizer_2, decoder=str(weights / "bigG.pth"))
    encoder = stable_diffusion_xl(large, bigG)
    encoder._fan_components = {"clip_l": large, "clip_g": bigG}
    encoder._fan_model_module = importlib.import_module("fan.model")
    # Upstream drops the causal mask as soon as a padding mask exists, and it
    # masks the target prompt's own pads as well as the references'. Install the
    # fix here rather than at encode time: it patches `fan.model` globally, and
    # callers that reach the encoder directly must get it too. It is a
    # documented no-op when only one mask is present, so `use_attn_mask=False`
    # stays bit-for-bit identical.
    install_mask_fix(encoder)
    return encoder


def fan_block(fan=None, policy=None):
    """The exact encoding settings behind an emitted image."""
    fan = fan or CONFIG["fan"]
    if policy is None:
        return {"commit": fan["commit"]}
    block = {
        "commit": fan["commit"],
        "alpha": float(policy["alpha"]),
        "pooled": policy["pooled_mode"],
        "skip": policy["skip"],
        "sample_size": profiling_argument(policy),
        "skip_pa": list(policy["skip_pa"]),
        "use_attn_mask": policy["use_attn_mask"],
    }
    if "embed_gain" in policy:
        block["embed_gain"] = float(policy["embed_gain"])
    return block


def effective_policy(personalization=None):
    """Resolve a JSON policy, including the temporary legacy service payload.

    Encoder settings come from the registered policy only; a legacy payload can
    still name the alpha and the profiling argument it was built with.
    """
    if personalization and personalization.get("effective_policy") is not None:
        return thaw_policy(freeze_policy(personalization["effective_policy"]))
    policy = thaw_policy(resolve_policy("legacy_exhibit", FAN_POLICIES))
    if not personalization:
        return policy

    sample_size = personalization.get("sample_size", profiling_argument(policy))
    if sample_size == 0:
        profiling = {"mode": "all"}
    elif type(sample_size) is int:
        profiling = {"mode": "count", "value": sample_size}
    else:
        profiling = {"mode": "ratio", "value": sample_size}
    legacy = {
        **policy,
        "alpha": personalization.get("alpha", policy["alpha"]),
        "profiling": profiling,
    }
    return thaw_policy(freeze_policy(legacy))


def effective_policy_id(personalization, policy, policy_hash):
    """Use a truthful registry label or an explicit custom content label."""
    supplied = personalization.get("policy_id") if personalization else None
    registered = FAN_POLICIES.get("policies", {})
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied:
            raise ValueError("policy_id must be a non-empty string")
        if supplied in registered:
            registered_hash = digest(thaw_policy(freeze_policy(registered[supplied])))
            if registered_hash != policy_hash:
                raise ValueError("policy_id does not match effective_policy")
        return supplied
    for policy_id, candidate in registered.items():
        candidate_hash = digest(thaw_policy(freeze_policy(candidate)))
        if candidate_hash == policy_hash:
            return policy_id
    return "custom:" + policy_hash[:16]


def generate(request, *, event_sink=None, conditioning_sink=None):
    import torch

    send = event_sink or emit

    settings = request.get("settings", CONFIG["generation"])
    fan = request.get("fan") or CONFIG["fan"]
    resolved_items = []
    for item in request["items"]:
        personalization = item.get("personalization")
        policy = effective_policy(personalization)
        policy_hash = digest(policy)
        supplied_hash = personalization.get("policy_hash") if personalization else None
        if supplied_hash is not None and supplied_hash != policy_hash:
            raise ValueError("policy_hash does not match effective_policy")
        policy_id = effective_policy_id(personalization, policy, policy_hash)
        resolved_items.append((item, personalization, policy, policy_hash, policy_id))

    started = time.monotonic()
    pipe = load_pipeline(settings)
    encoder = build_encoder(pipe, request.get("upstream"))
    send(
        "loaded",
        scheduler={
            "name": settings.get("scheduler", "EulerAncestralDiscreteScheduler"),
            "kwargs": settings.get("scheduler_kwargs", {}),
            "config": _json_safe(dict(pipe.scheduler.config)),
        },
        vae=settings.get("vae"),
        fan=fan_block(fan),
        load_seconds=round(time.monotonic() - started, 3),
    )
    plain = {}

    def plain_encode(prompt, policy):
        """Reference-free prompts are cached by their complete effective policy."""
        key = json.dumps([prompt, policy], sort_keys=True, separators=(",", ":"))
        if key not in plain:
            plain[key] = encode_conditioning(encoder, prompt, None, policy)
        return plain[key]

    for item, personalization, policy, policy_hash, policy_id in resolved_items:
        if request.get("emit_image_started"):
            send("image_started", id=item["id"])
        t = time.monotonic()
        negative = plain_encode(settings["negative_prompt"], policy)
        if personalization:
            conditioning = encode_conditioning(
                encoder,
                item["prompt"],
                personalization["refs"],
                policy,
                collect_trace=True,
            )
        else:
            conditioning = plain_encode(item["prompt"], policy)
        conditioning_metrics = None
        if conditioning_sink is not None:
            conditioning_metrics = conditioning_sink(
                item["id"], conditioning, plain_encode(item["prompt"], policy)
            )
        image = pipe(
            prompt_embeds=conditioning["hidden"],
            pooled_prompt_embeds=conditioning["pooled"],
            negative_prompt_embeds=negative["hidden"],
            negative_pooled_prompt_embeds=negative["pooled"],
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
        trace_path = path.with_suffix(".trace.json")
        write_json(trace_path, conditioning["trace"])
        personalization_hash = (
            (personalization.get("personalization_hash") or personalization.get("hash"))
            if personalization
            else None
        )
        send(
            "image",
            id=item["id"],
            path=str(path),
            sha256=file_hash(path),
            seed=item["seed"],
            prompt=item["prompt"],
            negative_prompt=settings["negative_prompt"],
            settings=settings,
            fan=fan_block(fan, policy),
            pooled=conditioning["trace"]["pooled_source"],
            policy_id=policy_id,
            effective_policy=conditioning["effective_policy"],
            policy_hash=policy_hash,
            profiling_trace=conditioning["trace"],
            profiling_trace_path=str(trace_path),
            personalization_hash=personalization_hash,
            conditioning_target_align=conditioning_metrics,
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
