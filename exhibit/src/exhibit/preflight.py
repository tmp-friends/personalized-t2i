"""Validate the fixed asset bundle and the FAN environment before an exhibition."""

import json
from pathlib import Path

from .config import ASSETS, CARDS_REVIEW, CONFIG, FAN_UPSTREAM, GPU_PYTHON
from .domain import (
    CARDS,
    build_legacy_personalization,
    file_hash,
    reviewed_ids,
    target_prompt,
)


def check_assets(root=ASSETS, *, require_samples=True, review=CARDS_REVIEW, catalog_id="catalog-v1"):
    from .catalog import load_catalog
    root = Path(root)
    errors = []

    def read(name, default):
        try:
            return json.loads((root / name).read_text())
        except (OSError, ValueError):
            errors.append(f"Missing or invalid {name}")
            return default

    manifest = read("manifest.json", {})
    if manifest.get("generation") != CONFIG["generation"]:
        errors.append("Generation settings mismatch")
    images = manifest.get("images", {}) if isinstance(manifest, dict) else {}

    def check_image(key, image):
        try:
            path = (root / image["path"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or file_hash(path) != image["sha256"]
            ):
                raise ValueError("hash mismatch")
        except (KeyError, OSError, TypeError, ValueError):
            errors.append(f"Missing or corrupt image: {key}")
            return False
        return True

    if catalog_id == "catalog-v1":
        catalog_cards = list(CARDS.values())
        reviewed = reviewed_ids(review)
    else:
        catalog = load_catalog(catalog_id, reviewed_only=True)
        all_catalog = load_catalog(catalog_id, reviewed_only=False)
        catalog_cards = all_catalog["all_cards"]
        reviewed = {card["id"] for card in catalog["cards"]}
    for card in catalog_cards:
        card_id = card["id"]
        image = images.get(card_id) or {}
        if check_image(card_id, image) and (
            image.get("seed") != card["seed"]
            or image.get("prompt") != card["prompt"]
            or image.get("ref_en") != card["ref_en"]
            or image.get("aspects") != card["aspects"]
            or image.get("settings") != CONFIG["generation"]
        ):
            errors.append(f"Card contract mismatch: {card_id}")
        if card_id not in reviewed:
            errors.append(f"Unreviewed card: {card_id}")

    prompts = read("generic-prompts.json", {})
    for topic in CONFIG["topics"]:
        prompt = target_prompt(topic)
        if (prompts.get(topic["id"]) or {}).get("prompt") != prompt:
            errors.append(f"Generic prompt mismatch: {topic['id']}")
        for index, seed in enumerate(CONFIG["seeds"]):
            key = f"{topic['id']}-{index}"
            image = images.get(key) or {}
            if check_image(key, image) and (
                image.get("seed") != seed
                or image.get("prompt") != prompt
                or image.get("personalization_hash") is not None
                or image.get("settings") != CONFIG["generation"]
            ):
                errors.append(f"Generic contract mismatch: {key}")

    samples = read("samples.json", []) if require_samples else []
    if require_samples and len(samples) != 6:
        errors.append("Six sample experiences required")
    for sample in samples:
        errors.extend(sample_errors(sample, root))
    return {
        "ready": not errors,
        "mode": "fan-live",
        "errors": errors,
        "fixed_images": len(images),
        "reviewed_cards": len(reviewed),
        "cards": len(catalog_cards),
        "samples": len(samples),
    }


def check_models():
    """Pinned Illustrious checkpoint, offline pipeline config and the FAN runtime."""
    errors = []
    models = []
    cache = Path.home() / ".cache/huggingface/hub"
    settings = CONFIG["generation"]
    path = (
        cache
        / ("models--" + settings["model"].replace("/", "--"))
        / "snapshots"
        / settings["revision"]
    )
    files = list(path.rglob("*.safetensors"))
    if not files or not all(p.exists() for p in files):
        errors.append(f"Missing pinned model: {settings['model']}")
    if settings.get("checkpoint"):
        if not (path / settings["checkpoint"]).is_file():
            errors.append(f"Missing pinned checkpoint: {settings['checkpoint']}")
        config = settings["pipeline_config"]
        config_path = (
            cache
            / ("models--" + config["model"].replace("/", "--"))
            / "snapshots"
            / config["revision"]
        )
        required = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
        ]
        for filename in required:
            if not (config_path / filename).is_file():
                errors.append(f"Missing offline pipeline config: {filename}")
    models.append(
        {
            "model": settings["model"],
            "revision": settings["revision"],
            "weight_files": len(files),
        }
    )
    vae = settings.get("vae")
    if vae:
        # The decoder is pinned separately; the checkpoint's own VAE is unused.
        vae_path = (
            cache
            / ("models--" + vae["model"].replace("/", "--"))
            / "snapshots"
            / vae["revision"]
        )
        vae_files = ["config.json", "diffusion_pytorch_model.safetensors"]
        missing = [name for name in vae_files if not (vae_path / name).is_file()]
        if missing:
            errors.append(f"Missing pinned VAE: {vae['model']} ({', '.join(missing)})")
        models.append(
            {
                "model": vae["model"],
                "revision": vae["revision"],
                "weight_files": len(vae_files) - len(missing),
                "role": "vae",
            }
        )
    fan = check_fan_env()
    errors.extend(fan["errors"])
    return {
        "ready": not errors,
        "errors": errors,
        "models": models,
        "gpu_python": GPU_PYTHON,
        "fan": fan,
    }


def check_fan_env(upstream=None, gpu_python=None):
    """The FAN environment and its published decoder weights, byte for byte."""
    errors = []
    gpu_python = gpu_python or GPU_PYTHON
    upstream = Path(upstream or FAN_UPSTREAM)
    if not Path(gpu_python).is_file():
        errors.append("FAN Python environment is missing")
    if not (upstream / "fan/model.py").is_file():
        errors.append(f"FAN upstream is missing: {upstream}")
    decoders = {}
    for name, expected in CONFIG["fan"]["decoders"].items():
        path = upstream / "weight" / name
        try:
            actual = file_hash(path)
        except OSError:
            errors.append(f"Missing decoder weight: {name}")
            continue
        decoders[name] = actual
        if actual != expected:
            errors.append(f"Decoder weight mismatch: {name}")
    return {
        "ready": not errors,
        "errors": errors,
        "python": gpu_python,
        "upstream": str(upstream),
        "commit": CONFIG["fan"]["commit"],
        "decoders": decoders,
    }


def write_preflight(path, *, include_models=False):
    """Evaluate and persist the same fresh preflight result returned to callers."""
    result = check_assets()
    if include_models:
        result["models"] = check_models()
        result["ready"] = result["ready"] and result["models"]["ready"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def sample_errors(sample, root=ASSETS):
    root = Path(root)
    errors = []
    sid = sample.get("id", "unknown")

    def fail(message):
        errors.append(f"Sample {sid}: {message}")

    topic = next(
        (t for t in CONFIG["topics"] if t["id"] == sample.get("topic_id")), None
    )
    if not topic:
        fail("unknown topic")
    selection = sample.get("selection")
    personalization = sample.get("personalization") or {}
    try:
        rebuilt = build_legacy_personalization(selection)
    except (KeyError, TypeError, ValueError):
        rebuilt = None
        fail("selection is unusable")
    if not rebuilt or rebuilt["hash"] != personalization.get("hash"):
        fail("personalization mismatch")
    images = sample.get("images", [])
    if len(images) != 4 or [x.get("seed") for x in images] != CONFIG["seeds"]:
        fail("seed order or image count")
    prompt = target_prompt(topic) if topic else None
    for image in images:
        if (
            image.get("settings") != CONFIG["generation"]
            or image.get("prompt") != prompt
        ):
            fail("generation settings or prompt")
        if image.get("personalization_hash") != personalization.get("hash"):
            fail("image personalization mismatch")
        try:
            path = (root / image["path"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or file_hash(path) != image["sha256"]
            ):
                fail("image hash")
        except (KeyError, OSError, TypeError):
            fail("image missing")
    return errors
