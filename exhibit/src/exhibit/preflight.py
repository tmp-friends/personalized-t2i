"""Validate the fixed asset bundle and the FAN environment before an exhibition."""

import json
from pathlib import Path

from .config import ASSETS, CARDS_REVIEW, CONFIG, FAN_UPSTREAM, GPU_PYTHON
from .domain import (
    build_legacy_personalization,
    file_hash,
    target_prompt,
)


def check_assets(root=ASSETS, *, require_samples=True, review=None, catalog_id=None):
    from .catalog import card_settings, load_catalog, validate_token_report

    root = Path(root)
    catalog_id = catalog_id or CONFIG.get("catalog_id")
    errors = []
    try:
        # v2 cards carry the catalog's own protective negative prompt.
        card_generation = card_settings(catalog_id)
    except (OSError, TypeError, ValueError):
        card_generation = CONFIG["generation"]

    def read(name, default):
        try:
            return json.loads((root / name).read_text())
        except (OSError, ValueError):
            errors.append(f"Missing or invalid {name}")
            return default

    def read_object(name):
        value = read(name, {})
        if not isinstance(value, dict):
            errors.append(f"Invalid {name} structure")
            return {}
        return value

    def image_map(manifest, name):
        value = manifest.get("images", {})
        if not isinstance(value, dict):
            errors.append(f"Invalid {name} images")
            return {}
        return value

    baseline_manifest = read_object("manifest.json")
    baseline_images = image_map(baseline_manifest, "manifest.json")
    if baseline_manifest.get("generation") != CONFIG["generation"]:
        errors.append("Generation settings mismatch")

    if catalog_id == "catalog-v1":
        card_manifest = baseline_manifest
        card_images = baseline_images
        review_path = review or CARDS_REVIEW
    elif catalog_id == "catalog-v2":
        card_manifest = read_object("catalog-v2.json")
        card_images = image_map(card_manifest, "catalog-v2.json")
        review_path = review
        if card_manifest.get("generation") != card_generation:
            errors.append("Generation settings mismatch: catalog-v2.json")
    else:
        errors.append(f"Unknown catalog: {catalog_id}")
        card_manifest = {}
        card_images = {}
        review_path = review

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

    try:
        eligible = load_catalog(
            catalog_id, reviewed_only=True, assets=root, review_path=review_path
        )
        complete = load_catalog(
            catalog_id, reviewed_only=False, assets=root, review_path=review_path
        )
        catalog_cards = complete["all_cards"]
        reviewed = {card["id"] for card in eligible["cards"]}
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Invalid catalog: {exc}")
        catalog_cards = []
        reviewed = set()

    if catalog_id == "catalog-v2" and catalog_cards:
        try:
            validate_token_report(
                card_manifest.get("token_validation"),
                catalog_cards,
                generation=card_generation,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"Invalid token validation: {exc}")

    card_fields = ["seed", "prompt", "ref_en", "aspects"]
    if catalog_id == "catalog-v2":
        card_fields.extend(("aspects_ja", "label", "profile_label"))
    for card in catalog_cards:
        card_id = card["id"]
        image = card_images.get(card_id) if isinstance(card_images, dict) else None
        if check_image(card_id, image or {}) and (
            image.get("path") != card["path"]
            or any(image.get(field) != card[field] for field in card_fields)
            or image.get("settings") != card_generation
        ):
            errors.append(f"Card contract mismatch: {card_id}")
        if card_id not in reviewed:
            errors.append(f"Unreviewed card: {card_id}")

    prompts = read_object("generic-prompts.json")
    for topic in CONFIG["topics"]:
        prompt = target_prompt(topic)
        prompt_record = prompts.get(topic["id"])
        if not isinstance(prompt_record, dict) or prompt_record.get("prompt") != prompt:
            errors.append(f"Generic prompt mismatch: {topic['id']}")
        for index, seed in enumerate(CONFIG["seeds"]):
            key = f"{topic['id']}-{index}"
            image = (
                baseline_images.get(key) if isinstance(baseline_images, dict) else None
            )
            if check_image(key, image or {}) and (
                image.get("seed") != seed
                or image.get("prompt") != prompt
                or image.get("personalization_hash") is not None
                or image.get("settings") != CONFIG["generation"]
            ):
                errors.append(f"Generic contract mismatch: {key}")

    samples = read("samples.json", []) if require_samples else []
    if require_samples:
        if not isinstance(samples, list):
            errors.append("Invalid samples manifest")
            samples = []
        contract = CONFIG.get("sample_manifest")
        required_ids = (
            contract.get("required_ids") if isinstance(contract, dict) else None
        )
        if (
            not isinstance(contract, dict)
            or set(contract) != {"catalog_id", "required_ids"}
            or contract.get("catalog_id") != "catalog-v1"
            or not isinstance(required_ids, list)
            or any(not isinstance(item, str) or not item for item in required_ids)
            or len(required_ids) != len(set(required_ids or ()))
        ):
            errors.append("Invalid sample manifest contract")
        else:
            actual_ids = [
                sample.get("id") if isinstance(sample, dict) else None
                for sample in samples
            ]
            if any(not isinstance(item, str) or not item for item in actual_ids):
                errors.append("Sample manifest has invalid id")
            elif len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(
                required_ids
            ):
                errors.append("Sample manifest IDs mismatch")
    for sample in samples:
        if isinstance(sample, dict):
            errors.extend(sample_errors(sample, root))
        else:
            errors.append("Invalid sample record")

    generic_count = sum(
        1
        for topic in CONFIG["topics"]
        for index, _ in enumerate(CONFIG["seeds"])
        if f"{topic['id']}-{index}" in baseline_images
    )
    return {
        "ready": not errors,
        "mode": "fan-live",
        "catalog_id": catalog_id,
        "errors": errors,
        "fixed_images": len(card_images) if isinstance(card_images, dict) else 0,
        "generic_images": generic_count,
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


def write_preflight(path, *, include_models=False, catalog_id=None):
    """Evaluate and persist the same fresh preflight result returned to callers."""
    result = (
        check_assets() if catalog_id is None else check_assets(catalog_id=catalog_id)
    )
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
    if not isinstance(sample, dict):
        return ["Invalid sample record"]
    raw_id = sample.get("id")
    sid = raw_id if isinstance(raw_id, str) and raw_id else "unknown"

    def fail(message):
        errors.append(f"Sample {sid}: {message}")

    if sid == "unknown":
        fail("invalid id")
    topic_id = sample.get("topic_id")
    topic = next(
        (
            topic
            for topic in CONFIG["topics"]
            if isinstance(topic_id, str) and topic["id"] == topic_id
        ),
        None,
    )
    if not topic:
        fail("unknown topic")

    selection = sample.get("selection")
    raw_personalization = sample.get("personalization")
    if not isinstance(raw_personalization, dict):
        fail("invalid personalization")
        personalization = {}
    else:
        personalization = raw_personalization
    try:
        rebuilt = build_legacy_personalization(selection)
    except (AttributeError, KeyError, TypeError, ValueError):
        rebuilt = None
        fail("selection is unusable")
    if not rebuilt or rebuilt["hash"] != personalization.get("hash"):
        fail("personalization mismatch")

    images = sample.get("images")
    if not isinstance(images, list):
        fail("invalid images")
        return errors
    valid_images = []
    for image in images:
        if not isinstance(image, dict):
            fail("invalid image record")
        else:
            valid_images.append(image)
    if (
        len(images) != 4
        or [image.get("seed") for image in valid_images] != CONFIG["seeds"]
    ):
        fail("seed order or image count")

    prompt = target_prompt(topic) if topic else None
    for image in valid_images:
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
        except (KeyError, OSError, TypeError, ValueError):
            fail("image missing")
    return errors
