"""Validate the fixed asset bundle before an offline exhibition."""

import json
from pathlib import Path

from .config import ASSETS, CONFIG, GPU_PYTHON
from .domain import digest, file_hash


def check_assets(root=ASSETS, *, require_samples=True):
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
    expected = [key for p in CONFIG["pairs"] for key in p["image_ids"]] + [
        f"{t['id']}-{i}" for t in CONFIG["topics"] for i in range(4)
    ]

    def check_image(key, image):
        try:
            path = (root / image["path"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or file_hash(path) != image["sha256"]
            ):
                raise ValueError("hash mismatch")
        except (KeyError, OSError, ValueError):
            errors.append(f"Missing or corrupt image: {key}")

    for key in expected:
        check_image(key, manifest.get("images", {}).get(key, {}))
    prompts = read("generic-prompts.json", {})
    for topic in CONFIG["topics"]:
        rewrite = prompts.get(topic["id"], {})
        if not rewrite.get("valid") or rewrite.get("model") != CONFIG["llm"]:
            errors.append(f"Rewrite revision mismatch: {topic['id']}")
        for i, seed in enumerate(CONFIG["seeds"]):
            image = manifest.get("images", {}).get(f"{topic['id']}-{i}", {})
            if (
                image.get("seed") != seed
                or image.get("prompt") != rewrite.get("prompt")
                or image.get("settings") != CONFIG["generation"]
            ):
                errors.append(f"Generic contract mismatch: {topic['id']}-{i}")
    evidence = read("evidence.json", [])
    if len(evidence) != 10:
        errors.append("Ten directed evidence records required")
    for fragment in evidence:
        if not fragment.get("reviewed") or fragment.get("source") != CONFIG["llm"]:
            errors.append(f"Unreviewed evidence: {fragment.get('id')}")
        for key, h in fragment.get("image_hashes", {}).items():
            if manifest.get("images", {}).get(key, {}).get("sha256") != h:
                errors.append(f"Stale evidence: {key}")
    samples = read("samples.json", []) if require_samples else []
    if require_samples and len(samples) != 6:
        errors.append("Six sample experiences required")
    for sample in samples:
        errors.extend(sample_errors(sample, root))
    return {
        "ready": not errors,
        "mode": "zipp-style-manual",
        "errors": errors,
        "fixed_images": len(manifest.get("images", {})),
        "evidence_directions": len(evidence),
        "samples": len(samples),
        "recommendation_enabled": False,
    }


def check_models():
    errors = []
    models = []
    cache = Path.home() / ".cache/huggingface/hub"
    for name in ("llm", "generation"):
        c = CONFIG[name]
        path = (
            cache
            / ("models--" + c["model"].replace("/", "--"))
            / "snapshots"
            / c["revision"]
        )
        files = list(path.rglob("*.safetensors"))
        if not files or not all(p.exists() for p in files):
            errors.append(f"Missing pinned model: {c['model']}")
        models.append(
            {"model": c["model"], "revision": c["revision"], "weight_files": len(files)}
        )
    if not Path(GPU_PYTHON).is_file():
        errors.append("GPU Python environment is missing")
    return {
        "ready": not errors,
        "errors": errors,
        "models": models,
        "gpu_python": GPU_PYTHON,
    }


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
    rewrite = sample.get("rewrite", {})
    if not rewrite.get("valid") or rewrite.get("model") != CONFIG["llm"]:
        fail("rewrite revision or validity")
    if not topic or not rewrite.get("prompt", "").startswith(topic["basic_prompt_en"]):
        fail("basic prompt mismatch")
    lengths = rewrite.get("token_lengths", [])
    if len(lengths) != 2 or any(n > 77 for n in lengths):
        fail("token lengths")
    images = sample.get("images", [])
    if len(images) != 4 or [x.get("seed") for x in images] != CONFIG["seeds"]:
        fail("seed order or image count")
    for image in images:
        if image.get("settings") != CONFIG["generation"] or image.get(
            "prompt"
        ) != rewrite.get("prompt"):
            fail("generation settings or prompt")
        try:
            path = (root / image["path"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or file_hash(path) != image["sha256"]
            ):
                fail("image hash")
        except (KeyError, OSError):
            fail("image missing")
    context = sample.get("context", {})
    if any(image.get("context_hash") != context.get("hash") for image in images):
        fail("image context mismatch")
    if digest({k: v for k, v in context.items() if k != "hash"}) != context.get(
        "hash"
    ) or rewrite.get("context_hash") != context.get("hash"):
        fail("context mismatch")
    return errors
