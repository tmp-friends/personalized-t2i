from pathlib import Path

import pytest

from exhibit import preflight


@pytest.fixture
def single_file_cache(tmp_path, monkeypatch):
    settings = {
        "llm": {"model": "demo/llm", "revision": "llm-revision"},
        "generation": {
            "model": "demo/illustration",
            "revision": "weight-revision",
            "checkpoint": "illustration.safetensors",
            "pipeline_config": {"model": "demo/config", "revision": "config-revision"},
        },
    }
    cache = tmp_path / ".cache/huggingface/hub"
    files = [
        "models--demo--llm/snapshots/llm-revision/model.safetensors",
        "models--demo--illustration/snapshots/weight-revision/illustration.safetensors",
    ]
    config_files = [
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
    files += [
        f"models--demo--config/snapshots/config-revision/{name}"
        for name in config_files
    ]
    for name in files:
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    gpu_python = tmp_path / "python"
    gpu_python.touch()
    monkeypatch.setattr(preflight, "CONFIG", settings)
    monkeypatch.setattr(preflight, "GPU_PYTHON", str(gpu_python))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return cache


def test_single_file_model_with_local_pipeline_config_is_ready(single_file_cache):
    assert preflight.check_models()["ready"]


def test_single_file_model_requires_the_exact_pinned_checkpoint(single_file_cache):
    checkpoint = (
        single_file_cache
        / "models--demo--illustration/snapshots/weight-revision/illustration.safetensors"
    )
    checkpoint.rename(checkpoint.with_name("unrelated.safetensors"))
    result = preflight.check_models()
    assert not result["ready"]
    assert any("checkpoint" in error.lower() for error in result["errors"])


@pytest.mark.parametrize(
    "missing", ["model_index.json", "unet/config.json", "tokenizer_2/merges.txt"]
)
def test_single_file_model_requires_offline_configuration(single_file_cache, missing):
    (
        single_file_cache / "models--demo--config/snapshots/config-revision" / missing
    ).unlink()
    result = preflight.check_models()
    assert not result["ready"]
    assert any(missing in error for error in result["errors"])
