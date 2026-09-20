from pathlib import Path

import pytest

from exhibit import preflight

DECODERS = {"L.pth": b"large decoder", "bigG.pth": b"bigG decoder"}


@pytest.fixture
def pinned_environment(tmp_path, monkeypatch):
    """A pinned single-file checkpoint plus the FAN runtime it needs."""
    upstream = tmp_path / "upstream"
    (upstream / "fan").mkdir(parents=True)
    (upstream / "fan/model.py").write_text("# FAN")
    (upstream / "weight").mkdir()
    decoders = {}
    for name, payload in DECODERS.items():
        path = upstream / "weight" / name
        path.write_bytes(payload)
        decoders[name] = preflight.file_hash(path)
    settings = {
        "generation": {
            "model": "demo/illustration",
            "revision": "weight-revision",
            "checkpoint": "illustration.safetensors",
            "pipeline_config": {"model": "demo/config", "revision": "config-revision"},
        },
        "fan": {
            "python": "unused",
            "upstream": str(upstream),
            "commit": "0" * 40,
            "decoders": decoders,
            "skip": -2,
            "sample_size": 0,
        },
    }
    cache = tmp_path / ".cache/huggingface/hub"
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
    files = [
        "models--demo--illustration/snapshots/weight-revision/illustration.safetensors"
    ] + [
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
    monkeypatch.setattr(preflight, "FAN_UPSTREAM", upstream)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return {"cache": cache, "upstream": upstream, "python": gpu_python}


def test_pinned_checkpoint_and_fan_runtime_are_ready(pinned_environment):
    result = preflight.check_models()
    assert result["ready"], result["errors"]
    assert result["fan"]["decoders"].keys() == DECODERS.keys()


def test_single_file_model_requires_the_exact_pinned_checkpoint(pinned_environment):
    checkpoint = (
        pinned_environment["cache"]
        / "models--demo--illustration/snapshots/weight-revision/illustration.safetensors"
    )
    checkpoint.rename(checkpoint.with_name("unrelated.safetensors"))
    result = preflight.check_models()
    assert not result["ready"]
    assert any("checkpoint" in error.lower() for error in result["errors"])


@pytest.mark.parametrize(
    "missing", ["model_index.json", "unet/config.json", "tokenizer_2/merges.txt"]
)
def test_single_file_model_requires_offline_configuration(pinned_environment, missing):
    (
        pinned_environment["cache"]
        / "models--demo--config/snapshots/config-revision"
        / missing
    ).unlink()
    result = preflight.check_models()
    assert not result["ready"]
    assert any(missing in error for error in result["errors"])


def test_missing_fan_environment_is_reported(pinned_environment):
    pinned_environment["python"].unlink()
    (pinned_environment["upstream"] / "fan/model.py").unlink()
    result = preflight.check_models()
    assert not result["ready"]
    assert any("FAN Python environment" in error for error in result["errors"])
    assert any("FAN upstream is missing" in error for error in result["errors"])


@pytest.mark.parametrize("name", list(DECODERS))
def test_decoder_weights_must_match_the_pinned_hashes(pinned_environment, name):
    (pinned_environment["upstream"] / "weight" / name).write_bytes(b"retrained")
    result = preflight.check_models()
    assert not result["ready"]
    assert f"Decoder weight mismatch: {name}" in result["errors"]


@pytest.mark.parametrize("name", list(DECODERS))
def test_missing_decoder_weights_are_reported(pinned_environment, name):
    (pinned_environment["upstream"] / "weight" / name).unlink()
    result = preflight.check_models()
    assert not result["ready"]
    assert f"Missing decoder weight: {name}" in result["errors"]


def test_the_shipped_configuration_pins_the_real_decoder_hashes():
    """The contract in configs/demo.json must describe the checked-out weights."""
    from exhibit.config import CONFIG, FAN_UPSTREAM

    if not (FAN_UPSTREAM / "fan/model.py").is_file():
        pytest.skip("FAN upstream has not been prepared in this checkout")
    result = preflight.check_fan_env(gpu_python=__file__)
    assert result["decoders"] == CONFIG["fan"]["decoders"]
