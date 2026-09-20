"""The offline loader must work with only inference resources cached."""

import sys
from types import SimpleNamespace

from exhibit import workers


def test_generate_loads_pinned_checkpoint_without_a_complete_hub_snapshot(monkeypatch):
    settings = {
        "model": "demo/weights",
        "revision": "weights-revision",
        "checkpoint": "v2.safetensors",
        "pipeline_config": {"model": "demo/config", "revision": "config-revision"},
    }
    requested = []

    def cached_file(repo, *, filename, revision, local_files_only):
        assert local_files_only is True
        requested.append((repo, filename, revision))
        return {
            (
                "demo/weights",
                "v2.safetensors",
                "weights-revision",
            ): "/cache/weights/v2.safetensors",
            (
                "demo/config",
                "model_index.json",
                "config-revision",
            ): "/cache/config/model_index.json",
        }[(repo, filename, revision)]

    def incomplete_snapshot(*args, **kwargs):
        raise RuntimeError("Optional repository PDFs are not cached")

    pipeline = SimpleNamespace(scheduler=SimpleNamespace(config={}))
    pipeline.to = lambda device: pipeline
    pipeline.set_progress_bar_config = lambda **kwargs: None

    def load_checkpoint(path, *, config, torch_dtype, local_files_only):
        assert path == "/cache/weights/v2.safetensors"
        assert str(config) == "/cache/config"
        assert torch_dtype == "fp16" and local_files_only is True
        return pipeline

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="fp16"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=cached_file,
            snapshot_download=incomplete_snapshot,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            StableDiffusionXLPipeline=SimpleNamespace(from_single_file=load_checkpoint),
            EulerAncestralDiscreteScheduler=SimpleNamespace(
                from_config=lambda config: SimpleNamespace(config=config)
            ),
        ),
    )
    events = []
    monkeypatch.setattr(workers, "emit", lambda kind, **data: events.append(kind))
    workers.generate({"settings": settings, "items": []})
    assert events == ["loaded"]
    assert requested == [
        ("demo/weights", "v2.safetensors", "weights-revision"),
        ("demo/config", "model_index.json", "config-revision"),
    ]
