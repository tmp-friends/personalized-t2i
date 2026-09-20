"""The offline loader must work with only inference resources cached."""

import sys
from pathlib import Path
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


def test_generate_uses_sdxl_default_hidden_state_instead_of_extra_diffusers_skip(
    monkeypatch, tmp_path
):
    settings = {
        "model": "demo/weights",
        "revision": "weights-revision",
        "checkpoint": "v2.safetensors",
        "pipeline_config": {"model": "demo/config", "revision": "config-revision"},
        "negative_prompt": "lowres, worst quality, bad hands",
        "steps": 28,
        "guidance_scale": 5.0,
        "width": 1024,
        "height": 1024,
    }
    calls = []

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeImage:
        def save(self, path):
            Path(path).write_bytes(b"fake png")

    class FakeTokenizer:
        model_max_length = 77

        def __call__(self, text, truncation):
            return {"input_ids": [1]}

    class FakePipeline:
        def __init__(self):
            self.scheduler = SimpleNamespace(config={})
            self.tokenizer = FakeTokenizer()
            self.tokenizer_2 = self.tokenizer

        def to(self, device):
            return self

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(images=[FakeImage()])

    pipeline = FakePipeline()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float16="fp16", Generator=FakeGenerator),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda repo, *, filename, revision, local_files_only: (
                "/cache/config/model_index.json"
                if filename == "model_index.json"
                else "/cache/weights/v2.safetensors"
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            StableDiffusionXLPipeline=SimpleNamespace(
                from_single_file=lambda *args, **kwargs: pipeline
            ),
            EulerAncestralDiscreteScheduler=SimpleNamespace(
                from_config=lambda config: SimpleNamespace(config=config)
            ),
        ),
    )
    monkeypatch.setattr(workers, "emit", lambda *args, **kwargs: None)

    workers.generate(
        {
            "settings": settings,
            "items": [
                {
                    "id": "cat-0",
                    "prompt": "masterpiece, 1girl, absurdres, highres.",
                    "seed": 230923,
                    "path": str(tmp_path / "cat-0.png"),
                }
            ],
        }
    )

    assert len(calls) == 1
    prompt, kwargs = calls[0]
    generator = kwargs.pop("generator")
    assert prompt == "masterpiece, 1girl, absurdres, highres."
    assert kwargs == {
        "negative_prompt": "lowres, worst quality, bad hands",
        "num_inference_steps": 28,
        "guidance_scale": 5.0,
        "width": 1024,
        "height": 1024,
    }
    assert generator.device == "cuda"
    assert generator.seed == 230923
