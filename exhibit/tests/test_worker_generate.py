"""The FAN generate stage, with torch/diffusers/fan replaced by recorders."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from exhibit.config import CONFIG

from exhibit import workers

SETTINGS = {
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


class Tensor:
    def __init__(self, name, dtype="fp32"):
        self.name = name
        self.dtype = dtype

    def to(self, dtype):
        return Tensor(self.name, dtype)


class Image:
    def save(self, path):
        Path(path).write_bytes(b"fake png")


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    """Everything the worker imports lazily, recorded instead of executed."""
    calls = {"pipeline": [], "encode": [], "fan": [], "wrapper": [], "downloads": []}

    class Generator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    @contextmanager
    def no_grad():
        yield

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float16="fp16",
            Generator=Generator,
            no_grad=no_grad,
            set_num_threads=lambda n: None,
            cuda=SimpleNamespace(
                synchronize=lambda: None, max_memory_allocated=lambda: 0
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda repo, *, filename, revision, local_files_only: (
                calls["downloads"].append((repo, filename, revision))
                or (
                    "/cache/config/model_index.json"
                    if filename == "model_index.json"
                    else "/cache/weights/v2.safetensors"
                )
            )
        ),
    )

    class Pipeline:
        def __init__(self):
            self.scheduler = SimpleNamespace(config={})
            self.text_encoder = "large-encoder"
            self.text_encoder_2 = "bigG-encoder"
            self.tokenizer = "large-tokenizer"
            self.tokenizer_2 = "bigG-tokenizer"

        def to(self, device):
            return self

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            calls["pipeline"].append(kwargs)
            return SimpleNamespace(images=[Image()])

    pipeline = Pipeline()
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

    def FAN(model, processor, decoder=None):
        calls["fan"].append((model, processor, decoder))
        return SimpleNamespace(model=model, decoder=decoder)

    def stable_diffusion_xl(large, bigG):
        calls["wrapper"].append((large, bigG))

        def encoder(prompt, ref_prompt=None, **kwargs):
            calls["encode"].append({"prompt": prompt, "refs": ref_prompt, **kwargs})
            suffix = "-personalized" if ref_prompt else ""
            return Tensor("cond" + suffix), Tensor("pooled" + suffix)

        return encoder

    fan_module = ModuleType("fan")
    fan_module.FAN = FAN
    wrapper_module = ModuleType("fan.wrapper")
    wrapper_module.stable_diffusion_xl = stable_diffusion_xl
    fan_module.wrapper = wrapper_module
    monkeypatch.setitem(sys.modules, "fan", fan_module)
    monkeypatch.setitem(sys.modules, "fan.wrapper", wrapper_module)

    events = []
    monkeypatch.setattr(
        workers, "emit", lambda kind, **data: events.append((kind, data))
    )
    calls["events"] = events
    calls["upstream"] = tmp_path / "upstream"
    return calls


def personalization(alpha=0.4):
    return {
        "refs": [
            {"card_id": "girl-warm_soft", "text": "warm color palette", "weight": 2.0},
            {
                "card_id": "barista-cool_clean",
                "text": "cool color palette",
                "weight": 1.0,
            },
        ],
        "alpha": alpha,
        "sample_size": 0,
        "hash": "personalization-hash",
    }


def request_for(tmp_path, upstream):
    return {
        "stage": "generate",
        "settings": SETTINGS,
        "upstream": str(upstream),
        "items": [
            {
                "id": "plain-0",
                "prompt": "masterpiece, 1girl, absurdres, highres.",
                "seed": 230923,
                "path": str(tmp_path / "plain-0.png"),
                "personalization": None,
            },
            {
                "id": "v0-0",
                "prompt": "masterpiece, 1girl, absurdres, highres.",
                "seed": 230923,
                "path": str(tmp_path / "v0-0.png"),
                "personalization": personalization(),
            },
        ],
    }


def test_the_encoder_is_built_once_and_only_references_differ(stubs, tmp_path):
    workers.generate(request_for(tmp_path, stubs["upstream"]))

    assert stubs["fan"] == [
        ("large-encoder", "large-tokenizer", str(stubs["upstream"] / "weight/L.pth")),
        ("bigG-encoder", "bigG-tokenizer", str(stubs["upstream"] / "weight/bigG.pth")),
    ]
    assert len(stubs["wrapper"]) == 1

    negative, plain, personal = stubs["encode"]
    assert negative["prompt"] == SETTINGS["negative_prompt"]
    assert negative["refs"] is None and negative["weight"] is None
    assert plain["prompt"] == "masterpiece, 1girl, absurdres, highres."
    assert plain["refs"] is None and plain["alpha"] is None
    assert personal["prompt"] == plain["prompt"]
    assert personal["refs"] == ["warm color palette", "cool color palette"]
    assert personal["weight"] == [2.0, 1.0]
    assert personal["alpha"] == 0.4
    assert len(stubs["encode"]) == 3, "the plain encoding is reused, not recomputed"
    for call in stubs["encode"]:
        assert call["skip"] == CONFIG["fan"]["skip"] == -2
        assert call["sample_size"] == CONFIG["fan"]["sample_size"] == 0
        # Measured settings: personalized attention is skipped in layers 0-7 and
        # the attention mask stays off, for the target and every reference alike.
        assert call["skip_pa"] == [0, 1, 2, 3, 4, 5, 6, 7]
        assert call["use_attn_mask"] is False


def test_every_embedding_reaches_the_pipeline_in_fp16(stubs, tmp_path):
    workers.generate(request_for(tmp_path, stubs["upstream"]))

    assert len(stubs["pipeline"]) == 2
    for kwargs in stubs["pipeline"]:
        embeds = {
            key: kwargs[key]
            for key in (
                "prompt_embeds",
                "pooled_prompt_embeds",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
            )
        }
        assert all(tensor.dtype == "fp16" for tensor in embeds.values())
        # Upstream's personalized pooled token is mis-detected; use the plain one.
        assert embeds["pooled_prompt_embeds"].name == "pooled"
        assert embeds["negative_prompt_embeds"].name == "cond"
        assert embeds["negative_pooled_prompt_embeds"].name == "pooled"
        assert kwargs["num_inference_steps"] == 28
        assert kwargs["guidance_scale"] == 5.0
        assert kwargs["width"] == kwargs["height"] == 1024
        assert kwargs["generator"].device == "cuda"
        assert kwargs["generator"].seed == 230923
        assert "prompt" not in kwargs and "negative_prompt" not in kwargs
    assert stubs["pipeline"][0]["prompt_embeds"].name == "cond"
    assert stubs["pipeline"][1]["prompt_embeds"].name == "cond-personalized"


def test_image_events_carry_the_personalization_of_their_item(stubs, tmp_path):
    workers.generate(request_for(tmp_path, stubs["upstream"]))

    kinds = [kind for kind, _ in stubs["events"]]
    assert kinds == ["loaded", "image", "image"]
    loaded = stubs["events"][0][1]
    assert loaded["fan"] == {
        "commit": CONFIG["fan"]["commit"],
        "pooled": "plain",
        "skip": -2,
        "sample_size": 0,
        "skip_pa": [0, 1, 2, 3, 4, 5, 6, 7],
        "use_attn_mask": False,
    }
    assert loaded["load_seconds"] >= 0
    plain, personal = (data for kind, data in stubs["events"] if kind == "image")
    assert plain["personalization_hash"] is None
    assert personal["personalization_hash"] == "personalization-hash"
    assert plain["negative_prompt"] == SETTINGS["negative_prompt"]
    assert plain["pooled"] == personal["pooled"] == "plain"
    assert plain["fan"] == personal["fan"] == loaded["fan"]
    assert Path(plain["path"]).read_bytes() == b"fake png"
    assert plain["sha256"] == personal["sha256"]
    assert not list(tmp_path.glob("*.tmp.png"))


def test_the_pinned_checkpoint_loads_without_a_complete_hub_snapshot(stubs, tmp_path):
    workers.generate({**request_for(tmp_path, stubs["upstream"]), "items": []})
    assert stubs["downloads"] == [
        ("demo/weights", "v2.safetensors", "weights-revision"),
        ("demo/config", "model_index.json", "config-revision"),
    ]


def test_config_tuning_knobs_reach_the_encoder(stubs):
    """Follow-up probes may add `use_attn_mask`/`skip_pa` without a code change."""
    fan = {**CONFIG["fan"], "use_attn_mask": True, "skip_pa": [0, 1]}

    def encoder(prompt, ref_prompt=None, **kwargs):
        stubs["encode"].append({"prompt": prompt, "refs": ref_prompt, **kwargs})
        return Tensor("cond"), Tensor("pooled")

    workers.encode(encoder, "a prompt", None, fan)
    assert stubs["encode"][-1]["use_attn_mask"] is True
    assert stubs["encode"][-1]["skip_pa"] == [0, 1]
    assert stubs["encode"][-1]["alpha"] is None


def test_the_worker_module_stays_python_3_10_compatible():
    """The FAN environment resolves to Python 3.10, not the exhibit 3.12 venv."""
    import ast

    from exhibit.config import ROOT

    for name in ("workers.py", "config.py", "domain.py"):
        source = (ROOT / "src/exhibit" / name).read_text()
        ast.parse(source, filename=name, feature_version=(3, 10))
