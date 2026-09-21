"""The FAN generate stage, with torch/diffusers/fan replaced by recorders."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from exhibit.config import CONFIG, FAN_POLICIES
from exhibit.fan_adapter import resolve_policy, thaw_policy

from exhibit import workers

SETTINGS = {
    "model": "demo/weights",
    "revision": "weights-revision",
    "checkpoint": "v2.safetensors",
    "pipeline_config": {"model": "demo/config", "revision": "config-revision"},
    "scheduler": "DPMSolverMultistepScheduler",
    "scheduler_kwargs": {
        "algorithm_type": "sde-dpmsolver++",
        "use_karras_sigmas": True,
    },
    "vae": {"model": "demo/vae-fp16-fix", "revision": "vae-revision"},
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
    calls = {
        "pipeline": [],
        "encode": [],
        "fan": [],
        "wrapper": [],
        "downloads": [],
        "vae": [],
        "scheduler": [],
    }

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
            self.vae = "checkpoint-vae"
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

    def load_vae(model, **kwargs):
        calls["vae"].append((model, kwargs))
        return SimpleNamespace(to=lambda device: f"{model}@{device}")

    def scheduler_factory(name):
        def from_config(config, **kwargs):
            calls["scheduler"].append((name, kwargs))
            return SimpleNamespace(config={**config, "name": name, **kwargs})

        return SimpleNamespace(from_config=from_config)

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            AutoencoderKL=SimpleNamespace(from_pretrained=load_vae),
            StableDiffusionXLPipeline=SimpleNamespace(
                from_single_file=lambda *args, **kwargs: pipeline
            ),
            DPMSolverMultistepScheduler=scheduler_factory(
                "DPMSolverMultistepScheduler"
            ),
            EulerAncestralDiscreteScheduler=scheduler_factory(
                "EulerAncestralDiscreteScheduler"
            ),
        ),
    )

    def FAN(model, processor, decoder=None):
        calls["fan"].append((model, processor, decoder))
        return SimpleNamespace(model=model, decoder=decoder)

    model_module = ModuleType("fan.model")
    model_module.sample_reference = lambda *args, **kwargs: None

    def stable_diffusion_xl(large, bigG):
        calls["wrapper"].append((large, bigG))

        def encoder(prompt, ref_prompt=None, **kwargs):
            calls["encode"].append({"prompt": prompt, "refs": ref_prompt, **kwargs})
            if ref_prompt and kwargs.get("sample_size"):
                for _ in range(3):
                    model_module.sample_reference(
                        "target", "context", kwargs.get("weight")
                    )
            suffix = "-personalized" if ref_prompt else ""
            return Tensor("cond" + suffix), Tensor("pooled" + suffix)

        encoder._fan_model_module = model_module
        return encoder

    fan_module = ModuleType("fan")
    fan_module.FAN = FAN
    wrapper_module = ModuleType("fan.wrapper")
    wrapper_module.stable_diffusion_xl = stable_diffusion_xl
    fan_module.wrapper = wrapper_module
    monkeypatch.setitem(sys.modules, "fan", fan_module)
    monkeypatch.setitem(sys.modules, "fan.wrapper", wrapper_module)
    monkeypatch.setitem(sys.modules, "fan.model", model_module)

    events = []
    monkeypatch.setattr(
        workers, "emit", lambda kind, **data: events.append((kind, data))
    )
    calls["events"] = events
    calls["upstream"] = tmp_path / "upstream"
    calls["sdxl"] = pipeline
    return calls


def personalization(alpha=0.5):
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

    negative, plain, personal, personal_plain = stubs["encode"]
    assert negative["prompt"] == SETTINGS["negative_prompt"]
    assert negative["refs"] is None and negative["weight"] is None
    assert plain["prompt"] == "masterpiece, 1girl, absurdres, highres."
    assert plain["refs"] is None and plain["alpha"] is None
    assert personal["prompt"] == plain["prompt"]
    assert personal["refs"] == ["warm color palette", "cool color palette"]
    assert personal["weight"] == [2.0, 1.0]
    assert personal["alpha"] == 0.5
    assert personal_plain["prompt"] == personal["prompt"]
    assert personal_plain["refs"] is None and personal_plain["alpha"] is None
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
    assert loaded["fan"] == {"commit": CONFIG["fan"]["commit"]}
    assert loaded["load_seconds"] >= 0
    plain, personal = (data for kind, data in stubs["events"] if kind == "image")
    assert plain["personalization_hash"] is None
    assert personal["personalization_hash"] == "personalization-hash"
    assert plain["negative_prompt"] == SETTINGS["negative_prompt"]
    assert plain["pooled"] == personal["pooled"] == "plain"
    assert plain["fan"]["pooled"] == personal["fan"]["pooled"] == "plain"
    assert (
        plain["fan"]["commit"] == personal["fan"]["commit"] == loaded["fan"]["commit"]
    )
    assert plain["policy_id"] == personal["policy_id"] == "legacy_exhibit"
    assert plain["policy_hash"] == personal["policy_hash"]
    assert plain["policy_hash"] == workers.digest(plain["effective_policy"])
    for event in (plain, personal):
        trace_path = Path(event["profiling_trace_path"])
        assert trace_path.is_file()
        assert json.loads(trace_path.read_text()) == event["profiling_trace"]
    assert Path(plain["path"]).read_bytes() == b"fake png"
    assert plain["sha256"] == personal["sha256"]
    assert not list(tmp_path.glob("*.tmp.png"))


def test_the_pinned_checkpoint_loads_without_a_complete_hub_snapshot(stubs, tmp_path):
    workers.generate({**request_for(tmp_path, stubs["upstream"]), "items": []})
    assert stubs["downloads"] == [
        ("demo/weights", "v2.safetensors", "weights-revision"),
        ("demo/config", "model_index.json", "config-revision"),
    ]


def test_the_worker_module_stays_python_3_10_compatible():
    """The FAN environment resolves to Python 3.10, not the exhibit 3.12 venv."""
    import ast

    from exhibit.config import ROOT

    for name in (
        "workers.py",
        "evaluation_worker.py",
        "fan_adapter.py",
        "gpu.py",
        "config.py",
        "domain.py",
    ):
        source = (ROOT / "src/exhibit" / name).read_text()
        ast.parse(source, filename=name, feature_version=(3, 10))
    script = ROOT / "scripts/evaluate_fan.py"
    ast.parse(script.read_text(), filename=script.name, feature_version=(3, 10))


def test_the_pinned_fp16_fix_vae_replaces_the_checkpoint_decoder(stubs, tmp_path):
    """The bundled VAE decodes washed out in fp16, so the decoder is pinned."""
    workers.generate({**request_for(tmp_path, stubs["upstream"]), "items": []})

    assert stubs["vae"] == [
        (
            "demo/vae-fp16-fix",
            {
                "revision": "vae-revision",
                "torch_dtype": "fp16",
                "use_safetensors": True,
                "local_files_only": True,
            },
        )
    ]
    assert stubs["sdxl"].vae == "demo/vae-fp16-fix@cuda"
    assert stubs["events"][0][1]["vae"] == SETTINGS["vae"]


def test_a_settings_block_without_a_vae_keeps_the_checkpoint_decoder(stubs, tmp_path):
    settings = {k: v for k, v in SETTINGS.items() if k != "vae"}
    request = {**request_for(tmp_path, stubs["upstream"]), "settings": settings}
    workers.generate({**request, "items": []})

    assert stubs["vae"] == []
    assert stubs["sdxl"].vae == "checkpoint-vae"
    assert stubs["events"][0][1]["vae"] is None


def test_the_scheduler_comes_from_the_configuration(stubs, tmp_path):
    """DPM++ 2M SDE Karras was measured sharper than Euler a with the fixed VAE."""
    workers.generate({**request_for(tmp_path, stubs["upstream"]), "items": []})

    assert stubs["scheduler"] == [
        (
            "DPMSolverMultistepScheduler",
            {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
        )
    ]
    loaded = stubs["events"][0][1]["scheduler"]
    assert loaded["name"] == "DPMSolverMultistepScheduler"
    assert loaded["kwargs"] == SETTINGS["scheduler_kwargs"]
    assert loaded["config"]["name"] == "DPMSolverMultistepScheduler"


def test_settings_without_scheduler_kwargs_still_load(stubs, tmp_path):
    settings = {
        k: v for k, v in SETTINGS.items() if k not in ("scheduler", "scheduler_kwargs")
    }
    request = {**request_for(tmp_path, stubs["upstream"]), "settings": settings}
    workers.generate({**request, "items": []})

    assert stubs["scheduler"] == [("EulerAncestralDiscreteScheduler", {})]
    assert stubs["events"][0][1]["scheduler"]["kwargs"] == {}


def test_an_unknown_scheduler_fails_at_load_time(stubs, tmp_path):
    settings = {**SETTINGS, "scheduler": "NoSuchScheduler"}
    request = {**request_for(tmp_path, stubs["upstream"]), "settings": settings}
    with pytest.raises(ValueError, match="Unknown scheduler: NoSuchScheduler"):
        workers.generate(request)
    assert stubs["events"] == []


def test_official_policy_keeps_fan_pooled_and_emits_trace(stubs, tmp_path):
    effective = thaw_policy(resolve_policy("official_encoder", FAN_POLICIES))
    personal = {
        **personalization(alpha=effective["alpha"]),
        "effective_policy": effective,
        "policy_hash": workers.digest(effective),
        "personalization_hash": "official-personalization-hash",
    }
    request = request_for(tmp_path, stubs["upstream"])
    request["items"] = [{**request["items"][1], "personalization": personal}]

    workers.generate(request)

    pipeline_call = stubs["pipeline"][0]
    assert pipeline_call["prompt_embeds"].name == "cond-personalized"
    assert pipeline_call["pooled_prompt_embeds"].name == "pooled-personalized"
    negative, positive = stubs["encode"]
    assert negative["refs"] is None
    assert positive["refs"] == ["warm color palette", "cool color palette"]
    assert positive["sample_size"] == 0.1
    assert positive["skip_pa"] == [0]

    image_event = next(data for kind, data in stubs["events"] if kind == "image")
    assert image_event["pooled"] == "fan"
    assert image_event["effective_policy"] == effective
    assert image_event["policy_id"] == "official_encoder"
    assert image_event["policy_hash"] == workers.digest(effective)
    assert image_event["personalization_hash"] == "official-personalization-hash"
    assert image_event["fan"] == {
        "commit": CONFIG["fan"]["commit"],
        "pooled": "fan",
        "skip": -2,
        "sample_size": 0.1,
        "skip_pa": [0],
        "use_attn_mask": False,
    }
    trace_path = Path(image_event["profiling_trace_path"])
    assert trace_path == Path(request["items"][0]["path"]).with_suffix(".trace.json")
    assert json.loads(trace_path.read_text()) == image_event["profiling_trace"]
    assert [call["phase"] for call in image_event["profiling_trace"]["calls"]] == [
        "clip_l_hidden",
        "clip_g_hidden",
        "clip_g_pool",
    ]


def test_plain_policy_gets_pooled_from_a_reference_free_positive_call(stubs, tmp_path):
    effective = thaw_policy(resolve_policy("legacy_exhibit", FAN_POLICIES))
    personal = {
        **personalization(alpha=effective["alpha"]),
        "effective_policy": effective,
        "policy_id": "legacy_exhibit",
        "policy_hash": workers.digest(effective),
        "personalization_hash": "legacy-personalization-hash",
    }
    request = request_for(tmp_path, stubs["upstream"])
    request["items"] = [{**request["items"][1], "personalization": personal}]

    workers.generate(request)

    negative, personalized, positive_plain = stubs["encode"]
    assert negative["refs"] is None
    assert personalized["refs"] == ["warm color palette", "cool color palette"]
    assert positive_plain["prompt"] == personalized["prompt"]
    assert positive_plain["refs"] is None
    assert stubs["pipeline"][0]["prompt_embeds"].name == "cond-personalized"
    assert stubs["pipeline"][0]["pooled_prompt_embeds"].name == "pooled"


def test_generate_rejects_a_policy_hash_that_conflicts_with_effective_policy(
    stubs, tmp_path
):
    effective = thaw_policy(resolve_policy("official_encoder", FAN_POLICIES))
    personal = {
        **personalization(alpha=effective["alpha"]),
        "policy_id": "official_encoder",
        "effective_policy": effective,
        "policy_hash": "conflicting-hash",
        "personalization_hash": "official-personalization-hash",
    }
    request = request_for(tmp_path, stubs["upstream"])
    request["items"] = [{**request["items"][1], "personalization": personal}]

    with pytest.raises(ValueError, match="policy_hash"):
        workers.generate(request)

    assert stubs["pipeline"] == []


def test_evaluation_can_observe_conditioning_without_duplicating_generation(
    stubs, tmp_path
):
    effective = thaw_policy(resolve_policy("official_encoder", FAN_POLICIES))
    personal = {
        **personalization(alpha=effective["alpha"]),
        "effective_policy": effective,
        "policy_hash": workers.digest(effective),
        "personalization_hash": "observed-personalization",
    }
    request = request_for(tmp_path, stubs["upstream"])
    request["items"] = [{**request["items"][1], "personalization": personal}]
    observed = []
    forwarded = []

    workers.generate(
        request,
        event_sink=lambda kind, **data: forwarded.append((kind, data)),
        conditioning_sink=lambda item_id, candidate, plain: (
            observed.append((item_id, candidate, plain))
            or {
                "hidden": {"valid": True, "cosine": 0.75},
                "pooled": {"valid": True, "cosine": 0.5},
            }
        ),
    )

    assert len(stubs["pipeline"]) == 1
    assert len(observed) == 1
    item_id, candidate, plain = observed[0]
    assert item_id == "v0-0"
    assert candidate["hidden"].name == "cond-personalized"
    assert plain["hidden"].name == "cond"
    image = next(data for kind, data in forwarded if kind == "image")
    assert image["conditioning_target_align"] == {
        "hidden": {"valid": True, "cosine": 0.75},
        "pooled": {"valid": True, "cosine": 0.5},
    }
    assert stubs["events"] == []
