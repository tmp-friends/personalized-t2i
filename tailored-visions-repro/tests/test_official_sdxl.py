import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "d0f4454ca08c68c5d30f08a01ff4a23a8b33b610"


class FakeVae:
    def __init__(self) -> None:
        self.sliced = False

    def enable_slicing(self) -> None:
        self.sliced = True


class FakePipeline:
    def __init__(self, model_id: str | None = None) -> None:
        self.vae = FakeVae()
        self.config = types.SimpleNamespace(_name_or_path=model_id)
        self.batch_sizes: list[int] = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, prompt, num_images_per_prompt, **kwargs):
        self.batch_sizes.append(num_images_per_prompt)
        return types.SimpleNamespace(images=[object()] * num_images_per_prompt)


class FakeAutoPipeline:
    requested_models: list[str] = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.requested_models.append(model_id)
        return FakePipeline(model_id)


def apply_patch_series(destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT / "upstream"), str(destination)],
        check=True,
    )
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", EXPECTED], check=True)
    names = [
        line.strip()
        for line in (ROOT / "patches/series").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for name in names:
        patch = ROOT / "patches" / name
        subprocess.run(["git", "-C", str(destination), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(destination), "apply", str(patch)], check=True)


def load_official_sd(path: Path):
    torch = types.ModuleType("torch")
    torch.float16 = object()
    torch.no_grad = lambda: (lambda function: function)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        mem_get_info=lambda: (24 * 1024**3, 24 * 1024**3),
    )
    diffusers = types.ModuleType("diffusers")
    diffusers.AutoPipelineForText2Image = FakeAutoPipeline
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace()

    spec = importlib.util.spec_from_file_location("tailored_official_sd", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"torch": torch, "diffusers": diffusers, "PIL": pil},
    ), mock.patch.dict(os.environ, {}, clear=True):
        spec.loader.exec_module(module)
    return module


class OfficialSdxlTests(unittest.TestCase):
    def test_loader_requests_model_and_uses_its_default_batch(self) -> None:
        cases = [
            (None, "stabilityai/stable-diffusion-xl-base-1.0", 2),
            ("stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-xl-base-1.0", 2),
            ("stable-diffusion-v1-5/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5", 4),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "upstream"
            apply_patch_series(destination)
            module = load_official_sd(destination / "SD.py")
            for requested, expected_model, expected_batch in cases:
                with self.subTest(requested=requested), mock.patch.dict(os.environ, {}, clear=True):
                    FakeAutoPipeline.requested_models.clear()
                    pipe = module.load_pipeline(requested, device="cpu")
                    images = module.text2img(pipe, "an astronaut riding a horse")
                    self.assertEqual(FakeAutoPipeline.requested_models, [expected_model])
                    self.assertEqual(len(images), expected_batch)
                    self.assertEqual(pipe.batch_sizes, [expected_batch])

    def test_direct_pipeline_uses_configured_model_or_legacy_fallback(self) -> None:
        cases = [
            (types.SimpleNamespace(_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"), 2),
            ({"_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5"}, 4),
            (types.SimpleNamespace(), 4),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "upstream"
            apply_patch_series(destination)
            module = load_official_sd(destination / "SD.py")
            for config, expected_batch in cases:
                with self.subTest(config=config), mock.patch.dict(os.environ, {}, clear=True):
                    pipe = FakePipeline()
                    pipe.config = config
                    images = module.text2img(pipe, "an astronaut riding a horse")
                    self.assertEqual(len(images), expected_batch)
                    self.assertEqual(pipe.batch_sizes, [expected_batch])


if __name__ == "__main__":
    unittest.main()
