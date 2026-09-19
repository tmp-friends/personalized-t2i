import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailored_visions_repro.generate import (  # noqa: E402
    GenConfig,
    SD15_MODEL,
    SDXL_MODEL,
    _enable_vae_slicing,
)


class GenerationConfigTests(unittest.TestCase):
    def test_model_defaults_follow_native_resolution_and_memory_cost(self) -> None:
        cases = [
            (SDXL_MODEL, (1024, 1024, 2)),
            (SD15_MODEL, (512, 512, 8)),
        ]
        for model_name, expected in cases:
            with self.subTest(model_name=model_name):
                config = GenConfig().resolve_for(model_name)
                self.assertEqual(
                    (config.height, config.width, config.batch_size),
                    expected,
                )

    def test_explicit_generation_limits_override_model_defaults(self) -> None:
        config = GenConfig(height=768, width=640, batch_size=3).resolve_for(SDXL_MODEL)
        self.assertEqual((config.height, config.width, config.batch_size), (768, 640, 3))

    def test_vae_slicing_uses_current_diffusers_api(self) -> None:
        class Vae:
            sliced = False

            def enable_slicing(self) -> None:
                self.sliced = True

        class Pipe:
            vae = Vae()

        pipe = Pipe()
        _enable_vae_slicing(pipe)
        self.assertTrue(pipe.vae.sliced)

    def test_vae_slicing_falls_back_to_legacy_pipeline_api(self) -> None:
        class Pipe:
            sliced = False

            def enable_vae_slicing(self) -> None:
                self.sliced = True

        pipe = Pipe()
        _enable_vae_slicing(pipe)
        self.assertTrue(pipe.sliced)


if __name__ == "__main__":
    unittest.main()
