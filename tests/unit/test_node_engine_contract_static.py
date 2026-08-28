import unittest
from pathlib import Path


class NodeEngineDelegationTests(unittest.TestCase):
    """M2: nodes.py must delegate its GPU-facing model-loading and
    single-diffusion-pass logic to seethrough_engine instead of keeping a
    second copy, so the standalone webui and the ComfyUI node graph share one
    implementation. See docs/M2_IMPLEMENTATION_SPEC.md."""

    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[2].joinpath("nodes.py").read_text(encoding="utf-8")

    def test_imports_seethrough_engine(self):
        self.assertIn("seethrough_engine import model_loading", self.source)
        self.assertIn("seethrough_engine import generation", self.source)

    def test_model_loading_is_delegated(self):
        self.assertIn("st_model_loading.resolve_model_path(", self.source)
        self.assertIn("st_model_loading.load_layerdiff_model(", self.source)
        self.assertIn("st_model_loading.load_depth_model(", self.source)
        self.assertIn("st_model_loading.scan_model_dirs(", self.source)

    def test_generate_layers_custom_diffusion_call_is_delegated(self):
        self.assertIn("st_generation.run_diffusion_stage(", self.source)
        self.assertIn("st_generation.layer_similarity(", self.source)


if __name__ == "__main__":
    unittest.main()
