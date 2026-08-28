import ast
import unittest
from pathlib import Path


class StaticNodeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[2].joinpath("nodes.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_portrait_inputs_are_declared(self):
        self.assertIn('"portrait_mode"', self.source)
        self.assertIn('"silhouette_guard"', self.source)
        self.assertIn('"subject_mask"', self.source)

    def test_portrait_metadata_is_passed_through(self):
        self.assertIn("portrait_result=getattr(layers, 'portrait_result', None)", self.source)
        self.assertIn('"portrait_result": portrait_result', self.source)

    def test_source_loader_exposes_foreground_mask(self):
        self.assertIn('RETURN_NAMES = ("image", "mask", "source_filename", "subject_mask")', self.source)


if __name__ == "__main__":
    unittest.main()
