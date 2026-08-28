import unittest

import numpy as np

from portrait_core import PortraitConfig, resolve_subject_mask
from tests.unit.helpers import portrait_subject, rgba


class SubjectMaskTests(unittest.TestCase):
    def setUp(self):
        self.config = PortraitConfig.load()

    def test_informative_source_alpha_wins(self):
        subject = portrait_subject()
        evidence = resolve_subject_mask(rgba(subject), config=self.config)
        self.assertEqual(evidence.source, "source_alpha")
        self.assertEqual(evidence.confidence, "HIGH")
        np.testing.assert_array_equal(evidence.binary, subject > 0.062745)

    def test_opaque_image_uses_generated_union_with_low_confidence(self):
        original = rgba(np.ones((32, 32)))
        layer = rgba(portrait_subject())
        evidence = resolve_subject_mask(original, generated_layers={"topwear": layer}, config=self.config)
        self.assertEqual(evidence.source, "fallback_union")
        self.assertEqual(evidence.confidence, "LOW")

    def test_opaque_image_uses_provided_mask(self):
        original = rgba(np.ones((32, 32)))
        subject = portrait_subject()
        evidence = resolve_subject_mask(original, provided_mask=subject, config=self.config)
        self.assertEqual(evidence.source, "provided_mask")
        self.assertEqual(evidence.confidence, "HIGH")

    def test_empty_inputs_fail(self):
        original = rgba(np.ones((32, 32)))
        with self.assertRaises(ValueError):
            resolve_subject_mask(original, config=self.config)


if __name__ == "__main__":
    unittest.main()
