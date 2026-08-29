import unittest

import numpy as np

from portrait_core import PortraitConfig, resolve_subject_mask
from portrait_core.masks import bbox_fill_ratio
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

    def test_pillarboxed_opaque_image_is_not_a_matte(self):
        """The failure this guards against: an opaque picture with transparent
        bars has plenty of transparency, so a transparency-quantity test calls
        its alpha informative. The guard then takes the whole rectangle as the
        subject and recovers the *background* into body_remainder, which
        reports as REWORK about layer quality rather than about the input."""
        alpha = np.zeros((32, 32), dtype=np.float32)
        alpha[:, 8:24] = 1.0  # pillarbox: opaque middle, transparent bars
        with self.assertRaises(ValueError) as ctx:
            resolve_subject_mask(rgba(alpha), config=self.config)
        self.assertIn("bounding box", str(ctx.exception))

    def test_a_provided_mask_still_rescues_a_pillarboxed_image(self):
        """Without this the escape hatch is dead: the source alpha would be
        called informative and the user's mask never consulted."""
        alpha = np.zeros((32, 32), dtype=np.float32)
        alpha[:, 8:24] = 1.0
        evidence = resolve_subject_mask(rgba(alpha), provided_mask=portrait_subject(),
                                        config=self.config)
        self.assertEqual(evidence.source, "provided_mask")
        self.assertEqual(evidence.confidence, "HIGH")
        self.assertTrue(any("padding" in w for w in evidence.warnings), evidence.warnings)

    def test_a_real_silhouette_is_still_informative(self):
        """The rejection must not fire on the shape it exists to protect."""
        evidence = resolve_subject_mask(rgba(portrait_subject()), config=self.config)
        self.assertEqual(evidence.source, "source_alpha")
        self.assertEqual(evidence.warnings, ())

    def test_empty_inputs_fail(self):
        original = rgba(np.ones((32, 32)))
        with self.assertRaises(ValueError):
            resolve_subject_mask(original, config=self.config)


class BboxFillRatioTests(unittest.TestCase):
    def test_a_filled_rectangle_fills_its_box(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[:, 8:24] = True
        self.assertAlmostEqual(bbox_fill_ratio(mask), 1.0)

    def test_a_portrait_silhouette_leaves_its_box_far_from_full(self):
        self.assertLess(bbox_fill_ratio(portrait_subject() > 0.5), 0.97)

    def test_empty_mask_is_zero(self):
        self.assertEqual(bbox_fill_ratio(np.zeros((8, 8), dtype=bool)), 0.0)


if __name__ == "__main__":
    unittest.main()
