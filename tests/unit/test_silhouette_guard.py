import unittest

import numpy as np

from portrait_core import PortraitConfig, apply_silhouette_guard, resolve_subject_mask
from tests.unit.helpers import portrait_subject, rgba


class SilhouetteGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = PortraitConfig.load()

    def test_partial_arm_omission_is_recovered(self):
        subject = portrait_subject()
        semantic = subject.copy()
        semantic[12:29, 5:9] = 0
        semantic[12:29, 23:27] = 0
        original = rgba(subject)
        evidence = resolve_subject_mask(original, config=self.config)
        result = apply_silhouette_guard(original, {"topwear": rgba(semantic)}, evidence, self.config)
        self.assertGreater(result.metrics.missing_ratio, 0)
        self.assertGreater(result.metrics.recovered_ratio, 0)
        self.assertGreaterEqual(result.metrics.post_recovery_coverage, 0.995)
        self.assertTrue(np.any(result.body_remainder[12:29, 5:9, 3]))

    def test_spill_is_clipped(self):
        subject = portrait_subject()
        spilling = subject.copy()
        spilling[:, :3] = 1
        original = rgba(subject)
        evidence = resolve_subject_mask(original, config=self.config)
        result = apply_silhouette_guard(original, {"topwear": rgba(spilling)}, evidence, self.config)
        self.assertGreater(result.metrics.spill_ratio, 0)
        self.assertLessEqual(result.metrics.post_spill_ratio, 0.002)
        self.assertFalse(np.any(result.guarded_layers["topwear"][:, :3, 3]))

    def test_low_confidence_mask_cannot_hard_pass(self):
        original = rgba(np.ones((32, 32)))
        layer = rgba(portrait_subject())
        evidence = resolve_subject_mask(original, generated_layers={"topwear": layer}, config=self.config)
        result = apply_silhouette_guard(original, {"topwear": layer}, evidence, self.config)
        self.assertEqual(result.verdict, "SOFT_PASS_LOW_CONFIDENCE")

    def test_disabled_guard_does_not_create_remainder(self):
        subject = portrait_subject()
        semantic = subject.copy()
        semantic[12:29, 5:9] = 0
        original = rgba(subject)
        evidence = resolve_subject_mask(original, config=self.config)
        disabled = PortraitConfig(raw={
            **self.config.raw,
            "guard": {**self.config.section("guard"), "enabled": False},
        })
        result = apply_silhouette_guard(original, {"topwear": rgba(semantic)}, evidence, disabled)
        self.assertFalse(np.any(result.body_remainder[..., 3]))
        self.assertLess(result.metrics.post_recovery_coverage, 0.995)


if __name__ == "__main__":
    unittest.main()
