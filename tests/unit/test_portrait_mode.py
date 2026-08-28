import unittest

import numpy as np

from portrait_core import PortraitConfig, evaluate_portrait_layers, resolve_subject_mask
from tests.unit.helpers import portrait_subject, rgba


class PortraitModeTests(unittest.TestCase):
    def setUp(self):
        self.config = PortraitConfig.load()
        self.subject = portrait_subject()
        self.evidence = resolve_subject_mask(rgba(self.subject), config=self.config)

    def test_missing_legs_and_handwear_do_not_fail_profile(self):
        layers = {
            "head": rgba(self.subject * 0 + np.pad(np.ones((8, 8)), ((3, 21), (12, 12)))),
            "front hair": rgba(self.subject * 0 + np.pad(np.ones((6, 10)), ((2, 24), (11, 11)))),
            "topwear": rgba(self.subject),
            "face": rgba(self.subject * 0 + np.pad(np.ones((6, 6)), ((5, 21), (13, 13)))),
        }
        result = evaluate_portrait_layers(layers, self.evidence, enable_head_detail=True, config=self.config)
        self.assertTrue(result.semantic_success)
        self.assertFalse(result.handwear_detected)
        self.assertNotIn("legwear", result.missing_critical_groups)

    def test_face_is_not_critical_when_head_detail_disabled(self):
        layers = {
            "head": rgba(self.subject),
            "front hair": rgba(self.subject),
            "topwear": rgba(self.subject),
        }
        result = evaluate_portrait_layers(layers, self.evidence, enable_head_detail=False, config=self.config)
        self.assertTrue(result.semantic_success)


if __name__ == "__main__":
    unittest.main()
