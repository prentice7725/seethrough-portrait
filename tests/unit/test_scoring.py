import unittest

import numpy as np

from portrait_core import PortraitConfig, resolve_subject_mask, select_best_layer_set
from tests.unit.helpers import portrait_subject, rgba


class ScoringTests(unittest.TestCase):
    def test_complete_silhouette_can_replace_smaller_high_similarity_layer(self):
        config = PortraitConfig.load()
        subject = portrait_subject()
        original = rgba(subject, rgb=(100, 100, 100))
        evidence = resolve_subject_mask(original, config=config)

        partial = subject.copy()
        partial[12:29, 5:9] = 0
        partial[12:29, 23:27] = 0
        head = np.zeros_like(subject)
        head[2:12, 10:22] = 1
        hair = np.zeros_like(subject)
        hair[1:8, 9:23] = 1
        face = np.zeros_like(subject)
        face[4:11, 12:20] = 1
        run1 = {
            "head": rgba(head, rgb=(100, 100, 100)),
            "front hair": rgba(hair, rgb=(100, 100, 100)),
            "face": rgba(face, rgb=(100, 100, 100)),
            "topwear": rgba(partial, rgb=(100, 100, 100)),
        }
        run2 = {"topwear": rgba(subject, rgb=(105, 105, 105))}
        result = select_best_layer_set([run1, run2], original, evidence, config=config)
        self.assertTrue(any(item["accepted"] for item in result.trace))
        np.testing.assert_array_equal(result.layers["topwear"][..., 3], run2["topwear"][..., 3])


if __name__ == "__main__":
    unittest.main()
