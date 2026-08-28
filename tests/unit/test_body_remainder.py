import unittest

import numpy as np

from portrait_core import build_body_remainder, composite_alpha
from tests.unit.helpers import rgba


class BodyRemainderTests(unittest.TestCase):
    def test_complete_coverage_produces_empty_remainder(self):
        subject = np.ones((8, 8), dtype=np.float32)
        remainder = build_body_remainder(rgba(subject), subject, subject)
        self.assertFalse(np.any(remainder[..., 3]))

    def test_empty_union_recovers_subject(self):
        subject = np.zeros((8, 8), dtype=np.float32)
        subject[1:7, 2:6] = 1.0
        original = rgba(subject)
        remainder = build_body_remainder(original, subject, np.zeros_like(subject))
        np.testing.assert_array_equal(remainder[..., 3], original[..., 3])
        np.testing.assert_array_equal(remainder[..., :3], original[..., :3])

    def test_residual_formula_reconstructs_translucent_edges(self):
        subject = np.array([[0.2, 0.5, 0.8, 1.0]], dtype=np.float32)
        union = np.array([[0.1, 0.3, 0.7, 0.9]], dtype=np.float32)
        remainder = build_body_remainder(rgba(subject), subject, union)
        recovered = composite_alpha(union, remainder[..., 3] / 255.0)
        np.testing.assert_allclose(recovered, subject, atol=1.0 / 255.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
