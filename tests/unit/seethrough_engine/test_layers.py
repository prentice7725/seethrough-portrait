import unittest

import numpy as np

from seethrough_engine.layers import (
    ALL_TAGS,
    align_subject_mask_to_canvas,
    crop_head,
    layer_similarity,
    make_preview,
)


class LayerSimilarityTests(unittest.TestCase):
    def test_identical_rgb_scores_perfect_match(self):
        img = np.zeros((5, 5, 4), dtype=np.uint8)
        img[1:4, 1:4] = (100, 150, 200, 255)
        self.assertEqual(layer_similarity(img, img), 1.0)

    def test_fully_transparent_layer_scores_zero(self):
        layer = np.zeros((5, 5, 4), dtype=np.uint8)
        original = np.full((5, 5, 4), 255, dtype=np.uint8)
        self.assertEqual(layer_similarity(layer, original), 0.0)

    def test_color_mismatch_reduces_score_by_normalized_mae(self):
        layer = np.zeros((2, 2, 4), dtype=np.uint8)
        layer[..., :3] = 100
        layer[..., 3] = 255
        original = np.zeros((2, 2, 4), dtype=np.uint8)
        original[..., :3] = 150  # constant +50 error on every channel
        original[..., 3] = 255
        expected = 1.0 - (50.0 / 255.0)
        self.assertAlmostEqual(layer_similarity(layer, original), expected, places=6)


class CropHeadTests(unittest.TestCase):
    def test_small_box_gets_padded_within_bounds(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped, (x1, y1, x2, y2) = crop_head(img, [40, 40, 10, 10])
        self.assertEqual((x1, y1, x2, y2), (38, 38, 52, 52))
        self.assertEqual(cropped.shape, (14, 14, 3))

    def test_box_covering_more_than_half_gets_no_padding(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped, (x1, y1, x2, y2) = crop_head(img, [0, 0, 60, 60])
        self.assertEqual((x1, y1, x2, y2), (0, 0, 60, 60))
        self.assertEqual(cropped.shape, (60, 60, 3))

    def test_padding_is_clamped_to_image_bounds(self):
        # Box sits flush against the top-left corner, so the padding term
        # (min(margin_on_far_side, margin_on_near_side, w//5)) has 0 room on
        # the near side and adds no padding there -- it must not go negative
        # or wrap around instead of clamping to the edge.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped, (x1, y1, x2, y2) = crop_head(img, [0, 0, 10, 10])
        self.assertEqual((x1, y1, x2, y2), (0, 0, 10, 10))
        self.assertEqual(cropped.shape, (10, 10, 3))

    def test_padding_extends_when_room_exists_on_both_sides(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped, (x1, y1, x2, y2) = crop_head(img, [30, 30, 10, 10])
        self.assertLess(x1, 30)
        self.assertGreater(x2, 40)
        self.assertLess(y1, 30)
        self.assertGreater(y2, 40)


class TagListTests(unittest.TestCase):
    def test_all_tags_has_no_duplicates(self):
        self.assertEqual(len(ALL_TAGS), len(set(ALL_TAGS)))


class AlignSubjectMaskToCanvasTests(unittest.TestCase):
    def test_output_is_square_canvas_float01(self):
        mask = np.zeros((20, 40), dtype=np.float32)
        mask[5:15, 10:30] = 1.0
        aligned = align_subject_mask_to_canvas(mask, 64)
        self.assertEqual(aligned.shape, (64, 64))
        self.assertGreaterEqual(float(aligned.min()), 0.0)
        self.assertLessEqual(float(aligned.max()), 1.0)
        self.assertTrue(np.any(aligned > 0.5))

    def test_all_zero_mask_stays_all_zero(self):
        mask = np.zeros((10, 10), dtype=np.float32)
        aligned = align_subject_mask_to_canvas(mask, 32)
        self.assertEqual(aligned.shape, (32, 32))
        self.assertEqual(float(aligned.max()), 0.0)


class MakePreviewTests(unittest.TestCase):
    def test_blends_visible_layers(self):
        layer_dict = {"topwear": np.zeros((32, 32, 4), dtype=np.uint8)}
        layer_dict["topwear"][10:20, 10:20] = (255, 0, 0, 255)
        preview = make_preview(layer_dict, 32)
        self.assertEqual(preview.shape, (32, 32, 3))
        self.assertEqual(preview.dtype, np.float32)
        self.assertGreaterEqual(float(preview.max()), 0.9)

    def test_no_visible_layers_returns_black_canvas(self):
        layer_dict = {"topwear": np.zeros((16, 16, 4), dtype=np.uint8)}
        preview = make_preview(layer_dict, 16)
        self.assertEqual(preview.shape, (16, 16, 3))
        self.assertEqual(float(preview.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
