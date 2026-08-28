import json
import os
import tempfile
import unittest

import numpy as np

from seethrough_engine.spine import (
    BODY_REMAINDER_DEPTH,
    DEFAULT_SPINE_NAMES,
    apply_depth_ordering,
    build_skeleton,
    fill_missing_depths,
    draw_order,
    layers_to_parts,
    rename_parts,
    semantic_rank,
    write_spine_project,
)


def _solid(h, w, box=None, alpha=255):
    """RGBA canvas, opaque inside `box` (x1, y1, x2, y2) or everywhere."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if box is None:
        img[..., :] = (10, 20, 30, alpha)
    else:
        x1, y1, x2, y2 = box
        img[y1:y2, x1:x2] = (10, 20, 30, alpha)
    return img


class CoordinateConversionTests(unittest.TestCase):
    """Pins the geometry `SeeThrough_ExportSpine` had inline before it
    delegated here: Spine's origin is bottom-center of the canvas, Y up."""

    def test_attachment_is_placed_at_layer_center_in_spine_coords(self):
        parts = {"head": {"img": np.zeros((20, 20, 4), np.uint8), "xyxy": [10, 20, 30, 40]}}
        skeleton, _ = build_skeleton(parts, (100, 200))

        att = skeleton["skins"][0]["attachments"]["head"]["head"]
        # center on canvas = (20, 30); canvas is 200 wide, 100 tall
        self.assertEqual(att["x"], 20 - 100.0)
        self.assertEqual(att["y"], 100 - 30.0)
        self.assertEqual((att["width"], att["height"]), (20, 20))

    def test_skeleton_header_describes_the_canvas(self):
        parts = {"head": {"img": np.zeros((4, 4, 4), np.uint8), "xyxy": [0, 0, 4, 4]}}
        skeleton, _ = build_skeleton(parts, (100, 200), spine_version="4.2.28")

        self.assertEqual(skeleton["skeleton"]["x"], -100.0)
        self.assertEqual(skeleton["skeleton"]["y"], 0)
        self.assertEqual(skeleton["skeleton"]["width"], 200)
        self.assertEqual(skeleton["skeleton"]["height"], 100)
        self.assertEqual(skeleton["skeleton"]["spine"], "4.2.28")
        self.assertEqual(skeleton["skeleton"]["images"], "./images/")
        self.assertEqual(skeleton["bones"], [{"name": "root"}])
        self.assertEqual(skeleton["slots"][0], {"name": "head", "bone": "root", "attachment": "head"})

    def test_space_in_tag_becomes_hyphen_in_slot_and_image_name(self):
        parts = {"front hair": {"img": np.zeros((4, 4, 4), np.uint8), "xyxy": [0, 0, 4, 4]}}
        skeleton, images = build_skeleton(parts, (10, 10))

        self.assertEqual(skeleton["slots"][0]["name"], "front-hair")
        self.assertIn("front-hair", images)


class DrawOrderTests(unittest.TestCase):
    def test_depth_sorts_back_to_front_descending(self):
        parts = {
            "near": {"img": None, "depth_median": 0.1},
            "far": {"img": None, "depth_median": 0.9},
            "mid": {"img": None, "depth_median": 0.5},
        }
        self.assertEqual(draw_order(parts), ["far", "mid", "near"])

    def test_layer_without_depth_defaults_to_one_as_the_node_did(self):
        # SeeThrough_ExportSpine used tag2pinfo[t].get("depth_median", 1), so a
        # layer missing the key sorted as if it were at depth 1 (far back).
        parts = {"has": {"img": None, "depth_median": 0.5}, "missing": {"img": None}}
        self.assertEqual(draw_order(parts), ["missing", "has"])

    def test_without_any_depth_falls_back_to_semantic_order(self):
        parts = {t: {"img": None} for t in ["mouth", "back hair", "face", "front hair"]}
        self.assertEqual(draw_order(parts), ["back hair", "face", "mouth", "front hair"])

    def test_semantic_order_agrees_with_the_depth_adjustments_nodes_py_applies(self):
        # nodes.py forces nose/mouth/eyes in front of `face` and ears behind it
        # once it has depth. The depth-free order must not contradict that.
        for front_of_face in ("nose", "mouth", "eyes"):
            self.assertGreater(semantic_rank(front_of_face), semantic_rank("face"), front_of_face)
        self.assertLess(semantic_rank("ears"), semantic_rank("face"))

    def test_body_remainder_sits_behind_every_semantic_layer(self):
        others = [t for t in DEFAULT_SPINE_NAMES if semantic_rank(t) >= 0]
        self.assertTrue(others)
        for tag in others:
            self.assertLess(semantic_rank("body_remainder"), semantic_rank(tag), tag)

    def test_unknown_tag_sorts_behind_everything_rather_than_over_a_face(self):
        self.assertLess(semantic_rank("no-such-tag"), semantic_rank("body_remainder"))

    def test_semantic_order_survives_renaming(self):
        parts = rename_parts({t: {"img": None} for t in ["mouth", "back hair", "face"]})
        self.assertEqual(draw_order(parts), ["back-hair", "face", "mouth"])


class DepthOrderingTests(unittest.TestCase):
    """The overrides nodes.py applies on top of raw Marigold output."""

    def test_facial_features_are_pulled_in_front_of_the_face(self):
        parts = apply_depth_ordering({
            "face": {"depth_median": 0.5},
            "nose": {"depth_median": 0.9},   # behind the face -- must be fixed
            "mouth": {"depth_median": 0.2},  # already in front -- left alone
        })
        self.assertLess(parts["nose"]["depth_median"], parts["face"]["depth_median"])
        self.assertEqual(parts["mouth"]["depth_median"], 0.2)

    def test_ears_are_pushed_behind_the_face_even_when_estimated_in_front(self):
        parts = apply_depth_ordering({"face": {"depth_median": 0.5}, "ears": {"depth_median": 0.1}})
        self.assertGreater(parts["ears"]["depth_median"], parts["face"]["depth_median"])

    def test_body_remainder_is_pinned_past_any_normalized_depth(self):
        parts = apply_depth_ordering({"body_remainder": {"depth_median": 0.3}})
        self.assertEqual(parts["body_remainder"]["depth_median"], BODY_REMAINDER_DEPTH)
        self.assertGreater(BODY_REMAINDER_DEPTH, 1.0)

    def test_no_face_layer_leaves_depths_untouched(self):
        parts = apply_depth_ordering({"nose": {"depth_median": 0.9}})
        self.assertEqual(parts["nose"]["depth_median"], 0.9)

    def test_depth_median_is_taken_over_visible_pixels_only(self):
        img = _solid(8, 8, box=(0, 0, 4, 8))          # left half opaque
        depth = np.zeros((8, 8), np.float32)
        depth[:, :4] = 0.25                            # under the visible half
        depth[:, 4:] = 0.99                            # under the transparent half
        parts = layers_to_parts({"topwear": img}, depth_dict={"topwear": depth})
        self.assertAlmostEqual(parts["topwear"]["depth_median"], 0.25, places=6)

    def test_depth_switches_draw_order_away_from_semantic(self):
        # semantic order would be back hair -> face; depth here says otherwise
        imgs = {"back hair": _solid(8, 8), "face": _solid(8, 8)}
        depths = {"back hair": np.full((8, 8), 0.1, np.float32),
                  "face": np.full((8, 8), 0.9, np.float32)}
        parts = layers_to_parts(imgs, depth_dict=depths)
        self.assertEqual(draw_order(parts), ["face", "back hair"])

    def test_layer_missing_from_depth_dict_is_interpolated_not_left_bare(self):
        # `head` is semantically behind `face`, and the only estimate present
        # is face's, so it is placed just behind it rather than left to sort at
        # draw_order's default of 1. See FillMissingDepthTests.
        parts = layers_to_parts(
            {"face": _solid(8, 8), "head": _solid(8, 8)},
            depth_dict={"face": np.full((8, 8), 0.5, np.float32)},
        )
        self.assertIn("depth_median", parts["face"])
        self.assertGreater(parts["head"]["depth_median"], parts["face"]["depth_median"])


class FillMissingDepthTests(unittest.TestCase):
    """`head` has no v2 batch slot, so the depth pass never covers it."""

    def test_uncovered_layer_lands_between_its_semantic_neighbours(self):
        # semantic: ... handwear < head < ears ...
        parts = fill_missing_depths({
            "handwear": {"depth_median": 0.8},
            "head": {},
            "ears": {"depth_median": 0.4},
        })
        self.assertAlmostEqual(parts["head"]["depth_median"], 0.6)

    def test_uncovered_layer_in_front_of_everything_known(self):
        parts = fill_missing_depths({"back hair": {"depth_median": 0.8}, "headwear": {}})
        self.assertLess(parts["headwear"]["depth_median"], 0.8)

    def test_uncovered_layer_behind_everything_known(self):
        parts = fill_missing_depths({"wings": {}, "face": {"depth_median": 0.3}})
        self.assertGreater(parts["wings"]["depth_median"], 0.3)

    def test_body_remainder_is_left_for_apply_depth_ordering(self):
        parts = fill_missing_depths({"body_remainder": {}, "face": {"depth_median": 0.3}})
        self.assertNotIn("depth_median", parts["body_remainder"])

    def test_nothing_to_interpolate_from_is_left_alone(self):
        parts = fill_missing_depths({"head": {}})
        self.assertNotIn("depth_median", parts["head"])

    def test_head_is_not_dumped_at_the_back_of_a_real_run(self):
        # regression: before this, `head` fell to draw_order's default of 1 and
        # sorted behind back hair, purely as an artifact of the default.
        imgs = {t: _solid(8, 8) for t in ["back hair", "head", "face", "front hair"]}
        depths = {"back hair": np.full((8, 8), 0.80, np.float32),
                  "face": np.full((8, 8), 0.40, np.float32),
                  "front hair": np.full((8, 8), 0.20, np.float32)}
        order = draw_order(layers_to_parts(imgs, depth_dict=depths))
        self.assertEqual(order.index("head"), 1, order)
        self.assertLess(order.index("back hair"), order.index("head"))


class LayersToPartsTests(unittest.TestCase):
    def test_layer_is_cropped_to_its_alpha_bounds(self):
        parts = layers_to_parts({"head": _solid(64, 64, box=(10, 20, 30, 50))})

        self.assertEqual(parts["head"]["xyxy"], [10, 20, 30, 50])
        self.assertEqual(parts["head"]["img"].shape, (30, 20, 4))

    def test_fully_transparent_layer_is_dropped(self):
        self.assertEqual(layers_to_parts({"tail": np.zeros((8, 8, 4), np.uint8)}), {})

    def test_non_rgba_layer_is_ignored(self):
        self.assertEqual(layers_to_parts({"rgb": np.zeros((8, 8, 3), np.uint8)}), {})

    def test_body_remainder_is_inserted_first(self):
        parts = layers_to_parts(
            {"head": _solid(16, 16, box=(0, 0, 4, 4))},
            body_remainder=_solid(16, 16, box=(8, 8, 12, 12)),
        )
        self.assertEqual(list(parts)[0], "body_remainder")
        self.assertTrue(parts["body_remainder"]["is_recovery"])

    def test_empty_body_remainder_is_not_added(self):
        parts = layers_to_parts(
            {"head": _solid(16, 16, box=(0, 0, 4, 4))},
            body_remainder=np.zeros((16, 16, 4), np.uint8),
        )
        self.assertNotIn("body_remainder", parts)


class RenameTests(unittest.TestCase):
    def test_default_mapping_is_applied_and_original_tag_kept(self):
        parts = rename_parts({"front hair": {"img": None}})
        self.assertIn("front-hair", parts)
        self.assertEqual(parts["front-hair"]["original_tag"], "front hair")
        self.assertEqual(parts["front-hair"]["tag"], "front-hair")

    def test_custom_mapping_overrides_default(self):
        parts = rename_parts({"topwear": {"img": None}}, {"topwear": "shirt"})
        self.assertIn("shirt", parts)

    def test_unmapped_tag_passes_through(self):
        self.assertIn("body_remainder", rename_parts({"body_remainder": {"img": None}}))


class WriteProjectTests(unittest.TestCase):
    def test_writes_json_next_to_the_images_it_references(self):
        parts = layers_to_parts({
            "back hair": _solid(32, 32, box=(0, 0, 16, 16)),
            "face": _solid(32, 32, box=(8, 8, 24, 24)),
        })
        with tempfile.TemporaryDirectory() as tmp:
            json_path = write_spine_project(tmp, "a001", rename_parts(parts), (32, 32))

            self.assertTrue(os.path.isfile(json_path))
            with open(json_path, encoding="utf-8") as f:
                skeleton = json.load(f)

            names = [slot["name"] for slot in skeleton["slots"]]
            self.assertEqual(names, ["back-hair", "face"])
            for name in names:
                self.assertTrue(
                    os.path.isfile(os.path.join(tmp, "images", f"{name}.png")),
                    f"{name}.png referenced by the skeleton but not written",
                )


if __name__ == "__main__":
    unittest.main()
