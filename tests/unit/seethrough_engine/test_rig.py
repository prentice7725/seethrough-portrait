import json
import os
import tempfile
import unittest

import numpy as np

from seethrough_engine.rig import (
    BODY_REMAINDER,
    BODY_WEIGHT,
    EYE_SPLIT_TAGS,
    GROUP_BODY,
    GROUP_HEAD,
    GROUP_NECK,
    HEAD_REMAINDER,
    HEAD_WEIGHT,
    NECK_REMAINDER,
    RIG_Z_ORDER,
    build_rig,
    depth_table,
    detect_anchors,
    group_for_tag,
    split_eyes,
    split_remainder,
    write_rig_project,
)

CANVAS = 128


def rgba(boxes, value=255):
    """Canvas-sized RGBA with `boxes` -- (x1, y1, x2, y2) -- filled opaque."""
    img = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        img[y1:y2, x1:x2, :3] = value
        img[y1:y2, x1:x2, 3] = 255
    return img


def portrait_layers():
    """A crude upper-body portrait: head block on top, neck between, torso
    below, with both eyes inside one `eyewhite` layer the way v3 emits them."""
    return {
        "back hair": rgba([(40, 8, 88, 60)]),
        "head": rgba([(48, 12, 80, 56)]),
        "face": rgba([(52, 20, 76, 52)]),
        "eyewhite": rgba([(56, 30, 62, 36), (66, 30, 72, 36)]),
        "mouth": rgba([(61, 44, 67, 47)]),
        "neck": rgba([(58, 56, 70, 72)]),
        "topwear": rgba([(36, 72, 92, 124)]),
    }


class GroupTests(unittest.TestCase):
    def test_known_tags_land_in_their_group(self):
        self.assertEqual(group_for_tag("face"), GROUP_HEAD)
        self.assertEqual(group_for_tag("back hair"), GROUP_HEAD)
        self.assertEqual(group_for_tag("neck"), GROUP_NECK)
        self.assertEqual(group_for_tag("topwear"), GROUP_BODY)

    def test_remainder_regions_follow_their_own_group(self):
        self.assertEqual(group_for_tag(HEAD_REMAINDER), GROUP_HEAD)
        self.assertEqual(group_for_tag(NECK_REMAINDER), GROUP_NECK)
        self.assertEqual(group_for_tag(BODY_REMAINDER), GROUP_BODY)

    def test_unknown_tag_falls_back_to_body(self):
        """A mystery layer that fails to follow the head is a missed
        opportunity; one that follows it can tear off the torso."""
        self.assertEqual(group_for_tag("no-such-tag"), GROUP_BODY)


class DepthTableTests(unittest.TestCase):
    def test_runs_from_far_to_near_over_the_z_order(self):
        table = depth_table()
        self.assertEqual(table[RIG_Z_ORDER[0]], 1.0)
        self.assertEqual(table[RIG_Z_ORDER[-1]], 0.0)

    def test_back_hair_is_further_than_front_hair(self):
        table = depth_table()
        self.assertGreater(table["back hair"], table["face"])
        self.assertGreater(table["face"], table["front hair"])

    def test_remainder_regions_sit_behind_what_they_move_with(self):
        table = depth_table()
        self.assertGreater(table[HEAD_REMAINDER], table["head"])
        self.assertGreater(table[NECK_REMAINDER], table["neck"])
        self.assertEqual(table[BODY_REMAINDER], 1.0)


class SplitRemainderTests(unittest.TestCase):
    def test_pixels_are_assigned_to_the_nearest_group(self):
        layers = portrait_layers()
        # One patch beside the head, one beside the torso.
        remainder = rgba([(30, 20, 40, 30), (20, 90, 30, 100)])
        regions = split_remainder(remainder, layers)

        self.assertIn(HEAD_REMAINDER, regions)
        self.assertIn(BODY_REMAINDER, regions)
        self.assertTrue(regions[HEAD_REMAINDER][20:30, 30:40, 3].all())
        self.assertFalse(regions[BODY_REMAINDER][20:30, 30:40, 3].any())
        self.assertTrue(regions[BODY_REMAINDER][90:100, 20:30, 3].all())

    def test_neck_band_is_carved_out_first(self):
        layers = portrait_layers()
        remainder = rgba([(59, 58, 69, 70)])  # inside the neck bbox
        regions = split_remainder(remainder, layers)
        self.assertIn(NECK_REMAINDER, regions)
        self.assertNotIn(HEAD_REMAINDER, regions)
        self.assertNotIn(BODY_REMAINDER, regions)

    def test_every_recovered_pixel_survives_exactly_one_region(self):
        """The split must not lose or duplicate recovered pixels -- losing them
        would undo the Silhouette Guard's whole point."""
        layers = portrait_layers()
        remainder = rgba([(30, 20, 40, 30), (20, 90, 30, 100), (59, 58, 69, 70)])
        regions = split_remainder(remainder, layers)
        total = np.zeros((CANVAS, CANVAS), dtype=np.int32)
        for img in regions.values():
            total += (img[..., 3] > 10).astype(np.int32)
        np.testing.assert_array_equal(total, (remainder[..., 3] > 10).astype(np.int32))

    def test_empty_remainder_produces_no_regions(self):
        self.assertEqual(split_remainder(np.zeros((CANVAS, CANVAS, 4), np.uint8),
                                         portrait_layers()), {})

    def test_rejects_non_rgba(self):
        with self.assertRaises(ValueError):
            split_remainder(np.zeros((CANVAS, CANVAS, 3), np.uint8), portrait_layers())


class SplitEyesTests(unittest.TestCase):
    def test_both_eyes_in_one_layer_are_separated(self):
        layers = portrait_layers()
        halves = split_eyes(layers, face_center_x=64.0)
        self.assertEqual(set(halves), {"eyewhitel", "eyewhiter"})
        self.assertTrue(halves["eyewhitel"][30:36, 56:62, 3].all())
        self.assertFalse(halves["eyewhitel"][30:36, 66:72, 3].any())
        self.assertTrue(halves["eyewhiter"][30:36, 66:72, 3].all())

    def test_single_component_layer_is_left_whole(self):
        """One eye visible (a three-quarter view, or an occluded eye) is not a
        failed split -- the caller keeps the layer intact."""
        layers = {"eyewhite": rgba([(56, 30, 62, 36)])}
        self.assertEqual(split_eyes(layers, face_center_x=64.0), {})

    def test_components_on_one_side_only_are_left_whole(self):
        layers = {"eyewhite": rgba([(20, 30, 26, 36), (30, 30, 36, 36)])}
        self.assertEqual(split_eyes(layers, face_center_x=64.0), {})

    def test_dilation_never_invents_alpha_outside_the_layer(self):
        layers = portrait_layers()
        halves = split_eyes(layers, face_center_x=64.0, dilate_px=4)
        source = layers["eyewhite"][..., 3] > 10
        for img in halves.values():
            self.assertFalse((img[..., 3] > 10)[~source].any())

    def test_every_split_tag_is_a_known_v3_eye_layer(self):
        self.assertIn("eyewhite", EYE_SPLIT_TAGS)
        self.assertIn("irides", EYE_SPLIT_TAGS)


class AnchorTests(unittest.TestCase):
    def test_neck_pivot_sits_near_the_bottom_of_the_neck(self):
        """Hinging at the bottom is what makes a tilt read as a neck bending
        rather than a head sliding sideways."""
        layers = portrait_layers()
        anchors = detect_anchors(layers, (CANVAS, CANVAS))
        x, y = anchors["neck_pivot"]
        self.assertAlmostEqual(x, 64.0, places=1)
        self.assertAlmostEqual(y, 56 + (72 - 56) * 0.85, places=1)

    def test_body_pivot_is_the_bottom_of_the_torso(self):
        anchors = detect_anchors(portrait_layers(), (CANVAS, CANVAS))
        self.assertAlmostEqual(anchors["body_pivot"][1], 124.0, places=1)

    def test_missing_anchors_are_omitted_not_guessed(self):
        """A fabricated eye position is worse than an absent one: the runtime
        can skip a motion it has no anchor for."""
        anchors = detect_anchors({"face": rgba([(52, 20, 76, 52)])}, (CANVAS, CANVAS))
        self.assertNotIn("eye_left", anchors)
        self.assertNotIn("mouth", anchors)
        self.assertIn("face_center", anchors)

    def test_eye_anchors_appear_once_the_eyes_are_split(self):
        layers = portrait_layers()
        layers.update(split_eyes(layers, face_center_x=64.0))
        anchors = detect_anchors(layers, (CANVAS, CANVAS))
        self.assertLess(anchors["eye_left"][0], anchors["eye_right"][0])


class BuildRigTests(unittest.TestCase):
    def setUp(self):
        self.layers = portrait_layers()
        self.remainder = rgba([(30, 20, 40, 30), (20, 90, 30, 100)])

    def test_manifest_shape(self):
        manifest, images = build_rig(self.layers, body_remainder=self.remainder,
                                     frame_size=(CANVAS, CANVAS), run_id="r1",
                                     tag_version="v3")
        self.assertEqual(manifest["version"], "0.1")
        self.assertEqual(manifest["canvas"], {"width": CANVAS, "height": CANVAS})
        self.assertEqual(manifest["source"]["depth"], "table")
        self.assertTrue(manifest["parts"])
        for part in manifest["parts"]:
            self.assertIn(part["name"], images)
            self.assertEqual(part["image"], f"rig/images/{part['name']}.png")

    def test_undivided_eye_layer_is_replaced_by_its_halves(self):
        manifest, images = build_rig(self.layers, frame_size=(CANVAS, CANVAS))
        names = {part["tag"] for part in manifest["parts"]}
        self.assertIn("eyewhitel", names)
        self.assertIn("eyewhiter", names)
        self.assertNotIn("eyewhite", names)  # would double-draw the eyes

    def test_remainder_regions_become_parts_in_their_own_groups(self):
        manifest, _ = build_rig(self.layers, body_remainder=self.remainder,
                                frame_size=(CANVAS, CANVAS))
        groups = {part["tag"]: part["group"] for part in manifest["parts"]}
        self.assertEqual(groups[HEAD_REMAINDER], GROUP_HEAD)
        self.assertEqual(groups[BODY_REMAINDER], GROUP_BODY)

    def test_head_remainder_is_drawn_behind_the_head_but_follows_it(self):
        """This is the ghost-silhouette fix: behind in z, head weight in motion."""
        manifest, _ = build_rig(self.layers, body_remainder=self.remainder,
                                frame_size=(CANVAS, CANVAS))
        parts = {part["tag"]: part for part in manifest["parts"]}
        self.assertLess(parts[HEAD_REMAINDER]["z"], parts["head"]["z"])
        self.assertEqual(parts[HEAD_REMAINDER]["weight"],
                         {"mode": "constant", "value": HEAD_WEIGHT})

    def test_parts_are_ordered_back_to_front(self):
        manifest, _ = build_rig(self.layers, frame_size=(CANVAS, CANVAS))
        zs = [part["z"] for part in manifest["parts"]]
        depths = [part["depth"] for part in manifest["parts"]]
        self.assertEqual(zs, sorted(zs))
        self.assertEqual(depths, sorted(depths, reverse=True))

    def test_neck_gets_a_gradient_spanning_the_whole_neck_group(self):
        manifest, _ = build_rig(self.layers, body_remainder=rgba([(59, 58, 69, 70)]),
                                frame_size=(CANVAS, CANVAS))
        weights = {part["tag"]: part["weight"] for part in manifest["parts"]}
        self.assertEqual(weights["neck"]["mode"], "gradient_y")
        self.assertGreater(weights["neck"]["top"], weights["neck"]["bottom"])
        # neck and neck_remainder must share one gradient or they deform
        # differently along the seam between them.
        self.assertEqual(weights["neck"], weights[NECK_REMAINDER])

    def test_body_follows_the_head_a_little(self):
        manifest, _ = build_rig(self.layers, frame_size=(CANVAS, CANVAS))
        weights = {part["tag"]: part["weight"] for part in manifest["parts"]}
        self.assertEqual(weights["topwear"], {"mode": "constant", "value": BODY_WEIGHT})

    def test_gradient_tags_opt_a_head_layer_into_a_falloff(self):
        """The documented `back hair` risk: hair reaching past the shoulder
        line tears if it follows the head at full weight."""
        manifest, _ = build_rig(self.layers, frame_size=(CANVAS, CANVAS),
                                gradient_tags=("back hair",))
        weights = {part["tag"]: part["weight"] for part in manifest["parts"]}
        self.assertEqual(weights["back hair"]["mode"], "gradient_y")
        self.assertEqual(weights["back hair"]["top"], HEAD_WEIGHT)
        self.assertEqual(weights["back hair"]["bottom"], BODY_WEIGHT)
        self.assertEqual(weights["face"]["mode"], "constant")

    def test_marigold_depth_overrides_the_table(self):
        depth = {"face": np.full((CANVAS, CANVAS), 0.9, dtype=np.float32)}
        manifest, _ = build_rig(self.layers, frame_size=(CANVAS, CANVAS),
                                depth_dict=depth)
        parts = {part["tag"]: part for part in manifest["parts"]}
        self.assertEqual(manifest["source"]["depth"], "marigold")
        self.assertAlmostEqual(parts["face"]["depth"], 0.9, places=4)

    def test_split_eyes_inherit_their_parent_depth(self):
        depth = {"eyewhite": np.full((CANVAS, CANVAS), 0.4, dtype=np.float32)}
        manifest, _ = build_rig(self.layers, frame_size=(CANVAS, CANVAS),
                                depth_dict=depth)
        parts = {part["tag"]: part for part in manifest["parts"]}
        self.assertAlmostEqual(parts["eyewhitel"]["depth"], 0.4, places=4)
        self.assertAlmostEqual(parts["eyewhiter"]["depth"], 0.4, places=4)

    def test_empty_layers_are_dropped(self):
        layers = dict(self.layers)
        layers["headwear"] = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
        manifest, images = build_rig(layers, frame_size=(CANVAS, CANVAS))
        self.assertNotIn("headwear", {part["tag"] for part in manifest["parts"]})
        self.assertNotIn("headwear", images)

    def test_frame_size_is_inferred_from_the_layers(self):
        manifest, _ = build_rig(self.layers)
        self.assertEqual(manifest["canvas"], {"width": CANVAS, "height": CANVAS})

    def test_frame_size_is_required_when_nothing_has_content(self):
        with self.assertRaises(ValueError):
            build_rig({"face": np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)})


class WriteRigProjectTests(unittest.TestCase):
    def test_writes_manifest_and_images_where_it_says_they_are(self):
        manifest, images = build_rig(portrait_layers(), frame_size=(CANVAS, CANVAS))
        with tempfile.TemporaryDirectory() as out_dir:
            path = write_rig_project(out_dir, "a001", manifest, images)
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as f:
                written = json.load(f)
            for part in written["parts"]:
                self.assertTrue(os.path.isfile(os.path.join(out_dir, part["image"])),
                                part["image"])


if __name__ == "__main__":
    unittest.main()
