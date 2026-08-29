import json
import os
import tempfile
import unittest

import numpy as np

from seethrough_engine.rig import (
    trim_layer_edges,
    BODY_REMAINDER,
    BODY_WEIGHT,
    EYE_SPLIT_TAGS,
    GROUP_BODY,
    GROUP_HEAD,
    GROUP_NECK,
    HEAD_REMAINDER,
    HEAD_WEIGHT,
    NECK_REMAINDER,
    RECLAIM_PAIRS,
    RIG_Z_ORDER,
    build_rig,
    composite_fidelity,
    composite_layers,
    depth_table,
    detect_anchors,
    group_for_tag,
    reclaim_occluded,
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


class EdgeTrimTests(unittest.TestCase):
    """A layer's outermost pixels are darkened where its alpha ends, and drawn
    over a lighter layer behind, that rim is a stroke the picture does not
    have."""

    def scene(self, rim=(20, 20, 20)):
        original = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
        original[20:100, 20:100, :3] = 220
        original[20:100, 20:100, 3] = 255
        back = np.zeros_like(original)
        back[20:100, 20:100, :3] = 220          # the layer behind is right
        back[20:100, 20:100, 3] = 255
        front = np.zeros_like(original)
        # Big enough to be a surface rather than a stroke: 60x60 is 81% interior.
        front[25:85, 25:85, :3] = 218           # the front is right inside
        front[25:85, 25:85, 3] = 255
        for d in range(3):                      # ... and dark at its own edge
            front[25 + d, 25:85, :3] = rim
            front[84 - d, 25:85, :3] = rim
            front[25:85, 25 + d, :3] = rim
            front[25:85, 84 - d, :3] = rim
        return original, {"neck": back, "face": front}

    def test_a_dark_rim_over_a_layer_that_is_right_is_handed_back(self):
        original, layers = self.scene()
        out, moved = trim_layer_edges(layers, original)
        self.assertGreater(moved.get("face", 0), 100)
        self.assertLess(int(out["face"][26, 50, 3]), 128)

    def test_the_layer_s_interior_is_untouched(self):
        original, layers = self.scene()
        out, _ = trim_layer_edges(layers, original)
        self.assertEqual(int(out["face"][50, 50, 3]), 255)

    def test_an_edge_that_matches_the_original_is_left_alone(self):
        original, layers = self.scene(rim=(218, 218, 218))
        out, moved = trim_layer_edges(layers, original)
        self.assertEqual(moved, {})
        self.assertEqual(int(out["face"][26, 50, 3]), 255)

    def test_nothing_is_handed_over_where_there_is_nothing_behind(self):
        original, layers = self.scene()
        _, moved = trim_layer_edges({"face": layers["face"]}, original)
        self.assertEqual(moved, {})

    def test_a_layer_that_is_all_edge_is_left_alone(self):
        """A stroke -- a mouth line, a lash, a nose -- is an outline, and an
        outline is meant to be dark. Trimming one thins it until it fades."""
        original, layers = self.scene()
        stroke = np.zeros_like(layers["face"])
        stroke[48:52, 30:70, :3] = (20, 20, 20)     # four pixels tall: no interior
        stroke[48:52, 30:70, 3] = 255
        layers["mouth"] = stroke
        out, moved = trim_layer_edges(layers, original)
        self.assertNotIn("mouth", moved)
        self.assertEqual(int(out["mouth"][49, 50, 3]), 255)

    def test_no_layer_s_colour_is_altered(self):
        original, layers = self.scene()
        out, _ = trim_layer_edges(layers, original)
        for tag in layers:
            self.assertTrue(np.array_equal(out[tag][..., :3], layers[tag][..., :3]))


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

    def test_the_original_is_optional_and_changes_nothing_when_absent(self):
        """Default behaviour must not move: without the original there is no
        ground truth to resolve a contested pixel against."""
        manifest, _ = build_rig(self.layers, frame_size=(CANVAS, CANVAS))
        self.assertEqual(manifest["source"]["reclaimed"], {})

    def test_passing_the_original_records_what_moved(self):
        layers, original = contested_scene()
        layers["face"] = rgba([(52, 20, 76, 52)])
        manifest, _ = build_rig(layers, original_rgba=original,
                                frame_size=(CANVAS, CANVAS))
        self.assertEqual(set(manifest["source"]["reclaimed"]), {"topwear<-neck"})

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

    def test_the_neck_gradient_ends_exactly_on_the_head_and_the_body(self):
        """These endpoints are not free parameters. A neck top below
        HEAD_WEIGHT puts a step at the jaw, and a bottom that is not
        BODY_WEIGHT puts one at the collar -- and with a stand collar hiding
        three quarters of the neck, the jaw step is all you see."""
        manifest, _ = build_rig(self.layers, frame_size=(CANVAS, CANVAS))
        weights = {part["tag"]: part["weight"] for part in manifest["parts"]}
        self.assertEqual(weights["neck"]["top"], HEAD_WEIGHT)
        self.assertEqual(weights["neck"]["bottom"], BODY_WEIGHT)
        self.assertEqual(weights["face"], {"mode": "constant", "value": HEAD_WEIGHT})

    def test_a_collar_shares_the_neck_gradient_exactly(self):
        """`reclaim_occluded` cuts a window in the garment for the neck to show
        through, so the window and its contents are two sides of one seam. Two
        different weight functions there and the window's edge slices the neck
        as the head turns -- a 2.05 px crack on the collar line, measured."""
        layers = dict(self.layers)
        layers["topwear"] = rgba([(36, 66, 92, 124)])  # collar rides up over the neck
        manifest, _ = build_rig(layers, frame_size=(CANVAS, CANVAS))
        weights = {part["tag"]: part["weight"] for part in manifest["parts"]}
        self.assertEqual(weights["topwear"], weights["neck"])
        self.assertEqual(weights["topwear"]["top"], HEAD_WEIGHT)
        self.assertEqual(weights["topwear"]["bottom"], BODY_WEIGHT)

    def test_a_garment_clear_of_the_neck_stays_rigid(self):
        """A low neckline is not a collar; ramping it would wobble the torso."""
        layers = dict(self.layers)
        layers["topwear"] = rgba([(36, 80, 92, 124)])
        manifest, _ = build_rig(layers, frame_size=(CANVAS, CANVAS))
        weights = {part["tag"]: part["weight"] for part in manifest["parts"]}
        self.assertEqual(weights["topwear"], {"mode": "constant", "value": BODY_WEIGHT})

    def test_gradient_parts_get_the_finer_mesh(self):
        """At CANVAS=128 both cells clamp to the 8px floor, so this has to run
        at a realistic size to say anything."""
        big = {tag: np.kron(img, np.ones((6, 6, 1), np.uint8)) for tag, img in self.layers.items()}
        manifest, _ = build_rig(big, frame_size=(CANVAS * 6, CANVAS * 6))
        parts = {part["tag"]: part for part in manifest["parts"]}
        self.assertLess(parts["neck"]["mesh"]["cell"], parts["face"]["mesh"]["cell"])

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


class CompositeTests(unittest.TestCase):
    """The gap these close: every coverage metric is alpha-only, and the
    `reconstruction` diagnostic copies the original's RGB, so a layer that is
    present and correctly shaped but the wrong colour scores as a clean PASS."""

    def test_composite_reproduces_a_complete_stack(self):
        layers = portrait_layers()
        composite = composite_layers(layers, (CANVAS, CANVAS))
        # Rebuild what the stack should look like: the frontmost layer wins.
        expected = np.zeros((CANVAS, CANVAS, 4), np.uint8)
        for tag in sorted(layers, key=lambda t: RIG_Z_ORDER.index(t)):
            covered = layers[tag][..., 3] > 10
            expected[covered] = layers[tag][covered]
        np.testing.assert_array_equal(composite[..., 3] > 10, expected[..., 3] > 10)

    def test_draw_order_decides_which_layer_shows(self):
        back = rgba([(0, 0, 32, 32)], value=10)
        front = rgba([(0, 0, 32, 32)], value=250)
        composite = composite_layers({"face": front, "back hair": back}, (CANVAS, CANVAS))
        self.assertGreater(int(composite[5, 5, 0]), 200)  # face is in front of back hair

    def test_a_perfect_stack_scores_zero(self):
        original = rgba([(0, 0, 64, 64)], value=200)
        metrics = composite_fidelity(original, original, original[..., 3] > 10)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["bad_ratio"], 0.0)

    def test_a_dropped_feature_layer_shows_up(self):
        """A missing `eyewhite` leaves skin where the sclera was: alpha coverage
        is unchanged, so only an RGB comparison can see it."""
        skin = rgba([(0, 0, 32, 32)], value=200)
        white = rgba([(10, 10, 20, 20)], value=255)
        original = composite_layers({"face": skin, "eyewhite": white}, (CANVAS, CANVAS))
        without = composite_layers({"face": skin}, (CANVAS, CANVAS))
        subject = original[..., 3] > 10
        # Alpha is identical -- the coverage metrics would see nothing wrong.
        np.testing.assert_array_equal(original[..., 3] > 10, without[..., 3] > 10)
        metrics = composite_fidelity(original, without, subject)
        self.assertGreater(metrics["bad_ratio"], 0.0)
        self.assertGreater(metrics["mae"], 0.0)

    def test_the_remainder_is_what_separates_the_two_figures(self):
        """The rendered stack includes the recovered remainder; the semantic
        layers alone do not. Scoring only the latter makes a remainder-heavy
        run look catastrophic for a reason nobody ever sees on screen."""
        skin = rgba([(0, 0, 32, 32)], value=200)
        recovered = rgba([(40, 40, 60, 60)], value=120)
        original = composite_layers({"face": skin, "body_remainder": recovered},
                                    (CANVAS, CANVAS))
        subject = original[..., 3] > 10
        rendered = composite_fidelity(
            original, composite_layers({"face": skin, "body_remainder": recovered},
                                       (CANVAS, CANVAS)), subject)
        semantic = composite_fidelity(
            original, composite_layers({"face": skin}, (CANVAS, CANVAS)), subject)
        # The rendered stack is exact; the semantic-only figure is wrong by
        # precisely the share of the subject the remainder carries.
        self.assertEqual(rendered["bad_ratio"], 0.0)
        remainder_share = float((recovered[..., 3] > 10).sum()) / float(subject.sum())
        self.assertAlmostEqual(semantic["bad_ratio"], remainder_share, places=4)

    def test_empty_subject_is_not_a_division_by_zero(self):
        blank = np.zeros((CANVAS, CANVAS, 4), np.uint8)
        self.assertEqual(composite_fidelity(blank, blank, blank[..., 3] > 10)["subject_px"], 0)

    def test_float_and_uint8_masks_are_both_accepted(self):
        original = rgba([(0, 0, 64, 64)], value=200)
        subject = original[..., 3]
        self.assertEqual(composite_fidelity(original, original, subject)["subject_px"],
                         int((subject > 127).sum()))
        self.assertEqual(
            composite_fidelity(original, original, subject.astype(np.float32) / 255.0)["subject_px"],
            int((subject > 127).sum()))


def contested_scene():
    """A garment holding the skin from its own opening: `topwear` covers the
    neck opaquely, and where it does it is painted a near-skin colour that is
    not quite what the original shows. Returns `(layers, original)`."""
    neck = rgba([(50, 50, 80, 95)], value=200)
    topwear = rgba([(40, 60, 90, 124)], value=90)
    topwear[60:88, 52:78, :3] = 205  # absorbed skin, close to the neck but off

    original = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    original[..., 3] = 255
    original[50:95, 50:80, :3] = 200          # neck
    original[60:124, 40:90, :3] = 90          # garment over it
    original[60:88, 52:78, :3] = 200          # ... except the opening: skin
    return {"neck": neck, "topwear": topwear}, original


class ReclaimOccludedTests(unittest.TestCase):
    def test_the_garment_gives_back_the_skin_it_absorbed(self):
        layers, original = contested_scene()
        out, moved = reclaim_occluded(layers, original)
        self.assertEqual(set(moved), {"topwear<-neck"})
        self.assertGreater(moved["topwear<-neck"], 400)
        # the opening is now a hole in the garment, so the neck shows through
        self.assertFalse((out["topwear"][65:83, 57:73, 3] > 10).any())

    def test_the_region_extends_past_the_margin_qualified_core(self):
        """The margin decides which regions change hands, not where each one
        ends. Applied per pixel it cut a decisively-neck area into lace: on
        A-001 the neck explained 93% of a band better while only 59% cleared
        the margin, leaving skin-coloured garment behind in ragged patches."""
        plain, plain_original = contested_scene()
        _, plain_moved = reclaim_occluded(plain, plain_original)

        # The same scene plus a band the neck wins by a hair: under the margin,
        # so it can never seed a region, but touching one that is already
        # seeded. Following the edge should take it; thresholding should not.
        layers, original = contested_scene()
        layers["topwear"][88:96, 52:78, :3] = 196
        original[88:96, 52:78, :3] = 200
        out, moved = reclaim_occluded(layers, original)

        self.assertGreater(moved["topwear<-neck"], plain_moved["topwear<-neck"],
                           "the region stopped at the margin instead of at the edge")
        cleared = out["topwear"][88:93, 57:73, 3] < 128
        self.assertGreater(cleared.mean(), 0.9)

    def test_an_area_with_no_decisive_core_is_left_alone(self):
        """Being better by a hair everywhere is not evidence of a
        mis-assignment; without a seed nothing changes hands."""
        layers, original = contested_scene()
        layers["topwear"][60:88, 52:78, :3] = 196
        original[60:88, 52:78, :3] = 200
        _, moved = reclaim_occluded(layers, original)
        self.assertEqual(moved, {})

    def test_the_handover_is_feathered(self):
        """A hard alpha cut along a jagged boundary glitters as sub-pixel
        motion moves it -- one flicker per breath at the collar."""
        layers, original = contested_scene()
        out, _ = reclaim_occluded(layers, original)
        alpha = out["topwear"][..., 3]
        self.assertTrue(((alpha > 0) & (alpha < 255)).any(),
                        "the handover has no partial alpha at all")
        hard = reclaim_occluded(layers, original, feather=0)[0]["topwear"][..., 3]
        self.assertFalse(((hard > 0) & (hard < 255)).any())

    def test_feathering_never_opens_a_gap(self):
        """The fade is clamped by the layer behind it: softening the garment
        where nothing shows through would be a hole, not a blend."""
        layers, original = contested_scene()
        out, _ = reclaim_occluded(layers, original)
        softened = (out["topwear"][..., 3] < layers["topwear"][..., 3])
        behind = layers["neck"][..., 3] > 0
        self.assertFalse((softened & ~behind).any())

    def test_the_real_collar_is_left_alone(self):
        """Where the garment genuinely covers the neck it explains the original
        and must keep its pixels -- otherwise the collar turns into skin."""
        layers, original = contested_scene()
        out, _ = reclaim_occluded(layers, original)
        self.assertTrue((out["topwear"][90:95, 55:75, 3] > 10).all())

    def test_the_back_layer_is_never_modified(self):
        layers, original = contested_scene()
        out, _ = reclaim_occluded(layers, original)
        np.testing.assert_array_equal(out["neck"], layers["neck"])

    def test_only_alpha_moves_never_colour(self):
        layers, original = contested_scene()
        out, _ = reclaim_occluded(layers, original)
        np.testing.assert_array_equal(out["topwear"][..., :3], layers["topwear"][..., :3])

    def test_a_faithful_garment_is_a_no_op(self):
        layers, original = contested_scene()
        layers["topwear"][60:88, 52:78, :3] = 90
        original[60:88, 52:78, :3] = 90
        out, moved = reclaim_occluded(layers, original)
        self.assertEqual(moved, {})
        np.testing.assert_array_equal(out["topwear"], layers["topwear"])

    def test_speckle_is_not_worth_moving(self):
        """A scattering of single pixels is noise in the decomposition, not a
        region the model mis-assigned."""
        layers, original = contested_scene()
        layers["topwear"][60:88, 52:78, :3] = 90
        original[60:88, 52:78, :3] = 90
        for y in range(62, 86, 6):
            for x in range(54, 76, 6):
                original[y, x, :3] = 200
        _, moved = reclaim_occluded(layers, original)
        self.assertEqual(moved, {})

    def test_only_the_listed_pairs_are_contested(self):
        """Other pairings were measured and rejected: `topwear` over `face` and
        `head` over `face` made the composite worse, and `back hair` over
        `head` moved 21k pixels for no measurable gain."""
        self.assertEqual(RECLAIM_PAIRS, (("topwear", "neck"), ("neckwear", "neck")))
        layers, original = contested_scene()
        layers["face"] = layers.pop("neck")
        out, moved = reclaim_occluded(layers, original)
        self.assertEqual(moved, {})

    def test_a_missing_layer_is_not_an_error(self):
        _, moved = reclaim_occluded({"topwear": rgba([(0, 0, 32, 32)])},
                                    np.zeros((CANVAS, CANVAS, 4), np.uint8))
        self.assertEqual(moved, {})

    def test_it_improves_how_well_the_stack_reproduces_the_original(self):
        """The point of resolving the tie by measurement: it is checkable.
        On A-001 this moved the composite from mae 18.74 to 17.54."""
        layers, original = contested_scene()
        subject = original[..., 3] > 10
        before = composite_fidelity(
            original, composite_layers(layers, (CANVAS, CANVAS)), subject)
        out, _ = reclaim_occluded(layers, original)
        after = composite_fidelity(
            original, composite_layers(out, (CANVAS, CANVAS)), subject)
        self.assertLess(after["mae"], before["mae"])


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
