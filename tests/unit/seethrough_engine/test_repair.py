import unittest
from unittest.mock import patch

import numpy as np

import seethrough_engine.repair as repair_module
from seethrough_engine.image import composite_fidelity, composite_layers
from seethrough_engine.repair import (
    RECLAIM_PAIRS,
    REPAIR_ORDER,
    REPAIR_VERSION,
    clean_garment_orphans,
    fit_edge_alpha,
    fit_layer_tone,
    fit_neckline_contact,
    fit_seam_residual,
    reclaim_occluded,
    repair_portrait_layers,
)
from seethrough_engine.semantic import SEMANTIC_Z_ORDER

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


class ToneFitTests(unittest.TestCase):
    """Every generated layer is a little off from the picture, and each by a
    different amount. Alone that is invisible; where two meet it is a line."""

    def scene(self, garment_bias=8, neck_bias=-2, base=200):
        """A neck above a garment, meeting at y=60: the two layers each carry
        their own bias and their boundary is where it shows."""
        original = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
        original[20:110, 20:110, :3] = base
        original[20:110, 20:110, 3] = 255
        garment = np.zeros_like(original)
        garment[60:110, 20:110, :3] = base + garment_bias
        garment[60:110, 20:110, 3] = 255
        neck = np.zeros_like(original)
        neck[20:60, 20:110, :3] = base + neck_bias
        neck[20:60, 20:110, 3] = 255
        return original, {"topwear": garment, "neck": neck}

    def test_each_layer_s_bias_is_measured_and_removed(self):
        original, layers = self.scene()
        out, shifts = fit_layer_tone(layers, original)
        self.assertTrue(all(row == [-8, -8, -8] for row in shifts["topwear"]))
        self.assertTrue(all(row == [2, 2, 2] for row in shifts["neck"]))
        self.assertEqual(int(out["topwear"][80, 60, 0]), 200)
        self.assertEqual(int(out["neck"][30, 60, 0]), 200)

    def test_the_step_where_they_meet_goes_with_it(self):
        original, layers = self.scene()
        before = abs(int(layers["neck"][59, 60, 0]) - int(layers["topwear"][61, 60, 0]))
        out, _ = fit_layer_tone(layers, original)
        after = abs(int(out["neck"][59, 60, 0]) - int(out["topwear"][61, 60, 0]))
        self.assertEqual(before, 10)
        self.assertLessEqual(after, 1)

    def test_a_layer_that_already_matches_is_left_alone(self):
        original, layers = self.scene(garment_bias=0, neck_bias=0)
        _, shifts = fit_layer_tone(layers, original)
        self.assertEqual(shifts, {})

    def test_a_layer_with_almost_nothing_showing_is_not_fitted(self):
        original, layers = self.scene()
        speck = np.zeros_like(layers["neck"])
        speck[30:32, 30:32, :3] = 10
        speck[30:32, 30:32, 3] = 255
        layers["mouth"] = speck
        _, shifts = fit_layer_tone(layers, original)
        self.assertNotIn("mouth", shifts)

    def test_the_shift_is_capped(self):
        original, layers = self.scene(garment_bias=40, base=120)
        _, shifts = fit_layer_tone(layers, original)
        self.assertTrue(all(row == [-16, -16, -16] for row in shifts["topwear"]))

    def test_one_layer_covering_two_materials_gets_a_constant_for_each(self):
        """`topwear` is a white shirt beside the neck and a cardigan everywhere
        else. One constant fits neither, and the misfit is a line at the seam."""
        original = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
        original[20:110, 20:110, :3] = 120          # cardigan
        original[20:50, 20:110, :3] = 230           # shirt
        original[20:110, 20:110, 3] = 255
        garment = np.zeros_like(original)
        garment[20:110, 20:110, :3] = 120 + 2       # each material off by its own
        garment[20:50, 20:110, :3] = 230 - 9
        garment[20:110, 20:110, 3] = 255
        out, shifts = fit_layer_tone({"topwear": garment}, original)
        self.assertGreaterEqual(len(shifts["topwear"]), 2)
        self.assertAlmostEqual(int(out["topwear"][80, 60, 0]), 120, delta=1)
        self.assertAlmostEqual(int(out["topwear"][30, 60, 0]), 230, delta=1)

    def test_a_hidden_part_of_a_material_is_corrected_with_the_rest_of_it(self):
        """It is the same cloth, and a turn may bring it into view."""
        original, layers = self.scene()
        hidden = np.zeros_like(layers["topwear"])
        hidden[60:110, 20:110] = layers["topwear"][60:110, 20:110]
        hidden[115:120, 20:110, :3] = 208           # off-canvas-subject, same cloth
        hidden[115:120, 20:110, 3] = 255
        out, _ = fit_layer_tone({"topwear": hidden, "neck": layers["neck"]}, original)
        self.assertLess(int(out["topwear"][117, 60, 0]), 208)

    def test_only_colour_moves_and_alpha_does_not(self):
        original, layers = self.scene()
        out, _ = fit_layer_tone(layers, original)
        for tag in layers:
            self.assertTrue(np.array_equal(out[tag][..., 3], layers[tag][..., 3]))


class SeamResidualTests(unittest.TestCase):
    """What survives owning the pixel, fitting the coverage and fitting the
    colour: two different generated layers meeting, each right on its own and
    disagreeing where they touch."""

    def scene(self, neck_bias=-2, garment_bias=2):
        original = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
        original[20:110, 20:110, :3] = 200
        original[20:110, 20:110, 3] = 255
        neck = np.zeros_like(original)
        neck[20:64, 20:110, :3] = 200 + neck_bias
        neck[20:64, 20:110, 3] = 255
        garment = np.zeros_like(original)
        garment[64:110, 20:110, :3] = 200 + garment_bias
        garment[64:110, 20:110, 3] = 255
        return original, {"neck": neck, "topwear": garment}

    def test_the_residual_at_the_boundary_is_taken_out(self):
        original, layers = self.scene()
        out, report = fit_seam_residual(layers, original)
        self.assertIn("px", report["topwear|neck"])
        self.assertAlmostEqual(int(out["neck"][63, 60, 0]), 200, delta=1)
        self.assertAlmostEqual(int(out["topwear"][64, 60, 0]), 200, delta=1)

    def test_it_fades_out_rather_than_ending(self):
        """Two levels spread over three pixels is a gradient, which nobody sees.
        The same two levels at a boundary is a line, which is the thing being
        removed -- so the correction must not create an edge of its own."""
        # A larger bias than the real one, so the fade is resolvable in whole
        # levels rather than rounding to the same number three times.
        original, layers = self.scene(neck_bias=-8)
        out, _ = fit_seam_residual(layers, original)
        near = int(out["neck"][63, 60, 0])
        mid = int(out["neck"][62, 60, 0])
        far = int(out["neck"][40, 60, 0])
        self.assertGreater(near, mid)
        self.assertGreater(mid, far)
        self.assertEqual(far, int(layers["neck"][40, 60, 0]))

    def test_alpha_is_not_touched(self):
        original, layers = self.scene()
        out, _ = fit_seam_residual(layers, original)
        for tag in layers:
            self.assertTrue(np.array_equal(out[tag][..., 3], layers[tag][..., 3]))

    def test_pipeline_fits_the_final_boundary_after_orphan_cleanup(self):
        """Cleanup can hand pixels back to neck after a seam fit.

        The residual pass must therefore run on the production layer set, not
        the intermediate one; otherwise that handoff reinstates a thin line.
        """
        original, layers = self.scene()
        identity = lambda current, _original, **_kwargs: (dict(current), {})

        def inject_garment_bias(current, _original):
            changed = {tag: np.array(image, copy=True)
                       for tag, image in current.items()}
            changed["topwear"][64:110, 20:110, :3] = 206
            return changed, {"topwear": {"transferred_px": {"neck": 1}}}

        with patch.object(repair_module, "reclaim_occluded", identity), \
             patch.object(repair_module, "fit_layer_tone", identity), \
             patch.object(repair_module, "fit_edge_alpha", identity), \
             patch.object(repair_module, "clean_garment_orphans", inject_garment_bias):
            result = repair_module.repair_portrait_layers(layers, original)

        self.assertEqual(int(result.layers["topwear"][64, 60, 0]), 200)
        self.assertEqual(result.report["order"], list(REPAIR_ORDER))
        self.assertIn("fit_edge_alpha_final", result.report)


class EdgeFitTests(unittest.TestCase):
    """A layer's edge alpha decides how much of it shows against what is behind,
    and the original says what that mixture should be."""

    def scene(self, rim=(20, 20, 20)):
        original = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
        original[20:100, 20:100, :3] = 220
        original[20:100, 20:100, 3] = 255
        back = np.zeros_like(original)
        back[20:100, 20:100, :3] = 220          # the layer behind is right
        back[20:100, 20:100, 3] = 255
        front = np.zeros_like(original)
        # Big enough to be a surface rather than a stroke: 60x60 is 81% interior.
        front[25:85, 25:85, :3] = 218
        front[25:85, 25:85, 3] = 255
        for d in range(3):                      # ... and dark at its own edge
            front[25 + d, 25:85, :3] = rim
            front[84 - d, 25:85, :3] = rim
            front[25:85, 25 + d, :3] = rim
            front[25:85, 84 - d, :3] = rim
        return original, {"neck": back, "face": front}

    def test_a_rim_the_picture_does_not_have_is_faded_out(self):
        original, layers = self.scene()
        out, moved = fit_edge_alpha(layers, original)
        self.assertGreater(moved.get("face", 0), 100)
        self.assertLess(int(out["face"][26, 50, 3]), 128)

    def test_the_layer_s_interior_is_untouched(self):
        original, layers = self.scene()
        out, _ = fit_edge_alpha(layers, original)
        self.assertEqual(int(out["face"][50, 50, 3]), 255)

    def test_an_edge_that_matches_the_original_is_left_alone(self):
        original, layers = self.scene(rim=(218, 218, 218))
        out, moved = fit_edge_alpha(layers, original)
        self.assertEqual(moved, {})
        self.assertEqual(int(out["face"][26, 50, 3]), 255)

    def test_an_edge_that_shows_too_little_is_raised(self):
        """The other direction, which is why this replaces trimming: the chin,
        where the layer fades over two rows and the original's contour fills
        both."""
        original, layers = self.scene()
        original[26, 25:85, :3] = 20            # the picture's line is solid here
        layers["face"][26, 25:85, 3] = 100      # ... but the layer only half shows
        out, _ = fit_edge_alpha(layers, original)
        self.assertGreater(int(out["face"][26, 50, 3]), 200)

    def test_a_layer_that_is_all_edge_is_left_alone(self):
        """A stroke -- a mouth line, a lash, a nose -- is an outline, and an
        outline is meant to be dark. Refitting one thins it until it fades."""
        original, layers = self.scene()
        stroke = np.zeros_like(layers["face"])
        stroke[48:52, 30:70, :3] = (20, 20, 20)     # four pixels tall: no interior
        stroke[48:52, 30:70, 3] = 255
        layers["mouth"] = stroke
        out, moved = fit_edge_alpha(layers, original)
        self.assertNotIn("mouth", moved)
        self.assertEqual(int(out["mouth"][49, 50, 3]), 255)

    def test_nothing_changes_where_there_is_nothing_behind(self):
        original, layers = self.scene()
        _, moved = fit_edge_alpha({"face": layers["face"]}, original)
        self.assertEqual(moved, {})

    def test_no_layer_s_colour_is_altered(self):
        original, layers = self.scene()
        out, _ = fit_edge_alpha(layers, original)
        for tag in layers:
            self.assertTrue(np.array_equal(out[tag][..., :3], layers[tag][..., :3]))


class CompositeTests(unittest.TestCase):
    """The gap these close: every coverage metric is alpha-only, and the
    `reconstruction` diagnostic copies the original's RGB, so a layer that is
    present and correctly shaped but the wrong colour scores as a clean PASS."""

    def test_composite_reproduces_a_complete_stack(self):
        layers = portrait_layers()
        composite = composite_layers(layers, (CANVAS, CANVAS))
        # Rebuild what the stack should look like: the frontmost layer wins.
        expected = np.zeros((CANVAS, CANVAS, 4), np.uint8)
        for tag in sorted(layers, key=lambda t: SEMANTIC_Z_ORDER.index(t)):
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


def orphan_scene():
    """Garment body plus one head-owned orphan and one real detached button."""
    original = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    original[..., 3] = 255
    original[18:60, 38:90, :3] = 205
    original[60:72, 44:84, :3] = 90
    original[68:124, 20:108, :3] = 90
    original[84:90, 12:18, :3] = 90

    head = rgba([(38, 18, 90, 60)], value=205)
    garment = rgba([(20, 68, 108, 124)], value=90)
    garment[60:72, 44:84, :3] = 90
    garment[60:72, 44:84, 3] = 255
    garment[36:42, 56:66, :3] = 70
    garment[36:42, 56:66, 3] = 255
    garment[84:90, 12:18, :3] = 90
    garment[84:90, 12:18, 3] = 255
    return {"head": head, "topwear": garment}, original


class GarmentOrphanCleanupTests(unittest.TestCase):
    def test_only_the_head_owned_orphan_is_removed(self):
        layers, original = orphan_scene()
        before_input = layers["topwear"].copy()
        out, report = clean_garment_orphans(layers, original)

        self.assertFalse((out["topwear"][36:42, 56:66, 3] > 0).any())
        self.assertTrue((out["topwear"][60:72, 44:84, 3] > 10).all())
        self.assertTrue((out["topwear"][68:124, 20:108, 3] > 10).all())
        self.assertTrue((out["topwear"][84:90, 12:18, 3] > 10).all())
        self.assertGreater(report["topwear"]["removed_px"], 0)
        np.testing.assert_array_equal(layers["topwear"], before_input)

    def test_a_disconnected_button_is_reported_not_deleted(self):
        layers, original = orphan_scene()
        out, report = clean_garment_orphans(layers, original)
        self.assertTrue((out["topwear"][84:90, 12:18, 3] > 10).all())
        self.assertTrue(any(
            row["status"] in {"kept", "ambiguous"}
            and row["bbox_xywh"] == [12, 84, 6, 6]
            for row in report["topwear"]["components"]
        ))

    def test_cleanup_never_regresses_published_composite_fidelity(self):
        layers, original = orphan_scene()
        subject = original[..., 3] > 10
        before = composite_fidelity(
            original, composite_layers(layers, (CANVAS, CANVAS)), subject)
        out, _ = clean_garment_orphans(layers, original)
        after = composite_fidelity(
            original, composite_layers(out, (CANVAS, CANVAS)), subject)
        self.assertLessEqual(after["mae"], before["mae"])
        self.assertLessEqual(after["bad_px"], before["bad_px"])

    def test_redundant_skin_fragment_is_transferred_to_neck_exactly(self):
        neck = rgba([(48, 48, 56, 56)], value=195)
        garment = rgba([(28, 68, 100, 118)], value=90)
        garment[50:54, 50:54, :3] = 205
        garment[50:54, 50:54, 3] = 128
        layers = {"neck": neck, "topwear": garment}
        original = composite_layers(layers, (CANVAS, CANVAS))
        before = composite_layers(layers, (CANVAS, CANVAS))

        out, report = clean_garment_orphans(layers, original)
        after = composite_layers(out, (CANVAS, CANVAS))

        self.assertFalse((out["topwear"][50:54, 50:54, 3] > 10).any())
        np.testing.assert_array_equal(after, before)
        row = next(
            item for item in report["topwear"]["components"]
            if item["bbox_xywh"] == [50, 50, 4, 4]
        )
        self.assertEqual(row["status"], "removed")
        self.assertEqual(row["strategy"], "transfer")
        self.assertEqual(row["transferred_px"], {"neck": 16})
        self.assertEqual(row["rgb_error_delta"], 0)

    def test_a001_neck_topwear_repair_does_not_regress(self):
        layers, original = contested_scene()
        subject = original[..., 3] > 10
        working, _ = reclaim_occluded(layers, original)
        working, _ = fit_layer_tone(working, original)
        working, _ = fit_edge_alpha(working, original)
        before = composite_fidelity(
            original, composite_layers(working, (CANVAS, CANVAS)), subject)
        cleaned, report = clean_garment_orphans(working, original)
        after = composite_fidelity(
            original, composite_layers(cleaned, (CANVAS, CANVAS)), subject)
        self.assertEqual(after, before)
        self.assertEqual(report, {})
        result = repair_portrait_layers(layers, original)
        self.assertEqual(result.report["version"], REPAIR_VERSION)
        self.assertEqual(result.report["order"], list(REPAIR_ORDER))

    def test_neckline_contact_adds_only_local_garment_underlap(self):
        """A collar edge may be present but too transparent over the neck."""
        body = rgba([(10, 70, 118, 110)], value=(60, 70, 90))
        neck = rgba([(42, 30, 86, 80)], value=(210, 170, 150))
        topwear = rgba([(20, 70, 108, 110)], value=(80, 100, 130))
        topwear[68:73, 46:82, :3] = (80, 100, 130)
        topwear[68:73, 46:82, 3] = 80
        layers = {"body_remainder": body, "neck": neck, "topwear": topwear}
        original = composite_layers(layers, (CANVAS, CANVAS))
        original[68:73, 46:82, :3] = (80, 100, 130)
        original[68:73, 46:82, 3] = 255

        out, report = fit_neckline_contact(layers, original)

        assert report["status"] == "applied"
        assert report["accepted_px"] > 0
        assert report["contact_mae_after"] <= report["contact_mae_before"]
        assert report["contact_bad_ratio_after"] <= report["contact_bad_ratio_before"]
        assert int(out["topwear"][70, 50, 3]) > int(topwear[70, 50, 3])
        np.testing.assert_array_equal(out["neck"], neck)
        # The main garment and its collar remain untouched outside the contact
        # neighbourhood; this is not a global dilation.
        np.testing.assert_array_equal(out["topwear"][90:, 20:108], topwear[90:, 20:108])

    def test_neckline_contact_preserves_a_low_neckline(self):
        """Skin wins in the opening, so no garment alpha is invented."""
        body = rgba([(10, 70, 118, 110)], value=(60, 70, 90))
        neck = rgba([(42, 30, 86, 80)], value=(210, 170, 150))
        topwear = rgba([(20, 70, 108, 110)], value=(80, 100, 130))
        topwear[68:73, 46:82, :3] = (80, 100, 130)
        topwear[68:73, 46:82, 3] = 80
        layers = {"body_remainder": body, "neck": neck, "topwear": topwear}
        original = composite_layers(layers, (CANVAS, CANVAS))
        original[68:73, 46:82, :3] = (210, 170, 150)
        original[68:73, 46:82, 3] = 255

        out, report = fit_neckline_contact(layers, original)

        assert report["status"] == "kept"
        np.testing.assert_array_equal(out["topwear"], topwear)
