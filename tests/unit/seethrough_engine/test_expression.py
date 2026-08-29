import os
import tempfile
import unittest

import numpy as np

from seethrough_engine.expression import (
    BOUNDARY_CORRECTION,
    build_expression_pack,
    drift_level,
    expression_rois,
    extract_part,
    infer_edit_mask,
    write_expression_pack,
)

CANVAS = 256
SKIN = (232, 202, 182)
SCLERA = (245, 245, 250)
IRIS = (40, 40, 60)
LIP = (150, 70, 80)

# One crude portrait: a slab of subject with two eyes and a mouth drawn on it.
# Coordinates are chosen so the eye and mouth regions do not touch once grown.
EYE_L = {"eyewhitel": (92, 100, 118, 118), "iridesl": (100, 102, 110, 116),
         "eyelashl": (90, 96, 120, 120)}
EYE_R = {"eyewhiter": (140, 100, 166, 118), "iridesr": (148, 102, 158, 116),
         "eyelashr": (138, 96, 168, 120)}
MOUTH = {"mouth": (116, 164, 142, 176)}
PART_BOXES = {**EYE_L, **EYE_R, **MOUTH}
ANCHORS = {"eye_left": (105, 109), "eye_right": (153, 109), "mouth": (129, 170)}


def base_portrait():
    img = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
    img[16:CANVAS, 56:200, :3] = SKIN
    img[16:CANVAS, 56:200, 3] = 255
    for boxes in (EYE_L, EYE_R):
        x1, y1, x2, y2 = boxes["eyewhitel" if "eyewhitel" in boxes else "eyewhiter"]
        img[y1:y2, x1:x2, :3] = SCLERA
        ix1, iy1, ix2, iy2 = boxes["iridesl" if "iridesl" in boxes else "iridesr"]
        img[iy1:iy2, ix1:ix2, :3] = IRIS
    mx1, my1, mx2, my2 = MOUTH["mouth"]
    img[my1:my2, mx1:mx2, :3] = LIP
    return img


def with_closed_eyes(base):
    """Both eyes painted out and replaced by a dark lid line -- the drawing the
    decomposition cannot produce."""
    donor = base.copy()
    for boxes in (EYE_L, EYE_R):
        lash = boxes["eyelashl" if "eyelashl" in boxes else "eyelashr"]
        x1, y1, x2, y2 = lash
        donor[y1:y2, x1:x2, :3] = SKIN
        donor[112:116, x1 + 2:x2 - 2, :3] = (30, 25, 30)
    return donor


def with_open_mouth(base):
    donor = base.copy()
    mx1, my1, mx2, my2 = MOUTH["mouth"]
    donor[my1:my2, mx1:mx2, :3] = SKIN
    donor[162:184, mx1:mx2, :3] = (70, 25, 35)
    return donor


def rois():
    return expression_rois(ANCHORS, PART_BOXES, (CANVAS, CANVAS))


def roi_named(name):
    return next(r for r in rois() if r.name == name)


class RoiTests(unittest.TestCase):
    def test_regions_come_from_this_run_s_own_boxes(self):
        by_name = {r.name: r for r in rois()}
        self.assertEqual(set(by_name), {"eye_l", "eye_r", "mouth"})
        for name, anchor_key in (("eye_l", "eye_left"), ("eye_r", "eye_right"),
                                 ("mouth", "mouth")):
            x1, y1, x2, y2 = by_name[name].box
            ax, ay = ANCHORS[anchor_key]
            self.assertTrue(x1 <= ax <= x2 and y1 <= ay <= y2,
                            f"{name} does not contain its own anchor")

    def test_a_region_is_larger_than_the_layer_it_grew_from(self):
        eye = roi_named("eye_l").box
        lash = EYE_L["eyelashl"]
        self.assertLess(eye[1], lash[1])
        self.assertGreater(eye[3], lash[3])

    def test_the_two_eyes_cannot_claim_each_other_s_pixels(self):
        left, right = roi_named("eye_l").box, roi_named("eye_r").box
        self.assertLessEqual(left[2], right[0])

    def test_eyes_and_mouth_do_not_overlap(self):
        for eye in ("eye_l", "eye_r"):
            self.assertLessEqual(roi_named(eye).box[3], roi_named("mouth").box[1])

    def test_a_missing_layer_drops_its_region_rather_than_guessing(self):
        only_mouth = expression_rois(ANCHORS, MOUTH, (CANVAS, CANVAS))
        self.assertEqual([r.name for r in only_mouth], ["mouth"])


class NoOpTests(unittest.TestCase):
    def test_an_unchanged_donor_recovers_nothing(self):
        base = base_portrait()
        pack = build_expression_pack(base, {"eye_closed": base.copy()},
                                     ANCHORS, PART_BOXES)
        self.assertEqual(pack["parts"], [])
        for entry in pack["report"]["states"]["eye_closed"].values():
            self.assertFalse(entry.get("recovered", False))

    def test_an_unchanged_region_reports_why_it_was_empty(self):
        base = base_portrait()
        mask, diagnostics = infer_edit_mask(base, base.copy(), roi_named("eye_l"))
        self.assertFalse(mask.any())
        self.assertEqual(diagnostics["reason"], "no core")

    def test_a_region_the_donor_left_alone_stays_empty(self):
        # The mouth donor changed the mouth; the eyes are untouched and must
        # come back empty even though they are searched.
        base = base_portrait()
        mask, _ = infer_edit_mask(base, with_open_mouth(base), roi_named("eye_l"))
        self.assertFalse(mask.any())


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.base = base_portrait()

    def test_a_closed_eye_donor_yields_one_part_per_eye(self):
        pack = build_expression_pack(self.base, {"eye_closed": with_closed_eyes(self.base)},
                                     ANCHORS, PART_BOXES)
        names = sorted(p.name for p in pack["parts"])
        self.assertEqual(names, ["eye_closed_l", "eye_closed_r"])

    def test_each_recovered_part_stays_inside_its_own_region(self):
        pack = build_expression_pack(self.base, {"eye_closed": with_closed_eyes(self.base)},
                                     ANCHORS, PART_BOXES)
        for part in pack["parts"]:
            region = roi_named(f"eye_{part.side}").box
            x1, y1, x2, y2 = part.xyxy
            self.assertTrue(region[0] <= x1 and x2 <= region[2]
                            and region[1] <= y1 and y2 <= region[3],
                            f"{part.name} {part.xyxy} escaped {region}")

    def test_the_recovered_eye_covers_the_drawing_it_came_from(self):
        part = extract_part(self.base, with_closed_eyes(self.base), roi_named("eye_l"),
                            "eye_closed_l")
        x1, y1, x2, y2 = part.xyxy
        # the lid line the donor drew, at y 112-116 across the lash box
        self.assertLessEqual(y1, 112)
        self.assertGreaterEqual(y2, 116)
        self.assertLessEqual(x1, EYE_L["eyelashl"][0] + 4)
        self.assertGreaterEqual(x2, EYE_L["eyelashl"][2] - 4)

    def test_an_open_mouth_donor_yields_one_part(self):
        pack = build_expression_pack(self.base, {"mouth_open": with_open_mouth(self.base)},
                                     ANCHORS, PART_BOXES)
        self.assertEqual([p.name for p in pack["parts"]], ["mouth_open"])
        part = pack["parts"][0]
        # the donor's mouth reaches to y 184, well below the closed mouth's box
        self.assertGreaterEqual(part.xyxy[3], 180)

    def test_a_donor_is_searched_only_where_its_kind_lives(self):
        # A mouth donor whose eyes also drifted badly must not rewrite the eyes.
        donor = with_open_mouth(self.base)
        donor[100:118, 92:118, :3] = (10, 10, 10)
        pack = build_expression_pack(self.base, {"mouth_open": donor}, ANCHORS, PART_BOXES)
        self.assertEqual([p.name for p in pack["parts"]], ["mouth_open"])

    def test_a_part_carries_its_own_alpha_and_is_not_a_rectangle(self):
        part = extract_part(self.base, with_closed_eyes(self.base), roi_named("eye_l"),
                            "eye_closed_l")
        alpha = part.image[:, :, 3]
        self.assertGreater(int((alpha > 0).sum()), 0)
        self.assertLess(int((alpha == 255).sum()), alpha.size,
                        "the sprite is fully opaque to its own bounding box: "
                        "the feather did nothing")


class DriftTests(unittest.TestCase):
    """The reason this is not one threshold: a generated donor differs from the
    base everywhere, and the difference in the eye has to be told apart from the
    difference on the cheek next to it."""

    def setUp(self):
        self.base = base_portrait()

    def test_uniform_drift_is_measured_where_nothing_is_taken(self):
        donor = with_closed_eyes(self.base).astype(np.int16)
        donor[:, :, :3] = np.clip(donor[:, :, :3] - 25, 0, 255)
        donor = donor.astype(np.uint8)
        self.assertAlmostEqual(drift_level(self.base, donor, rois()), 25, delta=2)

    def test_a_drifting_donor_does_not_hand_over_the_whole_region(self):
        donor = with_closed_eyes(self.base).astype(np.int16)
        donor[:, :, :3] = np.clip(donor[:, :, :3] - 25, 0, 255)
        donor = donor.astype(np.uint8)
        region = roi_named("eye_l")
        area = (region.box[2] - region.box[0]) * (region.box[3] - region.box[1])
        naive, _ = infer_edit_mask(self.base, donor, region, drift=0)
        measured, diagnostics = infer_edit_mask(self.base, donor, region,
                                                drift=drift_level(self.base, donor, rois()))
        self.assertGreater(int((naive > 0).sum()), area * 0.9,
                           "the naive threshold was expected to swallow the region")
        self.assertLess(int((measured > 0).sum()), area * 0.5)
        self.assertGreater(diagnostics["extent_threshold"], 25)

    def test_an_edit_outside_every_region_is_ignored(self):
        donor = with_closed_eyes(self.base)
        donor[210:240, 60:90, :3] = (10, 10, 10)     # something redrawn on the torso
        pack = build_expression_pack(self.base, {"eye_closed": donor}, ANCHORS, PART_BOXES)
        self.assertEqual(sorted(p.name for p in pack["parts"]),
                         ["eye_closed_l", "eye_closed_r"])
        for part in pack["parts"]:
            self.assertLess(part.xyxy[3], 210)

    def test_that_edit_does_not_poison_the_drift_estimate(self):
        donor = with_closed_eyes(self.base)
        donor[210:240, 60:90, :3] = (10, 10, 10)
        self.assertLess(drift_level(self.base, donor, rois()), 10)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.base = base_portrait()

    def test_a_donor_half_a_step_off_is_pulled_back_toward_the_base(self):
        shifted = with_closed_eyes(self.base).astype(np.int16)
        shifted[:, :, :3] = np.clip(shifted[:, :, :3] - 20, 0, 255)
        donor = shifted.astype(np.uint8)
        region = roi_named("eye_l")
        part = extract_part(self.base, donor, region, "eye_closed_l",
                            drift=drift_level(self.base, donor, rois()))
        self.assertIsNotNone(part)
        delta = part.diagnostics["boundary_delta"]
        self.assertTrue(all(d > 10 for d in delta), delta)
        # The skin inside the sprite should have moved most of the way back.
        x1, y1, _, _ = part.xyxy
        corrected = int(part.image[0, 0, 0]) if part.image[0, 0, 3] else None
        expected = int(round(donor[y1, x1, 0] + delta[0] * BOUNDARY_CORRECTION))
        if corrected is not None:
            self.assertAlmostEqual(corrected, expected, delta=1)

    def test_nothing_is_added_outside_the_base_silhouette(self):
        donor = with_closed_eyes(self.base)
        # the donor grows the character sideways, past where the base has alpha
        donor[100:120, 40:56, :3] = (10, 10, 10)
        donor[100:120, 40:56, 3] = 255
        part = extract_part(self.base, donor, roi_named("eye_l"), "eye_closed_l")
        x1, y1, x2, y2 = part.xyxy
        outside = self.base[y1:y2, x1:x2, 3] == 0
        self.assertEqual(int(part.image[:, :, 3][outside].sum()), 0)


class WriteTests(unittest.TestCase):
    def test_the_pack_writes_its_images_and_names_them_relatively(self):
        base = base_portrait()
        pack = build_expression_pack(base, {"eye_closed": with_closed_eyes(base)},
                                     ANCHORS, PART_BOXES)
        with tempfile.TemporaryDirectory() as out:
            block = write_expression_pack(out, pack)
            self.assertEqual(block["version"], "0.1")
            for name, entry in block["parts"].items():
                self.assertEqual(entry["image"], f"rig/images/{name}.png")
                self.assertTrue(os.path.isfile(os.path.join(out, "rig", "images", f"{name}.png")))
                self.assertIn(entry["side"], ("l", "r"))


if __name__ == "__main__":
    unittest.main()
