import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from seethrough_engine.seams import check_run, seam_report, write_baseline

CANVAS = 128


def build_run(directory, *, garment_bias=0, draw_edge=False):
    """A minimal run: skin above, garment below, meeting at y=64.

    `garment_bias` shifts the garment layer off the original, which is the
    fault this measures. `draw_edge` puts a real line in *both* the original and
    the composite at the same place, which is not a fault and must not count.
    """
    os.makedirs(os.path.join(directory, "rig", "images"), exist_ok=True)
    original = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
    original[20:110, 20:110, :3] = 200
    original[20:110, 20:110, 3] = 255
    if draw_edge:
        original[63:65, 20:110, :3] = 40

    layers = {}
    skin = np.zeros_like(original)
    skin[20:64, 20:110, :3] = 200
    skin[20:64, 20:110, 3] = 255
    garment = np.zeros_like(original)
    garment[64:110, 20:110, :3] = np.clip(200 + garment_bias, 0, 255)
    garment[64:110, 20:110, 3] = 255
    if draw_edge:
        skin[63, 20:110, :3] = 40
        garment[64, 20:110, :3] = 40
    layers["neck"] = skin
    layers["topwear"] = garment

    parts = []
    for z, (tag, image) in enumerate(layers.items()):
        ys, xs = np.nonzero(image[:, :, 3] > 0)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        crop = image[box[1]:box[3], box[0]:box[2]]
        Image.fromarray(crop).save(os.path.join(directory, "rig", "images", f"{tag}.png"))
        parts.append({"name": tag, "tag": tag, "image": f"rig/images/{tag}.png",
                      "xyxy": box, "group": "body", "z": z, "depth": 0.5,
                      "weight": {"mode": "constant", "value": 1.0},
                      "mesh": {"cell": 42}})
    manifest = {"version": "0.1", "canvas": {"width": CANVAS, "height": CANVAS},
                "source": {}, "anchors": {}, "parts": parts, "motion": {}}
    with open(os.path.join(directory, "a_rig_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    Image.fromarray(original).save(os.path.join(directory, "a_original.png"))
    return directory


def row_for(report, pair):
    return next((r for r in report["seams"] if r["pair"] == pair), None)


class SeamReportTests(unittest.TestCase):
    def test_layers_that_reproduce_the_picture_have_no_seam(self):
        with tempfile.TemporaryDirectory() as root:
            report = seam_report(build_run(os.path.join(root, "clean")))
            row = row_for(report, "neck | topwear")
            self.assertIsNotNone(row)
            self.assertEqual(row["longest_run_px"], 0)
            self.assertLess(row["mean_excess"], 0.5)

    def test_a_layer_off_by_a_few_levels_draws_a_line(self):
        with tempfile.TemporaryDirectory() as root:
            report = seam_report(build_run(os.path.join(root, "biased"), garment_bias=6))
            row = row_for(report, "neck | topwear")
            self.assertGreater(row["longest_run_px"], 60)
            self.assertGreater(row["mean_excess"], 4)

    def test_an_edge_the_picture_has_too_is_not_a_seam(self):
        """Sharpening an edge that is already an edge changes nothing anyone
        can see, which is why the brow at 70 luma of excess went unnoticed while
        the neck at 2 did not."""
        with tempfile.TemporaryDirectory() as root:
            report = seam_report(build_run(os.path.join(root, "edged"), draw_edge=True))
            row = row_for(report, "neck | topwear")
            self.assertEqual(row["longest_run_px"], 0)

    def test_the_run_survives_a_gap(self):
        """A seam at one or two luma dips below any threshold every few pixels.
        One gap does not make two lines."""
        with tempfile.TemporaryDirectory() as root:
            directory = build_run(os.path.join(root, "gapped"), garment_bias=6)
            path = os.path.join(directory, "rig", "images", "topwear.png")
            image = np.array(Image.open(path))
            image[0, 30:33, :3] = 200          # three pixels of the seam are right
            Image.fromarray(image).save(path)
            row = row_for(seam_report(directory), "neck | topwear")
            self.assertGreater(row["longest_run_px"], 60)


class SeamGuardTests(unittest.TestCase):
    def test_the_same_run_passes_its_own_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            directory = build_run(os.path.join(root, "run"), garment_bias=6)
            baseline = write_baseline(os.path.join(root, "baseline.json"), [directory])
            passed, complaints = check_run(directory, baseline["run"])
            self.assertTrue(passed, complaints)

    def test_a_seam_getting_worse_is_a_complaint(self):
        with tempfile.TemporaryDirectory() as root:
            good = build_run(os.path.join(root, "good"))
            baseline = write_baseline(os.path.join(root, "baseline.json"), [good])
            worse = build_run(os.path.join(root, "worse"), garment_bias=8)
            passed, complaints = check_run(worse, baseline["good"])
            self.assertFalse(passed)
            self.assertTrue(any("neck | topwear" in c for c in complaints), complaints)

    def test_a_seam_that_did_not_exist_before_is_a_complaint(self):
        """The usual way to make one boundary better is to move the fault to
        the next one."""
        with tempfile.TemporaryDirectory() as root:
            baseline = {"seams": []}
            directory = build_run(os.path.join(root, "run"), garment_bias=8)
            passed, complaints = check_run(directory, baseline)
            self.assertFalse(passed)
            self.assertTrue(any("new seam" in c for c in complaints), complaints)

    def test_a_seam_getting_better_is_not_a_complaint(self):
        with tempfile.TemporaryDirectory() as root:
            bad = build_run(os.path.join(root, "bad"), garment_bias=8)
            baseline = write_baseline(os.path.join(root, "baseline.json"), [bad])
            better = build_run(os.path.join(root, "better"))
            passed, _ = check_run(better, baseline["bad"])
            self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
