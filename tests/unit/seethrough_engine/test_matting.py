import unittest

import numpy as np

from seethrough_engine.matting import (
    DEFAULT_TOLERANCE,
    detect_flat_background,
    key_flat_background,
    repair_existing_alpha_edge,
)

SIZE = 96


def scene(bg=(235, 235, 238), *, alpha=None, fg=None):
    """Composite a known foreground over a flat background, the way an image
    model that cannot emit alpha hands us a portrait. Returns
    `(opaque_rgba, ground_truth_alpha, ground_truth_rgb)`."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    if alpha is None:
        r = np.sqrt((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2)
        alpha = np.clip(30 - r, 0, 1).astype(np.float32)  # 1px anti-aliased rim
    if fg is None:
        fg = np.zeros((SIZE, SIZE, 3), np.float32)
        fg[..., 0] = 40 + 60 * (xx / SIZE)
        fg[..., 1] = 30 + 40 * (yy / SIZE)
        fg[..., 2] = 35
    comp = fg * alpha[..., None] + np.asarray(bg, np.float32) * (1 - alpha[..., None])
    opaque = np.dstack([np.rint(comp).astype(np.uint8),
                        np.full((SIZE, SIZE), 255, np.uint8)])
    return opaque, alpha, fg


class DetectTests(unittest.TestCase):
    def test_samples_the_background_colour(self):
        opaque, _, _ = scene(bg=(12, 200, 90))
        found = detect_flat_background(opaque)
        self.assertTrue(found["flat"])
        self.assertEqual(found["color"], [12.0, 200.0, 90.0])
        self.assertEqual(found["std"], 0.0)

    def test_a_gradient_background_is_not_flat(self):
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        ramp = np.dstack([xx * 2, xx * 2, xx * 2]).astype(np.uint8)
        img = np.dstack([ramp, np.full((SIZE, SIZE), 255, np.uint8)])
        found = detect_flat_background(img)
        self.assertFalse(found["flat"])
        self.assertIn("varies", found["reason"])

    def test_a_pillarboxed_upload_samples_the_picture_not_the_bars(self):
        """The real shape of the problem: transparent bars around an opaque
        picture. Reading the canvas border would sample the bars."""
        opaque, _, _ = scene(bg=(200, 40, 40))
        padded = np.zeros((SIZE, SIZE * 2, 4), np.uint8)
        padded[:, SIZE // 2:SIZE // 2 + SIZE] = opaque
        found = detect_flat_background(padded)
        self.assertEqual(found["color"], [200.0, 40.0, 40.0])


class KeyTests(unittest.TestCase):
    def test_existing_alpha_edge_unmixes_premultiplied_background(self):
        bg = np.array([220, 217, 215], np.float32)
        fg = np.array([30, 32, 38], np.float32)
        alpha = np.zeros((SIZE, SIZE), np.float32)
        alpha[20:76, 45:51] = 1.0
        alpha[20:76, 44] = 0.25
        alpha[20:76, 51] = 0.25
        # Simulate a transparent PNG that retained the light background in its
        # soft edge RGB values.
        rgb = np.where(alpha[..., None] > 0, bg, bg)
        rgb[alpha == 1.0] = fg
        rgb[alpha == 0.25] = bg
        rgba = np.dstack([rgb.astype(np.uint8), np.rint(alpha * 255).astype(np.uint8)])
        repaired, info = repair_existing_alpha_edge(rgba)
        self.assertGreater(info["changed_px"], 0)
        self.assertLess(float(np.abs(repaired[alpha == 0.25, :3].astype(np.float32) - fg).mean()), 50.0)

    def test_alpha_matches_the_ground_truth(self):
        opaque, truth, _ = scene()
        out, info = key_flat_background(opaque)
        alpha = out[..., 3].astype(np.float32) / 255.0
        self.assertLess(np.abs(alpha - truth).mean(), 0.01)
        self.assertEqual(info["warnings"], ())

    def test_un_premultiplying_removes_the_background_tint(self):
        """A threshold keeps each edge pixel's share of the background, which
        shows as a rim of background colour on every hair strand. Compare
        against exactly that."""
        opaque, truth, fg = scene(bg=(255, 255, 255))
        out, _ = key_flat_background(opaque)

        edge = (truth > 0.05) & (truth < 0.95)
        keyed_err = np.abs(out[..., :3].astype(np.float32) - fg)[edge].mean()
        naive_err = np.abs(opaque[..., :3].astype(np.float32) - fg)[edge].mean()
        self.assertLess(keyed_err, naive_err / 3.0,
                        f"keyed {keyed_err:.1f} vs threshold-only {naive_err:.1f}")

    def test_thin_strand_uses_nearby_foreground_colour_instead_of_white_halo(self):
        """A two-pixel hair strand has no 7x7 solid interior, but its soft
        edge must still be un-premultiplied from the surrounding background."""
        bg = np.array([232, 231, 229], np.float32)
        fg = np.array([35, 35, 40], np.float32)
        alpha = np.zeros((SIZE, SIZE), np.float32)
        alpha[:, SIZE // 2 - 1:SIZE // 2 + 1] = 1.0
        alpha[:, SIZE // 2 - 2] = 0.25
        alpha[:, SIZE // 2 + 1] = 0.25
        rgb = fg * alpha[..., None] + bg * (1.0 - alpha[..., None])
        opaque = np.dstack([np.rint(rgb).astype(np.uint8),
                            np.full((SIZE, SIZE), 255, np.uint8)])
        out, _ = key_flat_background(opaque)
        edge = alpha == 0.25
        self.assertLess(np.abs(out[..., :3].astype(np.float32) - fg)[edge].mean(), 35.0)

    def test_solid_interior_keeps_its_exact_colour(self):
        opaque, truth, _ = scene()
        out, _ = key_flat_background(opaque)
        solid = truth > 0.999
        np.testing.assert_array_equal(out[..., :3][solid], opaque[..., :3][solid])
        self.assertTrue((out[..., 3][solid] == 255).all())

    def test_enclosed_background_colour_is_kept(self):
        """A white collar against a white background is foreground. Only
        border-connected regions are background."""
        bg = (255, 255, 255)
        alpha = np.zeros((SIZE, SIZE), np.float32)
        alpha[20:76, 20:76] = 1.0
        fg = np.full((SIZE, SIZE, 3), 60, np.float32)
        fg[40:56, 40:56] = bg  # a patch of exactly the background colour, enclosed
        opaque, _, _ = scene(bg=bg, alpha=alpha, fg=fg)
        out, info = key_flat_background(opaque)
        self.assertTrue((out[40:56, 40:56, 3] == 255).all(),
                        "the enclosed patch was punched out")
        self.assertGreater(info["enclosed_px"], 0)
        self.assertTrue(any("enclosed" in w for w in info["warnings"]))

    def test_background_is_fully_transparent(self):
        opaque, truth, _ = scene()
        out, _ = key_flat_background(opaque)
        self.assertTrue((out[..., 3][truth == 0] == 0).all())

    def test_accepts_rgb_without_an_alpha_channel(self):
        opaque, truth, _ = scene()
        out, _ = key_flat_background(opaque[..., :3])
        alpha = out[..., 3].astype(np.float32) / 255.0
        self.assertLess(np.abs(alpha - truth).mean(), 0.01)

    def test_keys_a_pillarboxed_upload_without_eating_the_bars(self):
        opaque, truth, _ = scene()
        padded = np.zeros((SIZE, SIZE * 2, 4), np.uint8)
        padded[:, SIZE // 2:SIZE // 2 + SIZE] = opaque
        out, _ = key_flat_background(padded)
        keyed = out[:, SIZE // 2:SIZE // 2 + SIZE, 3].astype(np.float32) / 255.0
        self.assertLess(np.abs(keyed - truth).mean(), 0.01)
        self.assertTrue((out[:, :SIZE // 2, 3] == 0).all())

    def test_a_subject_running_off_the_edge_is_not_eroded(self):
        """Pixels at the canvas edge are cropped, not anti-aliased, so they
        must stay solid rather than being treated as a soft boundary."""
        alpha = np.zeros((SIZE, SIZE), np.float32)
        alpha[SIZE // 2:, :] = 1.0  # subject fills the bottom half, touching 3 edges
        opaque, _, _ = scene(alpha=alpha)
        out, _ = key_flat_background(opaque)
        self.assertTrue((out[SIZE // 2 + 4:, :, 3] == 255).all())

    def test_a_key_that_removed_nothing_is_reported(self):
        img = np.dstack([np.full((SIZE, SIZE, 3), 90, np.uint8),
                         np.full((SIZE, SIZE), 255, np.uint8)])
        fg = np.full((SIZE, SIZE, 3), 90, np.float32)
        fg[10:20, 10:20] = 200
        img[..., :3] = np.rint(fg).astype(np.uint8)
        out, info = key_flat_background(img)
        self.assertTrue(any("survived" in w for w in info["warnings"]), info["warnings"])

    def test_an_explicit_colour_overrides_detection(self):
        opaque, _, _ = scene(bg=(235, 235, 238))
        out, info = key_flat_background(opaque, color=[0, 0, 0])
        self.assertEqual(info["color"], [0.0, 0.0, 0.0])
        # Nothing matches black, so nothing is removed -- and that is reported.
        self.assertTrue(any("survived" in w for w in info["warnings"]))

    def test_rejects_a_non_image(self):
        with self.assertRaises(ValueError):
            key_flat_background(np.zeros((SIZE, SIZE), np.uint8))

    def test_tolerance_default_is_exposed(self):
        self.assertGreater(DEFAULT_TOLERANCE, 0)


if __name__ == "__main__":
    unittest.main()
