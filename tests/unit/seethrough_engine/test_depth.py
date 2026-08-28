"""Depth batching / v2-slot folding, exercised with a stand-in for Marigold.

Guarded on torch because `seethrough_engine.depth` imports it, and the rest of
`tests/unit` is deliberately runnable without the inference stack installed.
"""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch

    from seethrough_engine import vendor
    from seethrough_engine.depth import COMPOSE_INTO_V2, estimate_layer_depths
    from seethrough_engine.layers import VALID_BODY_PARTS_V2

    vendor.ensure_seethrough_importable()
    vendor.img_alpha_blending  # noqa: B018 -- the vendored tree has to be reachable
    _AVAILABLE = True
except Exception:  # pragma: no cover - environment without torch / see-through
    _AVAILABLE = False


RES = 16


def _solid(box=None, alpha=255):
    img = np.zeros((RES, RES, 4), dtype=np.uint8)
    if box is None:
        img[..., :] = (10, 20, 30, alpha)
    else:
        x1, y1, x2, y2 = box
        img[y1:y2, x1:x2] = (10, 20, 30, alpha)
    return img


class _FakeMarigold:
    """Returns a caller-chosen depth per batch slot and records what it saw."""

    def __init__(self, depth_for_index):
        self._depth_for_index = depth_for_index
        self.seen_batch = None
        self.devices = []

    def to(self, device=None, **kwargs):
        self.devices.append(str(device))
        return self

    def __call__(self, color_map=None, show_progress_bar=False, img_list=None):
        self.seen_batch = img_list
        stack = np.stack([self._depth_for_index(i) for i in range(len(img_list))])
        return SimpleNamespace(depth_tensor=torch.from_numpy(stack))


@unittest.skipUnless(_AVAILABLE, "requires torch and the vendored see-through tree")
class EstimateLayerDepthsTests(unittest.TestCase):
    def _run(self, layer_dict, depth_for_index):
        marigold = _FakeMarigold(depth_for_index)
        depths = estimate_layer_depths(
            marigold, layer_dict, _solid(), RES,
            device=torch.device("cpu"), offload_device=torch.device("cpu"),
        )
        return marigold, depths

    def test_batch_is_one_slot_per_v2_tag_plus_the_composite_page(self):
        marigold, _ = self._run({"topwear": _solid()}, lambda i: np.zeros((RES, RES), np.float32))
        self.assertEqual(len(marigold.seen_batch), len(VALID_BODY_PARTS_V2) + 1)

    def test_marigold_is_returned_to_the_offload_device(self):
        marigold, _ = self._run({"topwear": _solid()}, lambda i: np.zeros((RES, RES), np.float32))
        self.assertEqual(marigold.devices[-1], "cpu")

    def test_plain_v2_tag_gets_its_own_slot_depth(self):
        idx = VALID_BODY_PARTS_V2.index("topwear")
        _, depths = self._run(
            {"topwear": _solid()},
            lambda i: np.full((RES, RES), 0.4 if i == idx else 0.0, np.float32),
        )
        self.assertAlmostEqual(float(np.median(depths["topwear"])), 0.4, places=6)

    def test_v3_eye_layers_are_folded_into_the_v2_eyes_slot_and_handed_back(self):
        members = COMPOSE_INTO_V2["eyes"]
        idx = VALID_BODY_PARTS_V2.index("eyes")
        # Give each eye layer its own disjoint stripe so nothing occludes.
        layer_dict = {
            tag: _solid(box=(0, 2 * n, RES, 2 * n + 2)) for n, tag in enumerate(members)
        }
        _, depths = self._run(
            layer_dict, lambda i: np.full((RES, RES), 0.7 if i == idx else 0.0, np.float32),
        )
        for tag in members:
            self.assertIn(tag, depths, f"{tag} should come back out of the eyes slot")
            visible = layer_dict[tag][..., -1] > 15
            self.assertAlmostEqual(float(np.median(depths[tag][visible])), 0.7, places=6)

    def test_hidden_pixels_take_the_layers_own_visible_median(self):
        # back hair is fully covered by front hair on the left half.
        idx = VALID_BODY_PARTS_V2.index("hair")
        layer_dict = {"back hair": _solid(), "front hair": _solid(box=(0, 0, RES // 2, RES))}
        depth = np.zeros((RES, RES), np.float32)
        depth[:, :RES // 2] = 0.1   # front hair region
        depth[:, RES // 2:] = 0.8   # only back hair visible here

        _, depths = self._run(
            layer_dict, lambda i, d=depth: d if i == idx else np.zeros((RES, RES), np.float32),
        )
        back = depths["back hair"]
        # Where it is hidden it must NOT inherit front hair's 0.1, but its own
        # visible median (0.8) -- otherwise it would sort in front of the hair
        # actually covering it.
        self.assertAlmostEqual(float(np.median(back[:, :RES // 2])), 0.8, places=6)
        self.assertAlmostEqual(float(np.median(back[:, RES // 2:])), 0.8, places=6)

    def test_v3_head_has_no_v2_slot_and_so_gets_no_depth(self):
        # Documented consequence of indexing the batch by VALID_BODY_PARTS_V2:
        # callers must tolerate a depth dict that misses layers.
        self.assertNotIn("head", VALID_BODY_PARTS_V2)
        _, depths = self._run({"head": _solid()}, lambda i: np.zeros((RES, RES), np.float32))
        self.assertNotIn("head", depths)


if __name__ == "__main__":
    unittest.main()
