"""The v3 head pass runs on its own square canvas, so its resolution can be
set independently of the body pass. That matters because the head canvas is
what decides whether the fine facial layers resolve at all: measured on A-001,
512 returns no `eyewhite` and composites to mae 15.1, while 768 returns it and
drops to 11.9. Running the body at 512 and the head at 768 buys the detail
without paying for a full-resolution body pass.

These exercise the geometry with a stand-in pipeline, so they run without a
GPU or a model: what has to hold is that each stage receives the canvas it was
asked for and that the head output still lands in the right place on the body
canvas afterwards.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from seethrough_engine import vendor
from seethrough_engine.generation import run_diffusion_stage
from seethrough_engine.layers import VALID_BODY_PARTS_V3_BODY, VALID_BODY_PARTS_V3_HEAD

BODY_RES = 128
HEAD_RES = 256


class RecordingPipeline:
    """Returns a layer per tag and records the canvas each stage was handed."""

    def __init__(self, head_box=(40, 30, 90, 80)):
        self.calls = []
        self.head_box = head_box

    def __call__(self, *, fullpage, group_index=None, **kwargs):
        self.calls.append({"group_index": group_index, "size": fullpage.shape[:2]})
        size = fullpage.shape[:2]
        if group_index == 0:
            images = []
            for tag in VALID_BODY_PARTS_V3_BODY:
                img = np.zeros((*size, 4), np.uint8)
                if tag == "head":
                    x1, y1, x2, y2 = self.head_box
                    img[y1:y2, x1:x2] = 255
                images.append(img)
            return SimpleNamespace(images=images)
        images = []
        for _ in VALID_BODY_PARTS_V3_HEAD:
            img = np.full((*size, 4), 200, np.uint8)
            images.append(img)
        return SimpleNamespace(images=images)


def run(head_resolution=None, *, body_res=BODY_RES):
    vendor.ensure_seethrough_importable()
    source = np.zeros((body_res, body_res, 4), np.uint8)
    source[10:body_res - 10, 20:body_res - 20] = 255
    fullpage, pad_size, pad_pos = vendor.center_square_pad_resize(
        source, body_res, return_pad_info=True)
    scale = pad_size[0] / body_res
    pipeline = RecordingPipeline()
    layers = run_diffusion_stage(
        pipeline, None, None, "v3", 1, fullpage,
        enable_head_detail=True, input_img=source, scale=scale, pad_pos=pad_pos,
        resolution=body_res, head_resolution=head_resolution,
    )
    return pipeline, layers


class HeadResolutionTests(unittest.TestCase):
    def test_head_pass_uses_its_own_canvas(self):
        pipeline, _ = run(head_resolution=HEAD_RES)
        body, head = pipeline.calls
        self.assertEqual(body["group_index"], 0)
        self.assertEqual(body["size"], (BODY_RES, BODY_RES))
        self.assertEqual(head["group_index"], 1)
        self.assertEqual(head["size"], (HEAD_RES, HEAD_RES),
                         "the head stage did not get the resolution it was asked for")

    def test_omitting_it_keeps_both_stages_together(self):
        """Default behaviour must not move: None means 'same as the body'."""
        pipeline, _ = run(head_resolution=None)
        self.assertEqual([c["size"] for c in pipeline.calls],
                         [(BODY_RES, BODY_RES), (BODY_RES, BODY_RES)])

    def test_every_layer_comes_back_on_the_body_canvas(self):
        """The head runs bigger, but its output has to land on the body canvas
        -- `canvas` is the one place the body resolution still rules."""
        _, layers = run(head_resolution=HEAD_RES)
        for tag, img in layers.items():
            self.assertEqual(img.shape, (BODY_RES, BODY_RES, 4), tag)

    def test_head_layers_land_in_the_same_place_at_either_resolution(self):
        """`center_square_pad_resize` derives pad_size/pad_pos from the source
        before resizing, so they carry no dependence on the target size and the
        mapping back is unchanged. If that ever stopped holding, the facial
        layers would drift when the head resolution changed."""
        _, same = run(head_resolution=None)
        _, bigger = run(head_resolution=HEAD_RES)
        for tag in VALID_BODY_PARTS_V3_HEAD:
            a = same[tag][..., 3] > 10
            b = bigger[tag][..., 3] > 10
            self.assertTrue(a.any(), tag)
            box_a = (np.flatnonzero(a.any(1))[[0, -1]], np.flatnonzero(a.any(0))[[0, -1]])
            box_b = (np.flatnonzero(b.any(1))[[0, -1]], np.flatnonzero(b.any(0))[[0, -1]])
            np.testing.assert_allclose(box_a[0], box_b[0], atol=2, err_msg=f"{tag} rows")
            np.testing.assert_allclose(box_a[1], box_b[1], atol=2, err_msg=f"{tag} cols")

    def test_body_layers_are_untouched_by_the_head_resolution(self):
        _, same = run(head_resolution=None)
        _, bigger = run(head_resolution=HEAD_RES)
        for tag in VALID_BODY_PARTS_V3_BODY:
            np.testing.assert_array_equal(same[tag], bigger[tag], tag)


if __name__ == "__main__":
    unittest.main()
