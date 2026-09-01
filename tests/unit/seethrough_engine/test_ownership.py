import cv2
import numpy as np

from portrait_core.body_remainder import build_body_remainder
from portrait_core.masks import compose_union_alpha
from seethrough_engine.image import composite_layers
from seethrough_engine.ownership import recover_missing_ownership


def rgba(size, color=(0, 0, 0)):
    image = np.zeros((size, size, 4), np.uint8)
    image[..., :3] = color
    return image


def sleeve_scene(size=128, *, sleeve_color=(82, 104, 142)):
    original = rgba(size)
    topwear = rgba(size)
    scale = size / 128.0
    torso = (int(38 * scale), int(54 * scale), int(88 * scale), int(112 * scale))
    sleeve = (int(20 * scale), int(58 * scale), int(39 * scale), int(91 * scale))
    x0, y0, x1, y1 = torso
    topwear[y0:y1, x0:x1, :3] = (82, 104, 142)
    topwear[y0:y1, x0:x1, 3] = 255
    original[y0:y1, x0:x1] = topwear[y0:y1, x0:x1]
    x0, y0, x1, y1 = sleeve
    original[y0:y1, x0:x1, :3] = sleeve_color
    original[y0:y1, x0:x1, 3] = 255
    subject = original[..., 3].astype(np.float32) / 255.0
    return {"topwear": topwear}, original, subject, sleeve


def test_missing_sleeve_continuation_returns_to_topwear_without_changing_render():
    layers, original, subject, sleeve = sleeve_scene()
    before_input = layers["topwear"].copy()
    before_remainder = build_body_remainder(
        original, subject, compose_union_alpha(layers))
    before = composite_layers(
        {**layers, "body_remainder": before_remainder}, original.shape[:2])

    result = recover_missing_ownership(layers, original, subject)
    after_remainder = build_body_remainder(
        original, subject, compose_union_alpha(result.layers))
    after = composite_layers(
        {**result.layers, "body_remainder": after_remainder}, original.shape[:2])

    x0, y0, x1, y1 = sleeve
    assert np.any(result.layers["topwear"][y0:y1, x0:x1, 3] > 10)
    assert result.report["recovered_by_tag"]["topwear"] > 0
    assert result.report["unresolved_remainder_px"] < result.report["initial_missing_px"]
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(layers["topwear"], before_input)


def test_ambiguous_skin_region_stays_in_remainder():
    layers, original, subject, sleeve = sleeve_scene(sleeve_color=(236, 194, 172))
    result = recover_missing_ownership(layers, original, subject)
    x0, y0, x1, y1 = sleeve
    # The final column is the pre-existing contact with the torso; no missing
    # skin pixel before it may be claimed as cloth.
    assert not np.any(result.layers["topwear"][y0:y1, x0:x1 - 1, 3] > 10)
    assert result.report["semantic_recovered_px"] == 0
    assert result.report["unresolved_remainder_px"] == result.report["initial_missing_px"]


def test_nearby_neck_semantic_vetoes_skin_coloured_garment_recovery():
    layers, original, subject, sleeve = sleeve_scene(
        sleeve_color=(220, 185, 170))
    layers["topwear"][..., :3][layers["topwear"][..., 3] > 0] = (190, 180, 170)
    neck = rgba(128)
    neck[48:58, 18:30] = (220, 185, 170, 255)
    layers["neck"] = neck
    original[48:58, 18:30] = neck[48:58, 18:30]
    subject = original[..., 3].astype(np.float32) / 255.0

    result = recover_missing_ownership(layers, original, subject)
    x0, y0, x1, y1 = sleeve
    assert not np.any(result.layers["topwear"][y0:y1, x0:x1 - 1, 3] > 10)
    assert result.report["recovered_by_tag"].get("topwear", 0) == 0


def test_resolution_normalization_keeps_recovery_ratio_stable():
    ratios = []
    for size in (768, 1024):
        layers, original, subject, _ = sleeve_scene(size)
        result = recover_missing_ownership(layers, original, subject)
        ratios.append(
            result.report["semantic_recovered_px"]
            / result.report["initial_missing_px"]
        )
    assert abs(ratios[0] - ratios[1]) < 0.02
