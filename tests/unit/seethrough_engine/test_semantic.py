import numpy as np

from seethrough_engine.semantic import semantic_warnings


def eye_scene():
    side = 64
    original = np.zeros((side, side, 4), np.uint8)
    original[..., :3] = (220, 188, 170)
    original[..., 3] = 255
    head = original.copy()
    irides = np.zeros_like(original)
    for x1, x2 in ((18, 22), (42, 46)):
        original[28:36, x1 - 3:x2 + 3, :3] = (245, 242, 238)
        head[28:36, x1 - 3:x2 + 3, :3] = (245, 242, 238)
        original[30:34, x1:x2, :3] = (55, 45, 40)
        head[30:34, x1:x2, :3] = (55, 45, 40)
        irides[30:34, x1:x2, :3] = (55, 45, 40)
        irides[30:34, x1:x2, 3] = 255
    return {"head": head, "irides": irides}, original


def test_missing_eyewhite_is_observed_from_the_original_eye_surface():
    layers, original = eye_scene()
    assert semantic_warnings(layers, original) == ["missing_eyewhite"]


def test_an_independent_eyewhite_layer_satisfies_the_observation():
    layers, original = eye_scene()
    eyewhite = np.zeros_like(original)
    eyewhite[28:36, 15:25] = original[28:36, 15:25]
    layers["eyewhite"] = eyewhite
    assert semantic_warnings(layers, original) == []


def test_dark_eye_surfaces_do_not_trigger_a_tag_only_warning():
    layers, original = eye_scene()
    original[26:38, 12:50, :3] = (70, 55, 50)
    assert semantic_warnings(layers, original) == []
