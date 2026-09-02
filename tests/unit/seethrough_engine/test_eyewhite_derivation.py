import numpy as np

from seethrough_engine.eyewhite_derivation import derive_missing_eyewhite

# Same construction as test_semantic.py's `eye_scene` -- two eyes, a bright
# ring baked into both `original` and `head` (the head/face drawing already
# covers where the sclera should be), a dark iris only in `irides`.
SIDE = 64
EYE_CENTERS = ((18, 22), (42, 46))


def eye_scene():
    original = np.zeros((SIDE, SIDE, 4), np.uint8)
    original[..., :3] = (220, 188, 170)
    original[..., 3] = 255
    head = original.copy()
    irides = np.zeros_like(original)
    for x1, x2 in EYE_CENTERS:
        original[28:36, x1 - 3:x2 + 3, :3] = (245, 242, 238)
        head[28:36, x1 - 3:x2 + 3, :3] = (245, 242, 238)
        original[30:34, x1:x2, :3] = (55, 45, 40)
        head[30:34, x1:x2, :3] = (55, 45, 40)
        irides[30:34, x1:x2, :3] = (55, 45, 40)
        irides[30:34, x1:x2, 3] = 255
    return {"head": head, "irides": irides}, original


def test_derives_a_ground_truth_patch_for_each_visible_sclera():
    layers, original = eye_scene()
    result = derive_missing_eyewhite(layers, original)

    assert result.reason == "ok"
    assert result.iris_components_seen == 2
    assert result.iris_components_used == 2
    assert result.layer is not None
    assert result.derived_px > 0
    assert result.layer.shape == original.shape
    # Every derived pixel is the original's own colour, not a guess.
    derived_mask = result.layer[..., 3] > 0
    np.testing.assert_array_equal(
        result.layer[..., :3][derived_mask], original[..., :3][derived_mask])
    # Nothing is derived at the iris itself -- irides stays the topmost owner.
    iris_mask = layers["irides"][..., 3] > 10
    assert not (derived_mask & iris_mask).any()


def test_dark_eyes_derive_nothing():
    layers, original = eye_scene()
    original[26:38, 12:50, :3] = (70, 55, 50)
    result = derive_missing_eyewhite(layers, original)
    assert result.layer is None
    assert result.reason == "no_ring_passed_the_evidence_gate"
    assert result.iris_components_seen == 2
    assert result.iris_components_used == 0


def test_no_irides_layer_derives_nothing():
    _, original = eye_scene()
    result = derive_missing_eyewhite({"head": np.zeros_like(original)}, original)
    assert result.layer is None
    assert result.reason == "no_irides_layer"


def test_no_head_or_face_support_derives_nothing():
    layers, original = eye_scene()
    del layers["head"]
    result = derive_missing_eyewhite(layers, original)
    assert result.layer is None
    assert result.reason == "no_head_or_face_support"


def test_an_already_present_eyewhite_is_irrelevant_to_derivation():
    """`derive_missing_eyewhite` only ever looks at `irides` + the original --
    an existing (wrong) `eyewhite` layer must not change what it proposes."""
    layers, original = eye_scene()
    without = derive_missing_eyewhite(layers, original)
    layers["eyewhite"] = np.zeros_like(original)
    with_empty_eyewhite = derive_missing_eyewhite(layers, original)
    np.testing.assert_array_equal(without.layer, with_empty_eyewhite.layer)
