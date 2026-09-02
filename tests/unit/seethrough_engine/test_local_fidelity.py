import numpy as np

from seethrough_engine.local_fidelity import local_fidelity_report


def eye_scene(*, closed=False):
    size = 192
    skin = np.zeros((size, size, 4), np.uint8)
    skin[25:170, 30:162] = (232, 190, 170, 255)
    original = skin.copy()
    irides = np.zeros_like(original)
    for x in (70, 122):
        if closed:
            original[74:78, x - 12:x + 12, :3] = 40
            irides[74:78, x - 3:x + 3] = (40, 35, 32, 255)
        else:
            original[68:82, x - 15:x + 15] = (244, 242, 238, 255)
            original[70:80, x - 4:x + 4] = (45, 38, 35, 255)
            irides[70:80, x - 4:x + 4] = (45, 38, 35, 255)
    mouth = np.zeros_like(original)
    mouth[120:125, 80:112] = (120, 50, 55, 255)
    original[120:125, 80:112] = mouth[120:125, 80:112]
    return original, {"head": skin, "irides": irides, "mouth": mouth}


def test_eye_local_guard_catches_visible_sclera_loss():
    original, layers = eye_scene()
    bad = original.copy()
    bad[65:85, 50:145, :3] = (232, 190, 170)
    report = local_fidelity_report(original, bad, layers)
    assert report["status"] == "review"
    assert report["warnings"] == ["missing_visible_eyewhite"]
    assert all(eye["sclera"]["lost_ratio"] > 0.5 for eye in report["eyes"])


def test_preserved_sclera_passes_even_without_an_eyewhite_tag():
    original, layers = eye_scene()
    report = local_fidelity_report(original, original, layers)
    assert report["status"] == "pass"
    assert report["warnings"] == []


def test_closed_eye_does_not_create_a_false_sclera_warning():
    original, layers = eye_scene(closed=True)
    report = local_fidelity_report(original, original, layers)
    assert report["status"] == "pass"
    assert report["warnings"] == []
    assert all(not eye["sclera"]["visible_in_original"] for eye in report["eyes"])


def neckline_scene():
    size = 192
    original = np.zeros((size, size, 4), np.uint8)
    original[..., 3] = 255
    neck = np.zeros_like(original)
    neck[28:100, 70:122] = (218, 178, 158, 255)
    topwear = np.zeros_like(original)
    topwear[94:168, 38:154] = (86, 104, 138, 255)
    # The original has a clean, faithful neckline handoff.
    original[neck[..., 3] > 0] = neck[neck[..., 3] > 0]
    original[topwear[..., 3] > 0] = topwear[topwear[..., 3] > 0]
    return original, {"neck": neck, "topwear": topwear}


def test_neckline_contact_guard_catches_a_horizontal_static_seam():
    original, layers = neckline_scene()
    seammed = original.copy()
    seammed[94:100, 68:124, :3] = (40, 40, 40)
    report = local_fidelity_report(original, seammed, layers)
    assert report["status"] == "review"
    assert report["neckline"]["status"] == "review"
    assert report["neckline"]["max_bad_row_px"] > 0
    assert "neckline_local_fidelity_regression" in report["warnings"]


def test_faithful_neckline_contact_passes():
    original, layers = neckline_scene()
    report = local_fidelity_report(original, original, layers)
    assert report["neckline"]["status"] == "pass"
