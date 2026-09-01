import numpy as np

from seethrough_engine.seams import compare_seam_report, seam_report_layers

CANVAS = 128


def build_scene(*, garment_bias=0, draw_edge=False, busy=False):
    original = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
    original[20:110, 20:110, :3] = 200
    original[20:110, 20:110, 3] = 255
    if busy:
        original[20:110, 20:110:3, :3] = 60
    if draw_edge:
        original[63:65, 20:110, :3] = 40

    neck = np.zeros_like(original)
    neck[20:64, 20:110, :3] = 200
    neck[20:64, 20:110, 3] = 255
    topwear = np.zeros_like(original)
    topwear[64:110, 20:110, :3] = np.clip(200 + garment_bias, 0, 255)
    topwear[64:110, 20:110, 3] = 255
    if busy:
        neck[20:64, 20:110:3, :3] = 60
        topwear[64:110, 20:110:3, :3] = np.clip(60 + garment_bias, 0, 255)
    if draw_edge:
        neck[63, 20:110, :3] = 40
        topwear[64, 20:110, :3] = 40
    return original, {"neck": neck, "topwear": topwear}


def report(**kwargs):
    original, layers = build_scene(**kwargs)
    return seam_report_layers(original, layers)


def row_for(value, pair="neck | topwear"):
    return next((row for row in value["seams"] if row["pair"] == pair), None)


def test_layers_that_reproduce_the_picture_have_no_seam():
    row = row_for(report())
    assert row["longest_run_px"] == 0
    assert row["mean_excess"] < 0.5


def test_a_layer_off_by_a_few_levels_draws_a_line():
    row = row_for(report(garment_bias=6))
    assert row["longest_run_px"] > 60
    assert row["mean_excess"] > 4


def test_an_edge_the_picture_has_too_is_not_a_seam():
    assert row_for(report(draw_edge=True))["longest_run_px"] == 0


def test_the_run_survives_a_gap():
    original, layers = build_scene(garment_bias=6)
    # Geometry is normalized to a 768 px reference. On this 128 px synthetic
    # canvas one pixel represents the small interruption the old three-pixel
    # fixture represented at production resolution.
    layers["topwear"][64, 50:51, :3] = 200
    row = row_for(seam_report_layers(original, layers))
    assert row["longest_run_px"] > 60


def test_the_same_error_inside_a_busy_region_is_not_flagged():
    quiet = row_for(report(garment_bias=6))
    busy = row_for(report(garment_bias=6, busy=True))
    assert quiet["longest_run_px"] > 60
    assert busy["longest_run_px"] < 10


def test_same_report_passes_its_baseline():
    value = report(garment_bias=6)
    passed, complaints = compare_seam_report(value, value)
    assert passed, complaints


def test_a_new_or_worse_seam_is_a_complaint():
    clean = report()
    worse = report(garment_bias=8)
    passed, complaints = compare_seam_report(worse, clean)
    assert not passed
    assert any("neck | topwear" in complaint for complaint in complaints)


def test_a_seam_getting_better_is_not_a_complaint():
    passed, _ = compare_seam_report(report(), report(garment_bias=8))
    assert passed
