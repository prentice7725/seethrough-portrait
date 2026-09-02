import json

import numpy as np
from PIL import Image

from experimental.source_reprojection import (
    evaluate_source_lock,
    lock_source_pixels,
    load_bundle_layers,
    run_experiment,
)

CANVAS = 16


def _solid(alpha, rgb):
    out = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out


def test_opaque_core_is_replaced_with_the_original_pixel_exactly():
    original = _solid(255, (150, 150, 150))
    layers = {"neck": _solid(255, (90, 90, 90))}  # fully opaque, wrong colour
    result = lock_source_pixels(layers, original)

    np.testing.assert_array_equal(result.layers["neck"][..., :3], np.full((CANVAS, CANVAS, 3), 150))
    assert result.source_locked_mask.all()
    assert result.changed_px["neck"] == CANVAS * CANVAS
    # Alpha is never touched by Source Pixel Lock.
    np.testing.assert_array_equal(result.layers["neck"][..., 3], layers["neck"][..., 3])


def test_semi_transparent_edge_solves_toward_the_true_foreground_colour():
    # original = 0.5 * true_front + 0.5 * back, with a wrong generated front.
    original = _solid(255, (150, 150, 150))
    layers = {
        "neck": _solid(255, (100, 100, 100)),          # behind, opaque
        "topwear": _solid(128, (0, 0, 0)),              # front, alpha ~0.5, wrong colour
    }
    result = lock_source_pixels(layers, original)

    solved_rgb = result.layers["topwear"][..., :3].astype(np.int32)
    np.testing.assert_allclose(solved_rgb, 200, atol=2)
    assert result.source_solved_edge_mask.all()
    assert result.changed_px["topwear"] == CANVAS * CANVAS
    np.testing.assert_array_equal(result.layers["topwear"][..., 3], layers["topwear"][..., 3])


def test_low_behind_contrast_rejects_the_solve():
    # original and the layer behind are nearly identical -- the solve is
    # numerically unreliable here and must be refused, leaving the generated
    # colour exactly as it was.
    original = _solid(255, (101, 101, 101))
    layers = {
        "neck": _solid(255, (100, 100, 100)),
        "topwear": _solid(128, (90, 90, 90)),
    }
    result = lock_source_pixels(layers, original)

    np.testing.assert_array_equal(result.layers["topwear"][..., :3], layers["topwear"][..., :3])
    assert not result.source_solved_edge_mask.any()
    assert result.generated_hidden_mask.all()
    assert result.changed_px["topwear"] == 0


def test_fully_hidden_pixels_are_never_written():
    original = _solid(255, (150, 150, 150))
    layers = {"topwear": _solid(0, (0, 0, 0))}  # no presence at all here
    result = lock_source_pixels(layers, original)

    np.testing.assert_array_equal(result.layers["topwear"], layers["topwear"])
    assert result.changed_px["topwear"] == 0
    assert not result.source_locked_mask.any()
    assert not result.source_solved_edge_mask.any()


def test_evaluate_source_lock_reports_full_hidden_preservation_and_changed_px():
    original = _solid(255, (150, 150, 150))
    before_layers = {"neck": _solid(255, (90, 90, 90))}
    result = lock_source_pixels(before_layers, original)

    report = evaluate_source_lock(original, before_layers, result)

    assert report["hidden_pixel_preservation_ratio"] == 1.0
    assert report["changed_px_total"] == sum(result.changed_px.values())
    assert report["alpha_edge_error"] == "not_applicable_rgb_only"
    assert report["global_mae"]["after"] <= report["global_mae"]["before"]


def _write_bundle(tmp_path, original, layers):
    root = tmp_path / "bundle"
    (root / "layers").mkdir(parents=True)
    Image.fromarray(original, "RGBA").save(root / "original.png")
    layer_entries = {}
    for tag, arr in layers.items():
        Image.fromarray(arr, "RGBA").save(root / "layers" / f"{tag}.png")
        layer_entries[tag] = {"path": f"layers/{tag}.png", "source_tag": tag}
    manifest = {
        "format": "portrait-bundle",
        "version": "1.0",
        "original": "original.png",
        "layers": layer_entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_run_experiment_writes_diagnostics_without_touching_the_bundle(tmp_path):
    original = _solid(255, (150, 150, 150))
    layers = {"neck": _solid(255, (90, 90, 90))}
    bundle_dir = _write_bundle(tmp_path, original, layers)
    original_layer_bytes = (bundle_dir / "layers" / "neck.png").read_bytes()

    output_dir = tmp_path / "ab_output"
    report = run_experiment(bundle_dir, output_dir)

    assert (output_dir / "diagnostics" / "source_reprojection.json").exists()
    assert (output_dir / "diagnostics" / "source_locked_mask.png").exists()
    assert (output_dir / "diagnostics" / "source_solved_edge_mask.png").exists()
    assert (output_dir / "diagnostics" / "generated_hidden_mask.png").exists()
    assert report["source_locked_px"] == CANVAS * CANVAS
    # The saved bundle itself must be untouched.
    assert (bundle_dir / "layers" / "neck.png").read_bytes() == original_layer_bytes


def test_load_bundle_layers_round_trips(tmp_path):
    original = _solid(255, (150, 150, 150))
    layers = {"neck": _solid(255, (90, 90, 90))}
    bundle_dir = _write_bundle(tmp_path, original, layers)

    loaded_original, loaded_layers = load_bundle_layers(bundle_dir)
    np.testing.assert_array_equal(loaded_original, original)
    np.testing.assert_array_equal(loaded_layers["neck"], layers["neck"])
