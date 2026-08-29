"""Write a Portrait Mode run (layers, diagnostics, JSON report) to disk for
the standalone webui. Independent of `SeeThrough_SavePSD` in nodes.py, which
additionally builds a browser-side PSD via the ComfyUI frontend and grouped
all-runs data this webui does not need; both write the same filename
conventions (`{base}_{tag}.png`, `{base}_portrait_report.json`, ...) so a
person comparing a ComfyUI run against a webui run recognizes the layout.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image

from . import rig as rig_export
from . import spine as spine_export
from .generation import PortraitPipelineResult

SPINE_SUBDIR = "spine"


def save_portrait_run(output_dir: str, base_name: str, result: PortraitPipelineResult,
                       source_filename: str = "", export_spine: bool = False,
                       spine_version: str = "4.2.28",
                       depth_dict: dict[str, np.ndarray] | None = None,
                       export_rig: bool = False,
                       rig_gradient_tags: tuple[str, ...] = ()) -> dict[str, Any]:
    """Persist one Portrait Mode run. Returns a manifest dict (also written to
    `{base_name}_manifest.json`) with every artifact's filename, relative to
    `output_dir`.

    With `export_spine`, also writes a Spine 2D project under `spine/` -- the
    skeleton JSON next to the cropped per-layer PNGs it references, which is
    the layout the Spine editor opens directly. It goes inside `output_dir` so
    that zipping the run carries it along. Without `depth_dict` the draw order
    is semantic -- see `seethrough_engine.spine.SEMANTIC_Z_ORDER`; pass one
    (from `seethrough_engine.depth.estimate_layer_depths`) to sort by estimated
    depth the way the ComfyUI graph does. The manifest records which was used.

    With `export_rig`, also writes `{base_name}_rig_manifest.json` plus part
    PNGs under `rig/` -- the pseudo-2.5D rig of
    `docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md`, which the browser preview
    consumes. It costs no model and no GPU pass, only numpy and cv2 over
    layers this function already has in hand.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_filename = f"{base_name}_original.png"
    Image.fromarray(result.fullpage).save(os.path.join(output_dir, original_filename))

    layer_files: dict[str, str] = {}
    for tag, img in result.layer_dict.items():
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] != 4 or not np.any(arr[..., 3] > 10):
            continue
        filename = f"{base_name}_{tag}.png"
        Image.fromarray(arr).save(os.path.join(output_dir, filename))
        layer_files[tag] = filename

    guard = result.guard
    diagnostics: dict[str, str] = {}
    diagnostic_arrays = {
        "coverage_mask": guard.generated_union_post_guard,
        "missing_mask": guard.missing_mask,
        "spill_mask": guard.spill_mask,
    }
    for name, arr in diagnostic_arrays.items():
        filename = f"{base_name}_{name}.png"
        u8 = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(os.path.join(output_dir, filename))
        diagnostics[name] = filename

    reconstruction_filename = f"{base_name}_reconstruction.png"
    Image.fromarray(guard.reconstruction_rgba).save(os.path.join(output_dir, reconstruction_filename))
    diagnostics["reconstruction"] = reconstruction_filename

    if np.any(guard.body_remainder[..., 3] > 0):
        remainder_filename = f"{base_name}_body_remainder.png"
        Image.fromarray(guard.body_remainder).save(os.path.join(output_dir, remainder_filename))
        diagnostics["body_remainder"] = remainder_filename

    # Every coverage metric in the report is computed on alpha, so a layer that
    # is present and correctly shaped but the wrong colour scores as well as a
    # right one -- and `reconstruction` above cannot show it either, since it
    # copies the original's RGB. Compositing the layers and diffing against the
    # original is what makes a dropped feature (a missing `eyewhite` leaving
    # skin where the sclera was) visible at all. Numpy-only, no GPU, no model.
    #
    # Composite the *rendered* stack, semantic layers plus the recovered
    # remainder, because that is what the exporters and the rig preview draw.
    # Scoring the semantic layers alone makes every remainder-heavy run look
    # catastrophic for a reason nobody will ever see: on run
    # 20260829_134652 -- 18.6% remainder -- semantic-only reads mae 112.9 while
    # what actually renders is 18.7. The semantic-only figure is still worth
    # keeping alongside, as the measure of how much the remainder is propping
    # the picture up.
    subject_mask = guard.subject_mask
    stack = dict(result.layer_dict)
    stack.update(rig_export.split_remainder(guard.body_remainder, result.layer_dict))

    composite = rig_export.composite_layers(stack, result.fullpage.shape[:2])
    composite_filename = f"{base_name}_layer_composite.png"
    Image.fromarray(composite).save(os.path.join(output_dir, composite_filename))
    diagnostics["layer_composite"] = composite_filename

    fidelity = rig_export.composite_fidelity(result.fullpage, composite, subject_mask)
    fidelity["semantic_only"] = rig_export.composite_fidelity(
        result.fullpage,
        rig_export.composite_layers(result.layer_dict, result.fullpage.shape[:2]),
        subject_mask,
    )

    error = np.abs(result.fullpage[..., :3].astype(np.int32)
                   - composite[..., :3].astype(np.int32)).sum(axis=2)
    error_filename = f"{base_name}_composite_error.png"
    Image.fromarray(np.clip(error, 0, 255).astype(np.uint8), mode="L").save(
        os.path.join(output_dir, error_filename))
    diagnostics["composite_error"] = error_filename

    report = dict(result.report)
    report["composite"] = fidelity
    report["source"] = {**report.get("source", {}), "filename": source_filename}
    report["artifacts"] = {"original": original_filename, "layers": dict(layer_files), **diagnostics}
    report_filename = f"{base_name}_portrait_report.json"
    with open(os.path.join(output_dir, report_filename), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    spine_manifest: dict[str, Any] = {}
    if export_spine:
        parts = spine_export.rename_parts(
            spine_export.layers_to_parts(
                result.layer_dict, original_rgba=result.fullpage,
            body_remainder=result.guard.body_remainder,
                depth_dict=depth_dict,
            )
        )
        project_dir = os.path.join(output_dir, SPINE_SUBDIR)
        json_path = spine_export.write_spine_project(
            project_dir, base_name, parts, result.fullpage.shape[:2],
            spine_version=spine_version,
        )
        spine_manifest = {
            "json": f"{SPINE_SUBDIR}/{os.path.basename(json_path)}",
            "images": f"{SPINE_SUBDIR}/images",
            "slots": list(spine_export.draw_order(parts)),
            "order": "depth" if depth_dict else "semantic",
        }

    rig_manifest: dict[str, Any] = {}
    if export_rig:
        rig_dict, rig_images = rig_export.build_rig(
            result.layer_dict, original_rgba=result.fullpage,
            body_remainder=result.guard.body_remainder,
            depth_dict=depth_dict, frame_size=result.fullpage.shape[:2],
            gradient_tags=rig_gradient_tags, run_id=os.path.basename(output_dir),
            tag_version=str(report.get("source", {}).get("tag_version", "")),
        )
        rig_path = rig_export.write_rig_project(output_dir, base_name, rig_dict, rig_images)
        rig_manifest = {
            "manifest": os.path.basename(rig_path),
            "images": f"{rig_export.RIG_SUBDIR}/images",
            "parts": [part["name"] for part in rig_dict["parts"]],
            "anchors": sorted(rig_dict["anchors"]),
            "depth": rig_dict["source"]["depth"],
            "reclaimed": rig_dict["source"]["reclaimed"],
        }

    manifest = {
        "base": base_name,
        "source_filename": source_filename,
        "width": int(result.fullpage.shape[1]),
        "height": int(result.fullpage.shape[0]),
        "original": original_filename,
        "layers": layer_files,
        "diagnostics": diagnostics,
        "report": report_filename,
        "composite": fidelity,
        "verdict": report["verdict"],
        "recovery_verdict": report["recovery_verdict"],
        "reasons": report["reasons"],
    }
    if spine_manifest:
        manifest["spine"] = spine_manifest
    if rig_manifest:
        manifest["rig"] = rig_manifest
    manifest_filename = f"{base_name}_manifest.json"
    with open(os.path.join(output_dir, manifest_filename), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    manifest["manifest_file"] = manifest_filename

    return manifest
