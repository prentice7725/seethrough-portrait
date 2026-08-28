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

from .generation import PortraitPipelineResult


def save_portrait_run(output_dir: str, base_name: str, result: PortraitPipelineResult,
                       source_filename: str = "") -> dict[str, Any]:
    """Persist one Portrait Mode run. Returns a manifest dict (also written to
    `{base_name}_manifest.json`) with every artifact's filename, relative to
    `output_dir`."""
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

    report = dict(result.report)
    report["source"] = {**report.get("source", {}), "filename": source_filename}
    report["artifacts"] = {"original": original_filename, "layers": dict(layer_files), **diagnostics}
    report_filename = f"{base_name}_portrait_report.json"
    with open(os.path.join(output_dir, report_filename), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    manifest = {
        "base": base_name,
        "source_filename": source_filename,
        "width": int(result.fullpage.shape[1]),
        "height": int(result.fullpage.shape[0]),
        "original": original_filename,
        "layers": layer_files,
        "diagnostics": diagnostics,
        "report": report_filename,
        "verdict": report["verdict"],
        "recovery_verdict": report["recovery_verdict"],
        "reasons": report["reasons"],
    }
    manifest_filename = f"{base_name}_manifest.json"
    with open(os.path.join(output_dir, manifest_filename), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    manifest["manifest_file"] = manifest_filename

    return manifest
