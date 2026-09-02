"""Portrait Bundle v1 writer; intentionally unaware of rigs and runtimes."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image

from .generation import PortraitPipelineResult
from .image import composite_fidelity, composite_layers
from .repair import REPAIR_ORDER, REPAIR_VERSION
from .ownership import OWNERSHIP_VERSION
from .seams import RUN_SLACK_PX, seam_report_layers
from .semantic import SEMANTIC_Z_ORDER, semantic_warnings
from .scale import scale_length

BUNDLE_FORMAT = "portrait-bundle"
BUNDLE_VERSION = "1.0"


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _save_rgba(path: str, value: np.ndarray) -> None:
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"bundle image must be HxWx4 RGBA, got {arr.shape}")
    Image.fromarray(arr.astype(np.uint8), mode="RGBA").save(path)


def _nonempty_layers(layer_dict: dict[str, np.ndarray], threshold: int = 10):
    for tag, value in layer_dict.items():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 4 and np.any(arr[..., 3] > threshold):
            yield tag, arr.astype(np.uint8)


def _static_verdict(fidelity: dict[str, Any]) -> str:
    ratio = float(fidelity.get("bad_ratio", 1.0))
    return "pass" if ratio <= 0.03 else "review" if ratio <= 0.10 else "fail"


def save_portrait_bundle(output_dir: str, result: PortraitPipelineResult, *,
                         source_filename: str = "",
                         preserve_raw_layers: bool = True) -> dict[str, Any]:
    """Publish one Portrait Bundle v1 into ``output_dir``."""
    output_dir = os.path.abspath(output_dir)
    for subdir in ("layers", "diagnostics"):
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    if preserve_raw_layers:
        os.makedirs(os.path.join(output_dir, "raw_layers"), exist_ok=True)

    _save_rgba(os.path.join(output_dir, "original.png"), result.fullpage)

    canonical = dict(result.layer_dict)
    remainder = np.asarray(result.guard.body_remainder)
    if remainder.ndim == 3 and remainder.shape[-1] == 4 and np.any(remainder[..., 3] > 10):
        if "body_remainder" in canonical:
            raise ValueError("canonical semantic layers already contain body_remainder")
        canonical["body_remainder"] = remainder

    layer_entries: dict[str, dict[str, str]] = {}
    for tag, arr in _nonempty_layers(canonical):
        relative = f"layers/{tag}.png"
        _save_rgba(os.path.join(output_dir, *relative.split("/")), arr)
        layer_entries[tag] = {"path": relative, "source_tag": tag}

    raw_entries: dict[str, str] = {}
    if preserve_raw_layers:
        for tag, arr in _nonempty_layers(result.raw_layer_dict):
            relative = f"raw_layers/{tag}.png"
            _save_rgba(os.path.join(output_dir, *relative.split("/")), arr)
            raw_entries[tag] = relative

    guard = result.guard
    diagnostic_paths: dict[str, str] = {}
    for name, value in {
        "coverage_mask": guard.generated_union_post_guard,
        "missing_mask": guard.missing_mask,
        "spill_mask": guard.spill_mask,
    }.items():
        relative = f"diagnostics/{name}.png"
        u8 = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(os.path.join(output_dir, *relative.split("/")))
        diagnostic_paths[name] = relative

    reconstruction_relative = "diagnostics/reconstruction.png"
    _save_rgba(os.path.join(output_dir, *reconstruction_relative.split("/")),
               guard.reconstruction_rgba)
    diagnostic_paths["reconstruction"] = reconstruction_relative

    composite = composite_layers(canonical, result.fullpage.shape[:2])
    composite_relative = "diagnostics/layer_composite.png"
    _save_rgba(os.path.join(output_dir, *composite_relative.split("/")), composite)
    diagnostic_paths["layer_composite"] = composite_relative

    fidelity = composite_fidelity(result.fullpage, composite, guard.subject_mask)
    fidelity["semantic_only"] = composite_fidelity(
        result.fullpage,
        composite_layers(result.layer_dict, result.fullpage.shape[:2]),
        guard.subject_mask,
    )
    fidelity["repair"] = dict(result.repair_report)
    fidelity_relative = "diagnostics/fidelity.json"
    _write_json(os.path.join(output_dir, *fidelity_relative.split("/")), fidelity)
    diagnostic_paths["fidelity"] = fidelity_relative

    neckline_relative = "diagnostics/neckline_contact.json"
    _write_json(
        os.path.join(output_dir, *neckline_relative.split("/")),
        dict(result.repair_report.get("fit_neckline_contact") or {}),
    )
    diagnostic_paths["neckline_contact"] = neckline_relative

    ownership_relative = "diagnostics/semantic_ownership.json"
    _write_json(
        os.path.join(output_dir, *ownership_relative.split("/")),
        dict(getattr(result, "ownership_report", {}) or {}),
    )
    diagnostic_paths["semantic_ownership"] = ownership_relative

    local_relative = "diagnostics/local_fidelity.json"
    local_report = dict(result.report.get("local_fidelity") or {})
    _write_json(os.path.join(output_dir, *local_relative.split("/")), local_report)
    diagnostic_paths["local_fidelity"] = local_relative

    error = np.abs(result.fullpage[..., :3].astype(np.int32)
                   - composite[..., :3].astype(np.int32)).sum(axis=2)
    error_relative = "diagnostics/composite_error.png"
    Image.fromarray(np.clip(error, 0, 255).astype(np.uint8), mode="L").save(
        os.path.join(output_dir, *error_relative.split("/")))
    diagnostic_paths["composite_error"] = error_relative

    report = dict(result.report)
    report["semantic"] = dict(report.get("semantic") or {})
    observed_semantic_warnings = list(
        report["semantic"].get("warnings")
        or semantic_warnings(canonical, result.fullpage)
    )
    report["semantic"]["warnings"] = observed_semantic_warnings
    report["composite"] = fidelity
    report["source"] = {**report.get("source", {}), "filename": source_filename}
    report_relative = "diagnostics/portrait_report.json"
    _write_json(os.path.join(output_dir, *report_relative.split("/")), report)
    diagnostic_paths["portrait_report"] = report_relative

    seams_relative = "diagnostics/seams.json"
    seams = seam_report_layers(result.fullpage, canonical)
    longest = max((int(row["longest_run_px"]) for row in seams["seams"]), default=0)
    seam_slack = scale_length(RUN_SLACK_PX, result.fullpage.shape)
    seam_status = "pass" if longest <= seam_slack else "review"
    seams["run_slack_px"] = seam_slack
    seams["status"] = seam_status
    _write_json(os.path.join(output_dir, *seams_relative.split("/")), seams)
    diagnostic_paths["seams"] = seams_relative

    tag_version = str(report.get("source", {}).get("tag_version", ""))
    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "canvas": {
            "width": int(result.fullpage.shape[1]),
            "height": int(result.fullpage.shape[0]),
            "coordinate_system": "top-left-y-down",
            "color_space": "srgb",
            "alpha": "straight",
        },
        "semantics": {
            "schema": "portrait-semantic-tags",
            "version": tag_version,
            "z_order": [tag for tag in SEMANTIC_Z_ORDER if tag in layer_entries],
            "warnings": observed_semantic_warnings,
        },
        "original": "original.png",
        "layers": layer_entries,
        "raw_layers": raw_entries,
        "layer_contract": {
            "canonical_stage": "production_repaired",
            "raw_layers_preserved": bool(preserve_raw_layers),
            "silhouette_guard": bool(report.get("run", {}).get("silhouette_guard", True)),
            "semantic_ownership": {
                "version": OWNERSHIP_VERSION,
                "stage": "post_repair_pre_remainder",
                "report": ownership_relative,
            },
            "fidelity_repair": {
                "version": REPAIR_VERSION,
                "order": list(REPAIR_ORDER),
                "report": fidelity_relative,
            },
        },
        "diagnostics": diagnostic_paths,
        "validation": {
            "static_reconstruction": _static_verdict(fidelity),
            "seams": seam_status,
            "local_fidelity": str(local_report.get("status", "unavailable")),
        },
        "source": {"filename": source_filename},
    }
    _write_json(os.path.join(output_dir, "manifest.json"), manifest)
    return manifest


def save_portrait_run(output_dir: str, base_name: str,
                      result: PortraitPipelineResult, source_filename: str = "",
                      **removed_options) -> dict[str, Any]:
    """Compatibility name while callers migrate to ``save_portrait_bundle``."""
    enabled = [name for name, value in removed_options.items() if value]
    if enabled:
        raise TypeError("Portrait Bundle export cannot create: " + ", ".join(enabled))
    return save_portrait_bundle(output_dir, result, source_filename=source_filename)
