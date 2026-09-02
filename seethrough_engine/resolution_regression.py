"""Portrait Bundle A/B snapshots for resolution-dependent semantic regressions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .local_fidelity import local_fidelity_report


def bundle_resolution_snapshot(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads(
        (root / manifest["diagnostics"]["portrait_report"]).read_text(encoding="utf-8")
    )
    original = np.array(Image.open(root / manifest["original"]).convert("RGBA"))
    composite = np.array(Image.open(
        root / manifest["diagnostics"]["layer_composite"]
    ).convert("RGBA"))
    semantic_layers = {
        tag: np.array(Image.open(root / info["path"]).convert("RGBA"))
        for tag, info in manifest["layers"].items()
        if tag != "body_remainder"
    }
    local = local_fidelity_report(original, composite, semantic_layers)
    cleanup = (
        report.get("composite", {}).get("repair", {})
        .get("clean_garment_orphans", {}).get("topwear", {})
    )
    ambiguous = [
        row for row in cleanup.get("components", [])
        if row.get("status") == "ambiguous"
    ]
    return {
        "resolution": int(report["run"]["resolution"]),
        "head_resolution": int(report["run"].get(
            "head_resolution", report["run"]["resolution"])),
        "seed": int(report["run"]["seed"]),
        "steps": int(report["run"]["steps"]),
        "eyewhite_present": "eyewhite" in manifest["layers"],
        # A neckline review must not masquerade as an eye/sclera regression.
        # This field is deliberately eye-specific; the complete local status
        # remains available below for callers that need all critical features.
        "sclera_preserved": bool(local["eyes"]) and not any(
            eye["status"] == "review" for eye in local["eyes"]
        ),
        "eye_local_fidelity": local,
        "neckline_local_fidelity": local.get("neckline"),
        "topwear_contamination": {
            "removed_px": int(cleanup.get("removed_px", 0)),
            "ambiguous_components": len(ambiguous),
            "ambiguous_area_px": int(sum(row.get("area_px", 0) for row in ambiguous)),
        },
        "initial_missing_px": int(report["coverage"]["missing_area_px"]),
        "unresolved_remainder_ratio": float(report["coverage"]["recovered_ratio"]),
        "semantic_tag_count": len(manifest["layers"]),
        "global_composite_fidelity": {
            key: report["composite"][key]
            for key in ("mae", "bad_ratio", "bad_px", "subject_px")
        },
        "semantic_only_mae": float(report["composite"]["semantic_only"]["mae"]),
        "semantic_warnings": list(manifest["semantics"].get("warnings", [])),
    }


def compare_resolution_snapshots(baseline: dict[str, Any],
                                 comparison: dict[str, Any]) -> dict[str, Any]:
    """Identify semantic/local regressions that a global metric can hide."""
    regressions: list[str] = []
    if baseline["eyewhite_present"] and not comparison["eyewhite_present"]:
        regressions.append("eyewhite_tag_lost")
    if baseline["sclera_preserved"] and not comparison["sclera_preserved"]:
        regressions.append("eye_local_fidelity_regressed")
    baseline_neckline = baseline.get("neckline_local_fidelity") or {}
    comparison_neckline = comparison.get("neckline_local_fidelity") or {}
    if (baseline_neckline.get("status") == "pass"
            and comparison_neckline.get("status") == "review"):
        regressions.append("neckline_local_fidelity_regressed")
    if comparison["semantic_tag_count"] < baseline["semantic_tag_count"]:
        regressions.append("semantic_tag_count_decreased")
    new_warnings = sorted(set(comparison["semantic_warnings"])
                          - set(baseline["semantic_warnings"]))
    if new_warnings:
        regressions.append("semantic_warnings_added")
    return {
        "status": "regression" if regressions else "pass",
        "baseline_resolution": baseline["resolution"],
        "comparison_resolution": comparison["resolution"],
        "regressions": regressions,
        "new_semantic_warnings": new_warnings,
        "safe_profile": (
            baseline["resolution"] if regressions else comparison["resolution"]
        ),
    }
