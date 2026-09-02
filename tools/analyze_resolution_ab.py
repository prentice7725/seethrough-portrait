"""Replay static post-processing and compare two Portrait Bundle resolutions.

Usage:
  python tools/analyze_resolution_ab.py BASELINE.portrait COMPARISON.portrait
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portrait_core import PortraitConfig, apply_silhouette_guard, resolve_subject_mask
from seethrough_engine.image import composite_fidelity, composite_layers
from seethrough_engine.local_fidelity import local_fidelity_report
from seethrough_engine.ownership import recover_missing_ownership
from seethrough_engine.repair import repair_portrait_layers
from seethrough_engine.resolution_regression import (
    bundle_resolution_snapshot,
    compare_resolution_snapshots,
)


def _rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def replay(bundle: Path, *, write_previews: bool = False) -> dict:
    original = _rgba(bundle / "original.png")
    raw = {path.stem: _rgba(path) for path in (bundle / "raw_layers").glob("*.png")}
    config = PortraitConfig.load()
    mask = resolve_subject_mask(original, generated_layers=raw, config=config)
    before_guard = apply_silhouette_guard(original, raw, mask, config)
    cv2.setRNGSeed(0)
    repaired = repair_portrait_layers(before_guard.guarded_layers, original)
    ownership = recover_missing_ownership(
        repaired.layers, original, before_guard.subject_mask)
    after_guard = apply_silhouette_guard(original, ownership.layers, mask, config)
    canonical = dict(after_guard.guarded_layers)
    if np.any(after_guard.body_remainder[..., 3] > 10):
        canonical["body_remainder"] = after_guard.body_remainder
    composite = composite_layers(canonical, original.shape[:2])
    if write_previews:
        diagnostics = bundle / "diagnostics"
        Image.fromarray(after_guard.guarded_layers["topwear"]).save(
            diagnostics / "replay_v13_topwear.png")
        Image.fromarray(after_guard.body_remainder).save(
            diagnostics / "replay_v13_body_remainder.png")
        Image.fromarray(composite).save(
            diagnostics / "replay_v13_layer_composite.png")
    cleanup = repaired.report["clean_garment_orphans"].get("topwear", {})
    ambiguous = [
        row for row in cleanup.get("components", [])
        if row.get("status") == "ambiguous"
    ]
    return {
        "ownership": ownership.report,
        "remainder_ratio_before": before_guard.metrics.recovered_ratio,
        "remainder_ratio_after": after_guard.metrics.recovered_ratio,
        "fidelity": composite_fidelity(
            original, composite, after_guard.subject_mask),
        "local_fidelity": local_fidelity_report(
            original, composite, after_guard.guarded_layers),
        "topwear_cleanup": {
            "removed_px": int(cleanup.get("removed_px", 0)),
            "ambiguous_components": len(ambiguous),
            "ambiguous_area_px": int(sum(row.get("area_px", 0) for row in ambiguous)),
            "components": cleanup.get("components", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args()
    baseline = bundle_resolution_snapshot(args.baseline)
    comparison = bundle_resolution_snapshot(args.comparison)
    result = {
        "captured": {"baseline": baseline, "comparison": comparison},
        "comparison": compare_resolution_snapshots(baseline, comparison),
    }
    if args.replay:
        result["replayed"] = {
            "baseline": replay(args.baseline, write_previews=args.write_previews),
            "comparison": replay(args.comparison, write_previews=args.write_previews),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
