"""Optional producer-side derivations for Portrait Bundle outputs.

The canonical semantic layer set is never mutated here.  Stratification is a
separate, explicitly requested derived stage: depth maps are supplied by the
caller (for example, the Marigold adapter), while left/right splitting uses
only the layer alpha geometry and connected components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


LR_TAGS: tuple[str, ...] = (
    # These optional derivatives use geometric canvas sides in the public
    # Bundle contract; left/right does not imply anatomical side.
    "handwear", "legwear", "footwear",
    "eyewhite", "irides", "eyelash", "eyebrow", "ears",
)

__all__ = [
    "LR_TAGS",
    "StratificationResult",
    "build_stratification",
    "stratify_left_right",
]


@dataclass(frozen=True)
class StratificationResult:
    """Derived outputs; ``layers`` contains no canonical layer replacements."""

    left_right: dict[str, dict[str, np.ndarray]]
    depth: dict[str, np.ndarray]
    report: dict[str, Any]


def _split_two_sides(image: np.ndarray, *, alpha_threshold: int) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(image)
    mask = arr[..., 3] > alpha_threshold
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 2:
        return None
    components = sorted(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:2]
    if len(components) < 2:
        return None
    # Geometric left/right is explicit in the Bundle; do not inherit the
    # historical rig naming convention where character-right used the smaller
    # canvas x coordinate.
    components.sort(key=lambda index: float(stats[index, cv2.CC_STAT_LEFT]))
    result: list[np.ndarray] = []
    for index in components:
        side = np.zeros_like(arr)
        side[labels == index] = arr[labels == index]
        result.append(side)
    return result[0], result[1]


def stratify_left_right(
    layer_dict: Mapping[str, np.ndarray], *,
    tags: tuple[str, ...] = LR_TAGS,
    alpha_threshold: int = 10,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    """Derive geometric left/right parts without changing canonical layers."""
    derived: dict[str, dict[str, np.ndarray]] = {}
    report: dict[str, Any] = {"status": "not_computed", "layers": {}}
    for tag in tags:
        image = layer_dict.get(tag)
        if image is None:
            continue
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 4:
            continue
        split = _split_two_sides(arr, alpha_threshold=alpha_threshold)
        if split is None:
            continue
        derived[tag] = {"left": split[0], "right": split[1]}
        report["status"] = "computed"
        report["layers"][tag] = {
            "left": int((split[0][..., 3] > alpha_threshold).sum()),
            "right": int((split[1][..., 3] > alpha_threshold).sum()),
        }
    return derived, report


def build_stratification(
    layer_dict: Mapping[str, np.ndarray], *,
    depth_maps: Mapping[str, np.ndarray] | None = None,
    left_right: bool = False,
    tags: tuple[str, ...] = LR_TAGS,
    alpha_threshold: int = 10,
) -> StratificationResult:
    """Run only the explicitly requested producer-side derived stages."""
    if left_right:
        lr_layers, lr_report = stratify_left_right(
            layer_dict, tags=tags, alpha_threshold=alpha_threshold
        )
    else:
        lr_layers, lr_report = {}, {"status": "not_computed", "layers": {}}

    depth: dict[str, np.ndarray] = {}
    depth_report: dict[str, Any] = {"status": "not_computed", "layers": {}}
    if depth_maps:
        shape = next(iter(layer_dict.values())).shape[:2] if layer_dict else None
        for tag, value in depth_maps.items():
            arr = np.asarray(value)
            if arr.ndim != 2 or (shape is not None and arr.shape != shape):
                raise ValueError(f"depth map for {tag!r} must match the canvas")
            depth[str(tag)] = np.clip(arr, 0.0, 1.0).astype(np.float32)
        depth_report = {
            "status": "computed" if depth else "not_computed",
            "layers": sorted(depth),
        }
    return StratificationResult(
        left_right=lr_layers,
        depth=depth,
        report={"left_right": lr_report, "depth": depth_report},
    )
