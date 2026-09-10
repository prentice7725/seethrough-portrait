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
    "handwear", "eyewhite", "irides", "eyelash", "eyebrow", "ears",
)

__all__ = [
    "LR_TAGS",
    "StratificationResult",
    "build_stratification",
    "label_lr_split",
    "part_lr_split",
    "process_cuts",
    "stratify_left_right",
    "tag_lr_split",
]


@dataclass(frozen=True)
class StratificationResult:
    """Derived outputs; ``layers`` contains no canonical layer replacements."""

    left_right: dict[str, dict[str, np.ndarray]]
    depth: dict[str, np.ndarray]
    report: dict[str, Any]


def label_lr_split(labels: np.ndarray, stats: np.ndarray, id1: int, id2: int):
    """Return masks ordered by canvas x (left, right) for node adapters."""
    label1 = (labels == id1).astype(np.uint8) * 255
    label2 = (labels == id2).astype(np.uint8) * 255
    stats1, stats2 = stats[id1], stats[id2]
    x1 = stats[id1][0] + stats[id1][2] / 2
    x2 = stats[id2][0] + stats[id2][2] / 2
    if x2 < x1:
        return label2, label1, stats2, stats1
    return label1, label2, stats1, stats2


def process_cuts(img: np.ndarray, depth: np.ndarray, src_xyxy, tgt_bbox, mask=None):
    """Crop one stratified part while preserving its depth map and alpha."""
    tx1, ty1, tx2, ty2 = (int(value) for value in tgt_bbox[:4])
    tx2 += tx1
    ty2 += ty1
    cropped = np.asarray(img)[ty1:ty2, tx1:tx2].copy()
    cropped_depth = np.asarray(depth)[ty1:ty2, tx1:tx2]
    depth_median = 1.0
    if mask is not None:
        local_mask = (np.asarray(mask)[ty1:ty2, tx1:tx2].copy() > 15).astype(np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
        local_mask = cv2.dilate(local_mask, element)
        cropped[..., -1] *= local_mask
        cropped_depth = 1 - (1 - cropped_depth) * local_mask
        if np.any(local_mask):
            depth_median = float(np.median(cropped_depth[local_mask > 0]))
    fxyxy = [tx1 + src_xyxy[0], ty1 + src_xyxy[1], tx2 + src_xyxy[0], ty2 + src_xyxy[1]]
    return cropped, cropped_depth, fxyxy, depth_median


def part_lr_split(tag: str, part_info: dict) -> dict[str, dict]:
    """Split a node-style part record into its two largest components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(part_info["mask"]).astype(np.uint8) * 255, connectivity=8
    )
    result: dict[str, dict] = {}
    if len(stats) <= 2:
        result[tag] = part_info
        return result
    stats = np.asarray(stats)
    stats_order = np.argsort(stats[..., -1])[::-1][1:]
    left_mask, right_mask, stats_left, stats_right = label_lr_split(
        labels, stats, int(stats_order[0]), int(stats_order[1])
    )
    img, depth, xyxy, dm = process_cuts(
        part_info["img"], part_info["depth"], part_info["xyxy"], stats_left,
        mask=left_mask,
    )
    result[f"{tag}-r"] = {
        "img": img, "xyxy": xyxy, "depth": depth,
        "depth_median": dm, "tag": f"{tag}-r",
    }
    img, depth, xyxy, dm = process_cuts(
        part_info["img"], part_info["depth"], part_info["xyxy"], stats_right,
        mask=right_mask,
    )
    result[f"{tag}-l"] = {
        "img": img, "xyxy": xyxy, "depth": depth,
        "depth_median": dm, "tag": f"{tag}-l",
    }
    return result


def tag_lr_split(tag: str, tag2pinfo: dict[str, dict]) -> None:
    """Mutate a node part mapping through the shared split implementation."""
    if tag in tag2pinfo:
        tag2pinfo.update(part_lr_split(tag, tag2pinfo.pop(tag)))


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
