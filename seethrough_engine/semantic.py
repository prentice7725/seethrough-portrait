"""Canonical semantic portrait tag policy.

This module belongs to static portrait production.  Exporters and repair use
the same back-to-front order, while rig-specific subdivisions are deliberately
absent from it.
"""

from __future__ import annotations

import cv2
import numpy as np

SEMANTIC_Z_ORDER: tuple[str, ...] = (
    "body_remainder",
    "wings",
    "hair", "back hair", "hairb",
    "tail",
    "objects",
    "footwear",
    "legwear",
    "bottomwear",
    "neck",
    "topwear",
    "neckwear",
    "handwear", "handwearl", "handwearr",
    "head",
    "ears", "earl", "earr",
    "earwear",
    "face",
    "eyebrow", "eyebrowl", "eyebrowr", "browl", "browr",
    "eyewhite", "eyewhitel", "eyewhiter",
    "irides", "iridesl", "iridesr",
    "eyelash", "eyelashl", "eyelashr",
    "eyes", "eyel", "eyer",
    "nose",
    "mouth",
    "eyewear",
    "front hair", "hairf",
    "headwear",
)

_RANK = {tag: index for index, tag in enumerate(SEMANTIC_Z_ORDER)}

EYEWHITE_TAGS = ("eyewhite", "eyewhitel", "eyewhiter")
IRIS_TAGS = ("irides", "iridesl", "iridesr")
EYE_SURFACE_TAGS = ("head", "face", "ears", "earl", "earr")


def semantic_rank(tag: str) -> int:
    """Back-to-front rank; unknown tags stay behind known facial layers."""
    return _RANK.get(tag, -1)


def ordered_tags(tags) -> list[str]:
    """Return tags in canonical back-to-front order."""
    return sorted(tags, key=semantic_rank)


def _alpha_union(layer_dict: dict[str, np.ndarray], tags: tuple[str, ...],
                 threshold: int) -> np.ndarray | None:
    union = None
    for tag in tags:
        layer = layer_dict.get(tag)
        if layer is None:
            continue
        mask = np.asarray(layer)[..., 3] > threshold
        union = mask if union is None else union | mask
    return union


def semantic_warnings(layer_dict: dict[str, np.ndarray], original_rgba: np.ndarray, *,
                      alpha_threshold: int = 10) -> list[str]:
    """Report observable missing semantic tags without making a rig verdict.

    A missing eyewhite tag is only reported when the original visibly contains
    a bright, low-chroma eye surface around an emitted iris and that surface is
    currently carried by a head/face semantic layer. Tag absence alone is not
    enough evidence: dark or fully closed eyes remain warning-free.
    """
    if _alpha_union(layer_dict, EYEWHITE_TAGS, alpha_threshold) is not None:
        if any(np.asarray(layer_dict[tag])[..., 3].max() > alpha_threshold
               for tag in EYEWHITE_TAGS if tag in layer_dict):
            return []

    irises = _alpha_union(layer_dict, IRIS_TAGS, alpha_threshold)
    support = _alpha_union(layer_dict, EYE_SURFACE_TAGS, alpha_threshold)
    if irises is None or support is None or not irises.any() or not support.any():
        return []

    original = np.asarray(original_rgba)
    if original.shape[:2] != irises.shape or original.ndim != 3 or original.shape[-1] != 4:
        raise ValueError("original_rgba and semantic layers must share an HxWx4 canvas")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        irises.astype(np.uint8), 8)
    scale = irises.size / float(768 * 768)
    min_iris_area = max(12, int(round(20 * scale)))
    rgb = original[..., :3].astype(np.float32)
    maximum = rgb.max(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    bright_neutral = (rgb.mean(axis=2) >= 200.0) & (
        chroma <= 0.08 * np.maximum(maximum, 1.0))

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_iris_area:
            continue
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        radius = max(3, int(round(max(width, height) * 0.75)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        iris = labels == index
        ring = cv2.dilate(iris.astype(np.uint8), kernel).astype(bool) & ~iris
        observed = ring & support & bright_neutral
        required = max(12, int(round(area * 0.08)))
        if int(observed.sum()) >= required:
            return ["missing_eyewhite"]

    return []
