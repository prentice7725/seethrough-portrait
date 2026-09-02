"""Deterministic recovery of a missing `eyewhite` layer from ground truth.

No inference here, and no new pixels are invented: every derived pixel is
copied verbatim from the original image, exactly where the picture already
shows it. This is the same category of operation as
`ownership.recover_missing_ownership` -- conservative recovery gated on
decisive local evidence, nothing folded in when the evidence is not
decisive -- aimed at one specific, frequently-lost semantic instead of the
general missing-pixel case.

It exists because `eyewhite` generation can fail per-character, at any head
resolution (see `docs/RESOLUTION_HEURISTICS_AUDIT.md`), and re-diffusing the
head crop at a higher resolution is comparatively expensive and not
guaranteed to help. Trying this first is strictly cheaper (no GPU call) and,
when it succeeds, exact rather than another generative guess -- see
`generation._rescue_head_semantic`, which calls this before spending any
resolution-escalation attempt.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .scale import scale_area
from .semantic import EYE_SURFACE_TAGS, IRIS_TAGS

__all__ = ["EyewhiteDeriveResult", "derive_missing_eyewhite"]

DERIVE_VERSION = "1.0"
ALPHA_THRESHOLD = 10

# A component smaller than this is noise, not an iris.
MIN_IRIS_AREA_AT_768 = 20
# The smallest derived patch worth keeping at all.
MIN_DERIVED_AREA_AT_768 = 12

# Ring geometry: dilate each iris by a radius proportional to its own size
# and look at what is just outside it -- the same idea
# `semantic.semantic_warnings` uses to *detect* a missing sclera, but tuned
# tighter here because this ring is used to *paint*, not just to flag: a
# smaller multiplier keeps the derived patch inside the sclera itself rather
# than bleeding into an eyebrow, an eyelid crease, or a skin highlight next
# to the eye. `semantic_warnings` uses 0.75 for a coarser "is anything here
# at all" check; 0.55 pulls the search ring in closer to the iris.
RING_RADIUS_IRIS_RATIO = 0.55

# The colour test itself: bright and low-chroma, same numbers
# `local_fidelity._sclera_observation` already validates against.
BRIGHT_MEAN_MIN = 190.0
BRIGHT_CHROMA_MAX_RATIO = 0.10

# How much of the search ring actually has to pass that colour test before a
# region is trusted as real sclera rather than a handful of stray bright
# pixels caught inside the dilation.
MIN_RING_COVERAGE_RATIO = 0.35


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


@dataclass(frozen=True)
class EyewhiteDeriveResult:
    layer: np.ndarray | None  # full-canvas RGBA, or None if nothing was derived
    derived_px: int
    iris_components_seen: int
    iris_components_used: int
    reason: str


def derive_missing_eyewhite(
    layer_dict: dict[str, np.ndarray],
    original_rgba: np.ndarray,
    *,
    alpha_threshold: int = ALPHA_THRESHOLD,
) -> EyewhiteDeriveResult:
    """Paint a conservative `eyewhite` candidate straight from the original,
    anchored on the existing `irides` layer.

    A region is only used when it is: a real connected patch of meaningful
    size, decisively bright and low-chroma over most of the ring around its
    iris, and sits where a head/face/ear semantic already has alpha -- so a
    stray highlight elsewhere on the face, or background bleeding in near
    the canvas edge, cannot be mistaken for a sclera. Returns `layer=None`
    (with `reason` explaining why) whenever the evidence is not decisive;
    the caller is expected to gate acceptance further on whether the result
    actually improves reconstruction, same as any other repair candidate.
    """
    original = np.asarray(original_rgba)
    if original.ndim != 3 or original.shape[-1] != 4:
        raise ValueError(f"original_rgba must be HxWx4, got {original.shape}")
    shape = original.shape[:2]

    iris = _alpha_union(layer_dict, IRIS_TAGS, alpha_threshold)
    if iris is None or not iris.any():
        return EyewhiteDeriveResult(None, 0, 0, 0, "no_irides_layer")

    support = _alpha_union(layer_dict, EYE_SURFACE_TAGS, alpha_threshold)
    if support is None or not support.any():
        return EyewhiteDeriveResult(None, 0, 0, 0, "no_head_or_face_support")

    rgb = original[..., :3].astype(np.float32)
    maximum = rgb.max(axis=2)
    chroma = maximum - rgb.min(axis=2)
    bright_neutral = (rgb.mean(axis=2) >= BRIGHT_MEAN_MIN) & (
        chroma <= BRIGHT_CHROMA_MAX_RATIO * np.maximum(maximum, 1.0))

    min_iris_area = scale_area(MIN_IRIS_AREA_AT_768, shape)
    min_derived_area = scale_area(MIN_DERIVED_AREA_AT_768, shape)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(iris.astype(np.uint8), 8)

    derived = np.zeros(shape, bool)
    seen = 0
    used = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_iris_area:
            continue
        seen += 1
        component = labels == index
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        radius = max(2, int(round(max(width, height) * RING_RADIUS_IRIS_RATIO)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
        ring &= support
        if not ring.any():
            continue

        candidate = ring & bright_neutral
        if int(candidate.sum()) < min_derived_area:
            continue
        if float(candidate.sum()) / float(ring.sum()) < MIN_RING_COVERAGE_RATIO:
            continue

        # Keep only pieces of meaningful size -- a bright patch elsewhere in
        # the ring (a highlight, a stray pixel near an eyebrow) is not
        # sclera just because the dilation happened to reach it.
        comp_count, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), 8)
        keep = np.zeros(shape, bool)
        for comp_index in range(1, comp_count):
            if int(comp_stats[comp_index, cv2.CC_STAT_AREA]) >= min_derived_area:
                keep |= comp_labels == comp_index
        if not keep.any():
            continue
        derived |= keep
        used += 1

    if not derived.any():
        return EyewhiteDeriveResult(None, 0, seen, 0, "no_ring_passed_the_evidence_gate")

    layer = np.zeros((*shape, 4), np.uint8)
    layer[..., :3][derived] = original[..., :3][derived]
    layer[..., 3][derived] = 255
    return EyewhiteDeriveResult(layer, int(derived.sum()), seen, used, "ok")
