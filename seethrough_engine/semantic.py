"""Canonical semantic portrait tag policy.

This module belongs to static portrait production. Exporters and repair use
the same back-to-front order, while rig-specific subdivisions are deliberately
absent from it. ``SEMANTIC_Z_ORDER`` is the producer reconstruction order for
the source portrait, never a downstream character's final draw-order policy.
"""

from __future__ import annotations

import cv2
import numpy as np

from .scale import scale_area

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

# `semantic_warnings` first looks for a ring around the iris that is bright
# and neutral in absolute terms (mean >= 200, chroma <= 8% of max -- tuned
# for flat-shaded, brightly-lit eyes). That floor is blind to sclera that is
# genuinely present but never reaches it: warm-toned or naturally dim
# sclera, common in photoreal portraits and on darker or warmly-lit skin,
# where the whole eye region sits well below 200 in absolute brightness even
# though the sclera is still visibly the lightest, least-colourful patch in
# its own neighbourhood. When the absolute test finds nothing, these three
# constants gate a second pass ranked against the ring's *own* local
# contrast instead of a fixed number, so detection adapts to whatever the
# local lighting and skin tone actually are rather than assuming a bright,
# neutral baseline. `eyewhite_derivation.derive_missing_eyewhite` and
# `local_fidelity._sclera_observation` import these too, so none of the
# three ever disagree about what counts as sclera evidence.
#
# Ranking against local contrast breaks down when the ring/ROI is not
# actually heterogeneous -- a flat, uniformly-lit patch of skin ranks its
# own brightest quartile as "the highlight" purely by construction, with
# nothing sclera-like about it. A real sclera band is always a small
# minority of its ring, so REL_MAX_CANDIDATE_RATIO throws the relative
# result out (falls back to "nothing decisive found") whenever it would
# claim an implausibly large share of the ring instead.
REL_BRIGHT_PERCENTILE = 75.0
REL_CHROMA_PERCENTILE = 40.0
REL_MIN_ABS_BRIGHTNESS = 90.0
REL_MAX_CANDIDATE_RATIO = 0.3


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
    min_iris_area = scale_area(20, irises.shape)
    rgb = original[..., :3].astype(np.float32)
    brightness = rgb.mean(axis=2)
    maximum = rgb.max(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    bright_neutral = (brightness >= 200.0) & (
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
        ring = (cv2.dilate(iris.astype(np.uint8), kernel).astype(bool) & ~iris) & support
        if not ring.any():
            continue
        required = max(scale_area(12, irises.shape), int(round(area * 0.08)))

        observed = ring & bright_neutral
        if int(observed.sum()) < required:
            # Nothing cleared the absolute floor -- try the same ring
            # ranked against its own local contrast (see the REL_* comment
            # above) before concluding there is no sclera here at all.
            ring_bright = brightness[ring]
            ring_chroma = chroma[ring]
            bright_cut = max(float(np.percentile(ring_bright, REL_BRIGHT_PERCENTILE)),
                              REL_MIN_ABS_BRIGHTNESS)
            chroma_cut = float(np.percentile(ring_chroma, REL_CHROMA_PERCENTILE))
            relative = ring & (brightness >= bright_cut) & (chroma <= chroma_cut)
            if float(relative.sum()) / float(ring.sum()) <= REL_MAX_CANDIDATE_RATIO:
                observed = observed | relative

        if int(observed.sum()) >= required:
            return ["missing_eyewhite"]

    return []
