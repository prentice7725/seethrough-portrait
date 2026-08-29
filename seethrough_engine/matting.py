"""Turn a flat-colour background into alpha.

The image models in this workflow do not emit RGBA, so a portrait arrives as an
opaque picture on a solid background. `masks.resolve_subject_mask` rejects that
outright -- an opaque rectangle is padding, not a matte -- so the background has
to become alpha before anything downstream can run.

Two things make this more than a colour threshold:

* **Anti-aliased edges.** Every boundary pixel is a blend of foreground and
  background. Thresholding keeps the background's contribution, so each hair
  strand ends up ringed with background colour. Portrait Mode's hair layers
  already account for a quarter of the composite error; a halo on top of that
  is not acceptable. Edge pixels get a fractional alpha and are
  un-premultiplied, which removes the tint instead of hiding it.
* **Same-coloured foreground.** A white collar against a white background is
  not background. Keying selects only the regions of background colour that are
  *connected to the border*, so an enclosed area of the same colour stays.

numpy and cv2 only, no torch, so it is testable without the inference stack.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

__all__ = [
    "DEFAULT_TOLERANCE",
    "detect_flat_background",
    "key_flat_background",
]

# Per-channel distance from the sampled background colour that still counts as
# background. Generous enough for the dithering and compression noise a
# generated PNG carries, tight enough not to eat a pale cheek.
DEFAULT_TOLERANCE = 18.0

# A background this uneven is a gradient, a texture, or a drop shadow -- not a
# flat colour, and keying it will leave a fringe wherever it varies.
DEFAULT_MAX_BORDER_STD = 6.0

# Width of the ring sampled to identify the background colour, and of the band
# inside the silhouette whose alpha is estimated rather than assumed.
BORDER_RING_PX = 2
EDGE_BAND_PX = 3

# An estimated alpha this close to 1 is opaque: snap it, so an essentially
# solid pixel keeps its exact colour instead of being un-premultiplied through
# a near-unity divisor and drifting by a bit or two.
OPAQUE_SNAP = 0.99

# Colour bin used to find the dominant border colour, and the least of the
# border that has to carry it before the guess stops being trustworthy.
_BG_BIN = 8
_BG_CANDIDATES = 6
MIN_BORDER_SHARE = 0.25


def _content_mask(image: np.ndarray) -> np.ndarray:
    """Where the picture actually is.

    Not always the whole canvas: a pillarboxed upload has transparent bars and
    an opaque picture between them, and sampling the canvas border there would
    read the bars rather than the background.

    Where such a surround exists, its innermost pixel is dropped: the boundary
    between padding and picture is itself an anti-aliased blend of the two and
    belongs to neither. Left in, that one-pixel seam is a tight colour cluster
    that both the detector and the flood fill have to work around.
    """
    arr = np.asarray(image)
    if arr.shape[-1] == 4 and np.any(arr[..., 3] < 255):
        mask = arr[..., 3] > 0
        if mask.any():
            trimmed = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            return trimmed if trimmed.any() else mask
    return np.ones(arr.shape[:2], dtype=bool)


def _border_ring(mask: np.ndarray, width: int) -> np.ndarray:
    """The outermost `width` pixels of `mask`, where the background must be if
    the subject is framed at all.

    `borderValue=0` is load-bearing: cv2's default for erode treats everything
    outside the array as maximal, so eroding a full-canvas mask returns it
    unchanged and the ring comes back empty.
    """
    kernel = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return mask & ~eroded


def detect_flat_background(image: np.ndarray, *, border: int = BORDER_RING_PX,
                           max_std: float = DEFAULT_MAX_BORDER_STD,
                           tolerance: float = DEFAULT_TOLERANCE,
                           min_share: float = MIN_BORDER_SHARE) -> dict[str, Any]:
    """Sample the background colour from the border of the picture.

    Returns `{"color", "std", "border_share", "flat", "reason"}`. `flat` is
    False when the colour is too uneven to key cleanly, or when too little of
    the border carries it -- reported rather than raised, so a caller can show
    the numbers and let a person decide.
    """
    arr = np.asarray(image)
    rgb = arr[..., :3].astype(np.float32)
    ring = _border_ring(_content_mask(arr), border)
    if not ring.any():
        return {"color": None, "std": 0.0, "border_share": 0.0, "flat": False,
                "reason": "no border to sample"}

    samples = rgb[ring]
    # The dominant colour, not the median. A bust portrait always runs off the
    # bottom edge -- every run so far has `touches_border.bottom` -- and often
    # off the top, so the ring reliably contains hair and clothing as well as
    # background. A median lands between them and names a colour that is
    # nowhere in the picture; the largest cluster is still the background.
    #
    # Bin counts alone are not enough to pick that cluster: a slightly noisy
    # background spreads across neighbouring bins while a tight artifact -- the
    # anti-aliased seam between a pillarbox bar and the picture -- sits in one
    # and outvotes it. So the top bins are only *candidates*, and the winner is
    # the one whose tolerance neighbourhood actually holds the most border.
    quantized = (samples // _BG_BIN).astype(np.int64)
    key = (quantized[:, 0] << 20) | (quantized[:, 1] << 10) | quantized[:, 2]
    values, counts = np.unique(key, return_counts=True)
    color, inliers = None, None
    for value in values[np.argsort(counts)[::-1][:_BG_CANDIDATES]]:
        candidate = np.median(samples[key == value], axis=0)
        hits = np.abs(samples - candidate).max(axis=1) <= tolerance
        if inliers is None or hits.sum() > inliers.sum():
            color, inliers = candidate, hits
    share = float(inliers.mean())
    std = float(np.sqrt(np.mean(np.sum((samples[inliers] - color) ** 2, axis=1))))

    if std > max_std:
        reason = f"the background colour itself varies by {std:.1f}, above {max_std}"
    elif share < min_share:
        reason = (f"only {share:.0%} of the border is that colour: the subject fills "
                  f"the frame, or the background is not one colour")
    else:
        reason = ""
    return {
        "color": [round(float(c), 1) for c in color],
        "std": round(std, 2),
        "border_share": round(share, 4),
        "flat": not reason,
        "reason": reason,
    }


def key_flat_background(image: np.ndarray, *, color: list[float] | None = None,
                        tolerance: float = DEFAULT_TOLERANCE,
                        border: int = BORDER_RING_PX,
                        edge_band: int = EDGE_BAND_PX,
                        max_std: float = DEFAULT_MAX_BORDER_STD,
                        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace a flat background with alpha. Returns `(rgba, info)`.

    `info` carries the sampled colour, the border spread, the resulting
    foreground ratio, and any warnings. It never raises on a poor result: the
    caller gets the numbers and decides, because "this looks like a bad key" is
    a judgement about the picture, not about the arithmetic.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"image must be HxWx3 or HxWx4, got {arr.shape}")
    rgb = arr[..., :3].astype(np.float32)
    height, width = rgb.shape[:2]

    detected = detect_flat_background(arr, border=border, max_std=max_std,
                                      tolerance=tolerance)
    if color is None and detected["color"] is None:
        raise ValueError(f"cannot sample a background colour: {detected['reason']}")
    bg = np.asarray(color if color is not None else detected["color"], dtype=np.float32)
    warnings: list[str] = []
    if color is None and not detected["flat"]:
        warnings.append(detected["reason"])

    content = _content_mask(arr)

    # Background-coloured regions, but only the ones reachable from the border:
    # a white collar against a white background is foreground, and a global
    # colour match would punch a hole through it.
    close = (np.abs(rgb - bg).max(axis=2) <= tolerance) & content
    count, labels = cv2.connectedComponents(close.astype(np.uint8), connectivity=8)
    ring = _border_ring(content, border)
    outer_labels = set(np.unique(labels[ring & close]).tolist()) - {0}
    outside = np.isin(labels, list(outer_labels)) if outer_labels else np.zeros_like(close)
    enclosed = int((close & ~outside).sum())

    inside = content & ~outside

    # Solid interior: far enough from the boundary that the pixel is pure
    # foreground, so its colour can stand in for the unknown F at the edge.
    # Default border handling here, unlike `_border_ring`: a subject running off
    # the canvas edge is cut off, not anti-aliased, so those pixels are solid.
    kernel = np.ones((2 * edge_band + 1, 2 * edge_band + 1), np.uint8)
    solid = cv2.erode(inside.astype(np.uint8), kernel).astype(bool)

    # Local average of solid colours, which is the nearest-foreground estimate
    # the alpha formula needs. A box blur over only the known pixels reaches
    # into the edge band without a per-pixel nearest-neighbour search.
    k = 2 * edge_band + 3
    weight = solid.astype(np.float32)
    num = cv2.blur(rgb * weight[..., None], (k, k))
    den = cv2.blur(weight, (k, k))[..., None]
    fore = np.where(den > 1e-4, num / np.maximum(den, 1e-6), rgb)

    # C = a*F + (1-a)*Bg  =>  a = (C-Bg).(F-Bg) / |F-Bg|^2
    delta_f = fore - bg
    denom = np.sum(delta_f * delta_f, axis=2)
    numer = np.sum((rgb - bg) * delta_f, axis=2)
    alpha = np.where(denom > 1e-3, numer / np.maximum(denom, 1e-6), 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Only the boundary band is uncertain. The interior is opaque by
    # construction, and letting the estimate speak there would punch holes
    # wherever the artwork happens to use the background colour.
    alpha = np.where(solid, 1.0, alpha)
    alpha = np.where(alpha >= OPAQUE_SNAP, 1.0, alpha)
    alpha = np.where(outside, 0.0, alpha)
    alpha = np.where(~content, 0.0, alpha)

    # Un-premultiply: strip the background's contribution out of the blend
    # rather than leaving it as a rim of background colour on every edge.
    safe = np.maximum(alpha, 1e-3)[..., None]
    unmixed = (rgb - (1.0 - alpha[..., None]) * bg) / safe
    keep = (alpha >= 1.0)[..., None]
    out_rgb = np.where(keep, rgb, np.clip(unmixed, 0.0, 255.0))

    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., :3] = np.rint(np.clip(out_rgb, 0, 255)).astype(np.uint8)
    rgba[..., 3] = np.rint(alpha * 255.0).astype(np.uint8)

    binary = rgba[..., 3] > 16
    ratio = float(binary.mean())
    if ratio < 0.02:
        warnings.append(f"only {ratio:.1%} of the canvas survived: the background "
                        "colour may match the subject")
    elif ratio > 0.95:
        warnings.append(f"{ratio:.1%} of the canvas survived: the background was "
                        "probably not keyed at all")
    if enclosed:
        warnings.append(f"{enclosed} background-coloured pixels are enclosed by the "
                        "subject and were kept opaque")

    info = {
        "color": [round(float(c), 1) for c in bg],
        "border_std": detected["std"],
        "border_share": detected["border_share"],
        "flat": bool(detected["flat"]),
        "foreground_ratio": round(ratio, 4),
        "soft_edge_px": int(((rgba[..., 3] > 0) & (rgba[..., 3] < 255)).sum()),
        "enclosed_px": enclosed,
        "components": int(count - 1),
        "warnings": tuple(warnings),
    }
    return rgba, info
