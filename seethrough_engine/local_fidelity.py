"""Feature-local static fidelity validation for canonical portrait layers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .scale import scale_area, scale_length

IRIS_TAGS = ("irides", "iridesl", "iridesr")
MOUTH_TAGS = ("mouth",)
NECK_TAGS = ("neck",)
GARMENT_CONTACT_TAGS = ("topwear", "neckwear")
ALPHA_THRESHOLD = 10
MIN_IRIS_AREA_AT_768 = 20
MIN_SCLERA_AREA_AT_768 = 12
LOCAL_BAD_RGB_SUM = 60
EYE_BAD_RATIO_REVIEW = 0.18
# Minor antialias/tone differences affect roughly one quarter of the observed
# sclera in the 768 baseline.  Losing most of the surface is qualitatively
# different; 1024 A002 loses 100% on both sides.  The majority threshold keeps
# the gate about feature preservation rather than pixel-perfect whitening.
SCLERA_LOST_RATIO_REVIEW = 0.50
MOUTH_BAD_RATIO_REVIEW = 0.18
# This is a contact neighbourhood, not a crop around the entire torso. It
# keeps a local neckline seam from being hidden by an otherwise good garment.
NECKLINE_CONTACT_BAND_PX = 6
NECKLINE_MIN_CONTACT_AREA_AT_768 = 24
NECKLINE_BAD_RATIO_REVIEW = 0.15


def _union(layer_dict: dict[str, np.ndarray], tags: tuple[str, ...]) -> np.ndarray | None:
    result = None
    for tag in tags:
        layer = layer_dict.get(tag)
        if layer is None:
            continue
        mask = np.asarray(layer)[..., 3] > ALPHA_THRESHOLD
        result = mask if result is None else result | mask
    return result


def _bbox_roi(mask: np.ndarray, *, x_factor: float, y_factor: float) -> tuple[int, int, int, int]:
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    pad_x = max(2, int(round(width * x_factor)))
    pad_y = max(2, int(round(height * y_factor)))
    return (
        max(0, x - pad_x), max(0, y - pad_y),
        min(mask.shape[1], x + width + pad_x),
        min(mask.shape[0], y + height + pad_y),
    )


def _roi_metrics(original: np.ndarray, composite: np.ndarray,
                 bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    lhs = original[y0:y1, x0:x1, :3].astype(np.int32)
    rhs = composite[y0:y1, x0:x1, :3].astype(np.int32)
    error = np.abs(lhs - rhs).sum(axis=2)
    pixels = max(int(error.size), 1)
    bad = int((error > LOCAL_BAD_RGB_SUM).sum())
    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "mae": round(float(error.mean()), 3),
        "bad_px": bad,
        "bad_ratio": round(bad / pixels, 6),
        "pixel_count": pixels,
    }


def _masked_metrics(original: np.ndarray, composite: np.ndarray,
                    mask: np.ndarray) -> dict[str, Any]:
    """Measure an irregular semantic contact band rather than its full bbox."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("masked local fidelity metrics need at least one pixel")
    error = np.abs(original[..., :3].astype(np.int32)
                   - composite[..., :3].astype(np.int32)).sum(axis=2)
    values = error[mask]
    bad = values > LOCAL_BAD_RGB_SUM
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bad_rows = np.bincount(ys[bad], minlength=mask.shape[0])
    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "mae": round(float(values.mean()), 3),
        "bad_px": int(bad.sum()),
        "bad_ratio": round(float(bad.mean()), 6),
        "pixel_count": int(values.size),
        # This makes thin horizontal errors auditable without treating a
        # geometric orientation as a different quality policy.
        "max_bad_row_px": int(bad_rows.max(initial=0)),
    }


def _neckline_contact_mask(layer_dict: dict[str, np.ndarray],
                           shape: tuple[int, ...]) -> np.ndarray | None:
    """Return only the local neck/garment handoff neighbourhood.

    A full neck or topwear bounding box is too broad: it lets a long, thin
    static seam disappear into a mostly faithful region. Contact is inferred
    from semantic alpha geometry and is resolution-normalized; no rig or pose
    information is involved.
    """
    neck = _union(layer_dict, NECK_TAGS)
    garment = _union(layer_dict, GARMENT_CONTACT_TAGS)
    if neck is None or garment is None or not neck.any() or not garment.any():
        return None
    band = max(1, scale_length(NECKLINE_CONTACT_BAND_PX, shape))
    near_neck = cv2.distanceTransform((~neck).astype(np.uint8), cv2.DIST_L2, 3) <= band
    near_garment = cv2.distanceTransform(
        (~garment).astype(np.uint8), cv2.DIST_L2, 3) <= band
    contact = (near_neck & near_garment) & (neck | garment)
    if int(contact.sum()) < scale_area(NECKLINE_MIN_CONTACT_AREA_AT_768, shape):
        return None
    return contact


def _sclera_observation(original: np.ndarray, composite: np.ndarray,
                        iris: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, Any]:
    x0, y0, x1, y1 = roi
    roi_mask = np.zeros(iris.shape, bool)
    roi_mask[y0:y1, x0:x1] = True
    rgb = original[..., :3].astype(np.float32)
    maximum = rgb.max(axis=2)
    chroma = maximum - rgb.min(axis=2)
    bright_neutral = (rgb.mean(axis=2) >= 190.0) & (
        chroma <= 0.10 * np.maximum(maximum, 1.0))
    observed = roi_mask & ~iris & bright_neutral
    minimum = scale_area(MIN_SCLERA_AREA_AT_768, original.shape)
    count = int(observed.sum())
    if count < minimum:
        return {
            "visible_in_original": False,
            "observed_px": count,
            "lost_px": 0,
            "lost_ratio": 0.0,
        }
    error = np.abs(original[..., :3].astype(np.int32)
                   - composite[..., :3].astype(np.int32)).sum(axis=2)
    comp_rgb = composite[..., :3].astype(np.float32)
    comp_max = comp_rgb.max(axis=2)
    comp_chroma = comp_max - comp_rgb.min(axis=2)
    reconstructed_neutral = (comp_rgb.mean(axis=2) >= 175.0) & (
        comp_chroma <= 0.14 * np.maximum(comp_max, 1.0))
    lost = observed & ((error > LOCAL_BAD_RGB_SUM) | ~reconstructed_neutral)
    lost_px = int(lost.sum())
    return {
        "visible_in_original": True,
        "observed_px": count,
        "lost_px": lost_px,
        "lost_ratio": round(lost_px / count, 6),
    }


def local_fidelity_report(original_rgba: np.ndarray, composite_rgba: np.ndarray,
                          layer_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    """Measure left/right eye and mouth reconstruction from semantic anchors.

    A missing eyewhite tag is not itself a failure.  Review is based on what is
    visibly present in the source eye ROI and absent from the canonical render.
    Closed or stylised dark eyes therefore do not produce a sclera warning.
    """
    original = np.asarray(original_rgba)
    composite = np.asarray(composite_rgba)
    if original.shape != composite.shape or original.ndim != 3 or original.shape[-1] != 4:
        raise ValueError("original and composite must share an HxWx4 canvas")

    eyes: list[dict[str, Any]] = []
    iris_union = _union(layer_dict, IRIS_TAGS)
    if iris_union is not None and iris_union.any():
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            iris_union.astype(np.uint8), 8)
        minimum = scale_area(MIN_IRIS_AREA_AT_768, original.shape)
        indices = [
            index for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum
        ]
        indices.sort(key=lambda index: float(centers[index][0]))
        for position, index in enumerate(indices[:2]):
            iris = labels == index
            roi = _bbox_roi(iris, x_factor=1.25, y_factor=0.75)
            metrics = _roi_metrics(original, composite, roi)
            sclera = _sclera_observation(original, composite, iris, roi)
            side = "left" if position == 0 else "right"
            review = (
                metrics["bad_ratio"] > EYE_BAD_RATIO_REVIEW
                or (sclera["visible_in_original"]
                    and sclera["lost_ratio"] > SCLERA_LOST_RATIO_REVIEW)
            )
            eyes.append({
                "feature": f"{side}_eye",
                **metrics,
                "sclera": sclera,
                "status": "review" if review else "pass",
            })

    mouth = None
    mouth_union = _union(layer_dict, MOUTH_TAGS)
    if mouth_union is not None and mouth_union.any():
        roi = _bbox_roi(mouth_union, x_factor=0.35, y_factor=0.75)
        metrics = _roi_metrics(original, composite, roi)
        mouth = {
            "feature": "mouth",
            **metrics,
            "status": "review" if metrics["bad_ratio"] > MOUTH_BAD_RATIO_REVIEW else "pass",
        }

    neckline = None
    neckline_mask = _neckline_contact_mask(layer_dict, original.shape)
    if neckline_mask is not None:
        metrics = _masked_metrics(original, composite, neckline_mask)
        neckline = {
            "feature": "neckline_contact",
            **metrics,
            "status": (
                "review" if metrics["bad_ratio"] > NECKLINE_BAD_RATIO_REVIEW
                else "pass"
            ),
        }

    warnings: list[str] = []
    if not eyes:
        status = "unavailable"
        warnings.append("eye_local_fidelity_unavailable")
    elif any(eye["status"] == "review" for eye in eyes):
        status = "review"
        if any(eye["sclera"]["visible_in_original"]
               and eye["sclera"]["lost_ratio"] > SCLERA_LOST_RATIO_REVIEW
               for eye in eyes):
            warnings.append("missing_visible_eyewhite")
        else:
            warnings.append("eye_local_fidelity_regression")
    else:
        status = "pass"
    if mouth is not None and mouth["status"] == "review":
        warnings.append("mouth_local_fidelity_regression")
        if status == "pass":
            status = "review"
    if neckline is not None and neckline["status"] == "review":
        warnings.append("neckline_local_fidelity_regression")
        if status in {"pass", "unavailable"}:
            status = "review"

    return {
        "version": "1.0",
        "status": status,
        "eyes": eyes,
        "mouth": mouth,
        "neckline": neckline,
        "warnings": warnings,
    }
