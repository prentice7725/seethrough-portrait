from __future__ import annotations

from collections.abc import Callable, Collection, Mapping

import numpy as np

from .types import MaskEvidence, PortraitConfig


DEFAULT_EXCLUDED_TAGS = frozenset({
    "body_remainder",
    "coverage_mask",
    "missing_mask",
    "spill_mask",
    "reconstruction_preview",
})


def _as_float01(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"Mask must be HxW or HxWx1, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Mask contains NaN or Inf")
    value = value.astype(np.float32, copy=False)
    if value.size and float(value.max()) > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _resize_nearest(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size: {target_hw}")
    if mask.shape == target_hw:
        return mask
    src_h, src_w = mask.shape
    if src_h == 0 or src_w == 0:
        raise ValueError("Cannot resize an empty mask")
    ys = np.minimum((np.arange(target_h) * src_h / target_h).astype(np.int64), src_h - 1)
    xs = np.minimum((np.arange(target_w) * src_w / target_w).astype(np.int64), src_w - 1)
    return mask[ys[:, None], xs[None, :]]


def normalize_foreground_mask(
    mask: np.ndarray,
    target_hw: tuple[int, int],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = _resize_nearest(_as_float01(mask), target_hw).astype(np.float32, copy=False)
    return alpha, alpha > float(threshold)


def _border_contacts(binary: np.ndarray) -> dict[str, bool]:
    if binary.size == 0:
        return {"top": False, "right": False, "bottom": False, "left": False}
    return {
        "top": bool(np.any(binary[0, :])),
        "right": bool(np.any(binary[:, -1])),
        "bottom": bool(np.any(binary[-1, :])),
        "left": bool(np.any(binary[:, 0])),
    }


def bbox_fill_ratio(binary: np.ndarray) -> float:
    """How much of its own bounding box the foreground fills.

    A real subject silhouette leaves plenty of its box empty -- around 0.64 for
    a framed upper-body portrait. A ratio at 1.0 means the "foreground" is an
    axis-aligned filled rectangle, which is the signature of an opaque image
    that was letterboxed or pillarboxed: the transparency is padding, not a
    matte, and the alpha channel says nothing about where the subject is.
    """
    if not binary.any():
        return 0.0
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    height = int(np.flatnonzero(rows)[-1] - np.flatnonzero(rows)[0] + 1)
    width = int(np.flatnonzero(cols)[-1] - np.flatnonzero(cols)[0] + 1)
    return float(binary.sum()) / float(height * width)


def _evidence(alpha: np.ndarray, source: str, confidence: str, threshold: float,
              warnings: tuple[str, ...] = ()) -> MaskEvidence:
    binary = alpha > threshold
    return MaskEvidence(
        alpha=alpha.astype(np.float32, copy=False),
        binary=binary,
        source=source,
        confidence=confidence,
        foreground_ratio=float(np.mean(binary)) if binary.size else 0.0,
        touches_border=_border_contacts(binary),
        warnings=warnings,
    )


def compose_union_alpha(
    layers: Mapping[str, np.ndarray],
    excluded_tags: Collection[str] = DEFAULT_EXCLUDED_TAGS,
) -> np.ndarray:
    union: np.ndarray | None = None
    excluded = set(excluded_tags)
    for tag, rgba in layers.items():
        if tag in excluded or rgba is None:
            continue
        rgba = np.asarray(rgba)
        if rgba.ndim != 3 or rgba.shape[-1] != 4:
            raise ValueError(f"Layer {tag!r} must be HxWx4 RGBA, got {rgba.shape}")
        if not np.all(np.isfinite(rgba)):
            raise ValueError(f"Layer {tag!r} contains NaN or Inf")
        alpha = rgba[..., 3].astype(np.float32)
        if alpha.size and float(alpha.max()) > 1.0:
            alpha /= 255.0
        alpha = np.clip(alpha, 0.0, 1.0)
        if union is None:
            union = np.zeros_like(alpha, dtype=np.float32)
        elif union.shape != alpha.shape:
            raise ValueError(f"Layer {tag!r} shape {alpha.shape} does not match {union.shape}")
        union = union + alpha * (1.0 - union)
    if union is None:
        raise ValueError("Cannot compose union from an empty layer mapping")
    return np.clip(union, 0.0, 1.0)


def resolve_subject_mask(
    original_rgba: np.ndarray,
    provided_mask: np.ndarray | None = None,
    segmentation_adapter: Callable[[np.ndarray], np.ndarray] | None = None,
    generated_layers: Mapping[str, np.ndarray] | None = None,
    config: PortraitConfig | None = None,
) -> MaskEvidence:
    config = config or PortraitConfig.load()
    alpha_cfg = config.section("alpha")
    original_rgba = np.asarray(original_rgba)
    if original_rgba.ndim != 3 or original_rgba.shape[-1] != 4:
        raise ValueError(f"original_rgba must be HxWx4, got {original_rgba.shape}")
    target_hw = original_rgba.shape[:2]
    threshold = float(alpha_cfg["binary_threshold"])

    source_alpha, source_binary = normalize_foreground_mask(
        original_rgba[..., 3], target_hw, threshold
    )
    transparent_ratio = float(np.mean(source_alpha < (1.0 - float(alpha_cfg["presence_threshold"]))))
    foreground_ratio = float(np.mean(source_binary))
    fill = bbox_fill_ratio(source_binary)
    # Measuring *how much* transparency exists is not enough: a pillarboxed
    # opaque image has plenty (the bars) while telling us nothing about the
    # subject. Rejecting a foreground that fills its own bounding box is what
    # separates padding from a matte -- without it the guard is handed the
    # whole rectangle as the subject and dutifully "recovers" the background
    # into body_remainder, which then reads as a REWORK verdict about layer
    # quality rather than about the input.
    padding_like = fill >= float(alpha_cfg["informative_bbox_fill_max"])
    informative = (
        transparent_ratio >= float(alpha_cfg["informative_transparency_min"])
        and float(alpha_cfg["subject_area_min"]) <= foreground_ratio <= float(alpha_cfg["subject_area_max"])
        and bool(np.any(source_binary))
        and not padding_like
    )
    if informative:
        return _evidence(source_alpha, "source_alpha", "HIGH", threshold)

    rejection = (
        f"Source alpha fills {fill:.3f} of its bounding box: it is padding around an "
        f"opaque image, not a subject matte."
        if padding_like else "Source alpha was not informative."
    )

    if provided_mask is not None:
        alpha, binary = normalize_foreground_mask(provided_mask, target_hw, threshold)
        ratio = float(np.mean(binary))
        if np.any(binary) and float(alpha_cfg["subject_area_min"]) <= ratio <= float(alpha_cfg["subject_area_max"]):
            return _evidence(alpha, "provided_mask", "HIGH", threshold,
                             (f"{rejection} Used the provided mask.",))

    if segmentation_adapter is not None:
        segmented = segmentation_adapter(original_rgba[..., :3])
        alpha, binary = normalize_foreground_mask(segmented, target_hw, threshold)
        ratio = float(np.mean(binary))
        if np.any(binary) and float(alpha_cfg["subject_area_min"]) <= ratio <= float(alpha_cfg["subject_area_max"]):
            return _evidence(alpha, "segmentation", "MEDIUM", threshold,
                             (f"{rejection} Used segmentation.",))

    if generated_layers:
        alpha = compose_union_alpha(generated_layers)
        if np.any(alpha > threshold):
            return _evidence(alpha, "fallback_union", "LOW", threshold, (
                rejection,
                "No independent subject mask was available; generated union cannot prove missing pixels.",
            ))

    raise ValueError(f"Unable to resolve a non-empty subject mask. {rejection}")
