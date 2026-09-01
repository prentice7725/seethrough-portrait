"""Static full-canvas RGBA operations for canonical portrait layers."""

from __future__ import annotations

import cv2
import numpy as np

from .semantic import SEMANTIC_Z_ORDER, semantic_rank


def crop_to_alpha(img: np.ndarray, alpha_threshold: int = 10) -> tuple[np.ndarray, list[int]] | None:
    """Return the visible crop and its full-canvas ``xyxy`` box."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"img must be HxWx4, got {arr.shape}")
    nz = cv2.findNonZero((arr[..., 3] > alpha_threshold).astype(np.uint8))
    if nz is None:
        return None
    x, y, width, height = cv2.boundingRect(nz)
    return arr[y:y + height, x:x + width], [int(x), int(y), int(x + width), int(y + height)]


def composite_layers(layer_dict: dict[str, np.ndarray], frame_size: tuple[int, int], *,
                     order: tuple[str, ...] = SEMANTIC_Z_ORDER,
                     alpha_threshold: int = 10) -> np.ndarray:
    """Alpha-blend full-canvas straight-alpha layers back to front."""
    canvas_h, canvas_w = int(frame_size[0]), int(frame_size[1])
    rgb = np.zeros((canvas_h, canvas_w, 3), np.float32)
    acc = np.zeros((canvas_h, canvas_w, 1), np.float32)
    rank = {tag: index for index, tag in enumerate(order)}

    for tag in sorted(layer_dict, key=lambda item: rank.get(item, -1)):
        img = layer_dict.get(tag)
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.shape != (canvas_h, canvas_w, 4):
            raise ValueError(
                f"layer {tag!r} must match canvas {(canvas_h, canvas_w, 4)}, got {arr.shape}"
            )
        if not np.any(arr[..., 3] > alpha_threshold):
            continue
        src_a = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = arr[..., :3].astype(np.float32) * src_a + rgb * acc * (1.0 - src_a)
        acc = src_a + acc * (1.0 - src_a)
        rgb = rgb / np.maximum(acc, 1e-6)

    out = np.zeros((canvas_h, canvas_w, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    out[..., 3] = np.rint(np.clip(acc[..., 0] * 255.0, 0, 255)).astype(np.uint8)
    return out


def composite_fidelity(original_rgba: np.ndarray, composite: np.ndarray,
                       subject_mask: np.ndarray, *, bad_threshold: int = 30) -> dict[str, float]:
    """Measure RGB reconstruction error inside the subject silhouette."""
    original = np.asarray(original_rgba)[..., :3].astype(np.int32)
    made = np.asarray(composite)[..., :3].astype(np.int32)
    if original.shape != made.shape:
        raise ValueError(f"original and composite shapes differ: {original.shape} != {made.shape}")
    mask = np.asarray(subject_mask)
    if mask.dtype != bool:
        mask = mask > (0.5 if mask.size and mask.max() <= 1.0 else 127)
    total = int(mask.sum())
    if total == 0:
        return {"mae": 0.0, "bad_ratio": 0.0, "bad_px": 0, "subject_px": 0}
    diff = np.abs(original - made).sum(axis=2)[mask]
    bad = int((diff > bad_threshold).sum())
    return {
        "mae": round(float(diff.mean()), 3),
        "bad_ratio": round(bad / total, 5),
        "bad_px": bad,
        "subject_px": total,
    }
