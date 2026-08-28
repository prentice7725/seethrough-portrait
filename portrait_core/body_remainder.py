from __future__ import annotations

import numpy as np


def _alpha01(alpha: np.ndarray) -> np.ndarray:
    value = np.asarray(alpha, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("Alpha contains NaN or Inf")
    if value.size and float(value.max()) > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def composite_alpha(front_alpha: np.ndarray, back_alpha: np.ndarray) -> np.ndarray:
    front = _alpha01(front_alpha)
    back = _alpha01(back_alpha)
    if front.shape != back.shape:
        raise ValueError(f"Alpha shapes differ: {front.shape} vs {back.shape}")
    return np.clip(front + back * (1.0 - front), 0.0, 1.0)


def build_body_remainder(
    original_rgba: np.ndarray,
    subject_alpha: np.ndarray,
    generated_union_alpha: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    original = np.asarray(original_rgba)
    if original.ndim != 3 or original.shape[-1] != 4:
        raise ValueError(f"original_rgba must be HxWx4, got {original.shape}")
    subject = _alpha01(subject_alpha)
    union = _alpha01(generated_union_alpha)
    if original.shape[:2] != subject.shape or subject.shape != union.shape:
        raise ValueError(
            f"Shape mismatch: original={original.shape[:2]}, subject={subject.shape}, union={union.shape}"
        )
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    numerator = np.maximum(subject - union, 0.0)
    denominator = np.maximum(1.0 - union, float(epsilon))
    remainder_alpha = np.where(subject > union, numerator / denominator, 0.0)
    remainder_alpha = np.clip(remainder_alpha, 0.0, 1.0)

    rgba = np.empty_like(original, dtype=np.uint8)
    if original.dtype == np.uint8:
        rgba[..., :3] = original[..., :3]
    else:
        rgb = original[..., :3].astype(np.float32)
        if rgb.size and float(rgb.max()) <= 1.0:
            rgb *= 255.0
        rgba[..., :3] = np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    rgba[..., 3] = np.rint(remainder_alpha * 255.0).astype(np.uint8)
    return rgba
