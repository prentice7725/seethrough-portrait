from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .body_remainder import build_body_remainder, composite_alpha
from .masks import DEFAULT_EXCLUDED_TAGS, compose_union_alpha
from .types import CoverageMetrics, GuardResult, MaskEvidence, PortraitConfig


def _guard_layer(layer: np.ndarray, subject_alpha: np.ndarray) -> np.ndarray:
    value = np.asarray(layer)
    if value.ndim != 3 or value.shape[-1] != 4:
        raise ValueError(f"Layer must be HxWx4 RGBA, got {value.shape}")
    if value.shape[:2] != subject_alpha.shape:
        raise ValueError(f"Layer shape {value.shape[:2]} does not match mask {subject_alpha.shape}")
    guarded = value.astype(np.uint8, copy=True)
    layer_alpha = guarded[..., 3].astype(np.float32) / 255.0
    guarded[..., 3] = np.rint(np.minimum(layer_alpha, subject_alpha) * 255.0).astype(np.uint8)
    return guarded


def _reconstruct_rgba(original_rgba: np.ndarray, union_alpha: np.ndarray,
                      remainder: np.ndarray) -> np.ndarray:
    result = np.asarray(original_rgba).astype(np.uint8, copy=True)
    post = composite_alpha(union_alpha, remainder[..., 3].astype(np.float32) / 255.0)
    result[..., 3] = np.rint(post * 255.0).astype(np.uint8)
    return result


def apply_silhouette_guard(
    original_rgba: np.ndarray,
    layers: Mapping[str, np.ndarray],
    mask_evidence: MaskEvidence,
    config: PortraitConfig | None = None,
) -> GuardResult:
    config = config or PortraitConfig.load()
    guard_cfg = config.section("guard")
    verdict_cfg = config.section("verdict")
    subject = mask_evidence.alpha.astype(np.float32, copy=False)
    if not np.any(mask_evidence.binary):
        raise ValueError("Subject mask is empty")
    if np.asarray(original_rgba).shape[:2] != subject.shape:
        raise ValueError("Original image and subject mask are not aligned")

    union_pre = compose_union_alpha(layers)
    if union_pre.shape != subject.shape:
        raise ValueError("Generated layers and subject mask are not aligned")

    guard_enabled = bool(guard_cfg.get("enabled", True))
    if guard_enabled and bool(guard_cfg["clip_layers_to_subject"]):
        guarded_layers = {
            tag: (_guard_layer(layer, subject) if tag not in DEFAULT_EXCLUDED_TAGS else np.asarray(layer).copy())
            for tag, layer in layers.items()
        }
    else:
        guarded_layers = {tag: np.asarray(layer).copy() for tag, layer in layers.items()}
    union_guarded = compose_union_alpha(guarded_layers)
    missing_float = np.maximum(subject - union_guarded, 0.0)
    spill_float = np.maximum(union_pre - subject, 0.0)
    if guard_enabled:
        remainder = build_body_remainder(
            original_rgba,
            subject,
            union_guarded,
            float(guard_cfg["reconstruction_epsilon"]),
        )
    else:
        remainder = np.zeros_like(original_rgba, dtype=np.uint8)
    remainder_alpha = remainder[..., 3].astype(np.float32) / 255.0
    post = composite_alpha(union_guarded, remainder_alpha)
    threshold = float(guard_cfg["metric_binary_threshold"])
    subject_binary = subject > threshold
    generated_binary = union_pre > threshold
    guarded_binary = union_guarded > threshold
    missing_binary = missing_float > threshold
    spill_binary = spill_float > threshold
    remainder_binary = remainder_alpha > threshold
    post_binary = post > threshold

    subject_area = int(subject_binary.sum())
    generated_area = int(generated_binary.sum())
    intersection = int(np.logical_and(subject_binary, guarded_binary).sum())
    missing_area = int(missing_binary.sum())
    spill_area = int(spill_binary.sum())
    post_spill = int(np.logical_and(post_binary, ~subject_binary).sum())
    valid_layer_count = sum(
        1 for tag, layer in guarded_layers.items()
        if tag not in DEFAULT_EXCLUDED_TAGS and np.any(np.asarray(layer)[..., 3] > 10)
    )
    metrics = CoverageMetrics(
        subject_area_px=subject_area,
        generated_area_px=generated_area,
        intersection_area_px=intersection,
        missing_area_px=missing_area,
        spill_area_px=spill_area,
        silhouette_coverage=float(intersection / subject_area),
        missing_ratio=float(missing_area / subject_area),
        spill_ratio=float(spill_area / max(generated_area, 1)),
        recovered_ratio=float(remainder_binary.sum() / subject_area),
        post_recovery_coverage=float(np.logical_and(post_binary, subject_binary).sum() / subject_area),
        post_alpha_mae=float(np.mean(np.abs(post - subject))),
        post_spill_ratio=float(post_spill / max(int(post_binary.sum()), 1)),
        handwear_detected=bool(
            "handwear" in guarded_layers and np.any(np.asarray(guarded_layers["handwear"])[..., 3] > 10)
        ),
        valid_layer_count=valid_layer_count,
    )

    reasons: list[str] = []
    if metrics.post_recovery_coverage < float(verdict_cfg["soft_pass_post_coverage_min"]):
        verdict = "FAIL"
        reasons.append("Post-recovery silhouette coverage is below threshold.")
    elif mask_evidence.confidence == "LOW":
        verdict = "SOFT_PASS_LOW_CONFIDENCE"
        reasons.append("No independent subject mask was available.")
    elif (
        metrics.silhouette_coverage >= float(verdict_cfg["pass_pre_coverage_min"])
        and metrics.recovered_ratio <= float(verdict_cfg["pass_remainder_max"])
        and metrics.post_spill_ratio <= float(verdict_cfg["post_spill_max"])
    ):
        verdict = "PASS"
    elif metrics.recovered_ratio <= float(verdict_cfg["soft_pass_remainder_max"]):
        verdict = "SOFT_PASS"
        reasons.append("Silhouette was recovered, but semantic layers rely on BODY_REMAINDER.")
    else:
        verdict = "REWORK"
        reasons.append("Recovery succeeded, but BODY_REMAINDER dependency is too high.")

    return GuardResult(
        guarded_layers=guarded_layers,
        body_remainder=remainder,
        subject_mask=subject,
        generated_union_pre_guard=union_pre,
        generated_union_post_guard=union_guarded,
        missing_mask=missing_float,
        spill_mask=spill_float,
        reconstruction_rgba=_reconstruct_rgba(original_rgba, union_guarded, remainder),
        metrics=metrics,
        verdict=verdict,
        reasons=reasons,
    )
