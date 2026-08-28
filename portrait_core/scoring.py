from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .masks import compose_union_alpha
from .portrait_mode import default_portrait_profile, evaluate_portrait_layers
from .types import LayerSelectionResult, MaskEvidence, PortraitConfig, PortraitProfile


def masked_rgb_similarity(layer: np.ndarray, original_rgba: np.ndarray,
                          subject_alpha: np.ndarray) -> float:
    layer = np.asarray(layer)
    original = np.asarray(original_rgba)
    if layer.shape != original.shape or layer.ndim != 3 or layer.shape[-1] != 4:
        return 0.0
    alpha = layer[..., 3].astype(np.float32) / 255.0
    weights = np.minimum(alpha, subject_alpha.astype(np.float32))
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return 0.0
    delta = np.abs(layer[..., :3].astype(np.float32) - original[..., :3].astype(np.float32)) / 255.0
    mae = float((delta.mean(axis=2) * weights).sum() / weight_sum)
    return float(np.clip(1.0 - mae, 0.0, 1.0))


def estimate_shoulder_arm_region(subject_mask: np.ndarray, head_mask: np.ndarray | None
                                 ) -> tuple[np.ndarray | None, str]:
    subject = np.asarray(subject_mask, dtype=bool)
    if not np.any(subject) or head_mask is None:
        return None, "LOW"
    head = np.asarray(head_mask, dtype=bool)
    if head.shape != subject.shape or not np.any(head):
        return None, "LOW"
    sy, sx = np.where(subject)
    hy, hx = np.where(head)
    x0, x1 = int(sx.min()), int(sx.max()) + 1
    y1 = int(sy.max()) + 1
    head_bottom = int(hy.max()) + 1
    width = max(x1 - x0, 1)
    center_left = x0 + int(width * 0.35)
    center_right = x0 + int(width * 0.65)
    region = np.zeros_like(subject)
    region[max(int(hy.min()), head_bottom - max(1, int((hy.max() - hy.min() + 1) * 0.12))):y1, x0:center_left] = True
    region[max(int(hy.min()), head_bottom - max(1, int((hy.max() - hy.min() + 1) * 0.12))):y1, center_right:x1] = True
    region &= subject
    ratio = float(region.sum() / max(subject.sum(), 1))
    if ratio < 0.03 or ratio > 0.55:
        return None, "LOW"
    return region, "MEDIUM"


def _set_score(layers: Mapping[str, np.ndarray], original_rgba: np.ndarray,
               mask: MaskEvidence, profile: PortraitProfile, config: PortraitConfig,
               enable_head_detail: bool) -> float:
    scoring = config.section("scoring")
    union = compose_union_alpha(layers)
    subject = mask.alpha
    threshold = float(config.section("guard")["metric_binary_threshold"])
    subject_binary = subject > threshold
    denom = max(int(subject_binary.sum()), 1)
    coverage = float(np.logical_and(union > threshold, subject_binary).sum() / denom)
    spill = float(np.logical_and(union > threshold, ~subject_binary).sum() / max(int((union > threshold).sum()), 1))
    similarities = [masked_rgb_similarity(layer, original_rgba, subject) for layer in layers.values()]
    similarity = float(np.mean(similarities)) if similarities else 0.0

    shoulder_score = 0.0
    shoulder_weight = float(scoring["shoulder_arm_weight"])
    head_layer = layers.get("head")
    head_mask = None if head_layer is None else head_layer[..., 3] > 10
    shoulder_region, confidence = estimate_shoulder_arm_region(subject_binary, head_mask)
    if shoulder_region is None or confidence == "LOW":
        shoulder_weight = 0.0
    else:
        shoulder_score = float(np.logical_and(union > threshold, shoulder_region).sum() /
                               max(int(shoulder_region.sum()), 1))

    sim_weight = float(scoring["similarity_weight"])
    silhouette_weight = float(scoring["silhouette_weight"])
    active_sum = sim_weight + silhouette_weight + shoulder_weight
    if active_sum <= 0:
        raise ValueError("At least one positive scoring weight is required")
    sim_weight /= active_sum
    silhouette_weight /= active_sum
    shoulder_weight /= active_sum

    evaluation = evaluate_portrait_layers(
        layers, mask, profile, enable_head_detail, config
    )
    critical_penalty = float(scoring["critical_group_penalty"]) * len(evaluation.missing_critical_groups)
    return (
        similarity * sim_weight
        + coverage * silhouette_weight
        + shoulder_score * shoulder_weight
        - spill * float(scoring["spill_penalty_weight"])
        - critical_penalty
    )


def select_best_layer_set(
    runs: Sequence[Mapping[str, np.ndarray]],
    original_rgba: np.ndarray,
    subject_mask: MaskEvidence,
    profile: PortraitProfile | None = None,
    config: PortraitConfig | None = None,
    enable_head_detail: bool = True,
) -> LayerSelectionResult:
    if not runs:
        raise ValueError("At least one inference run is required")
    profile = profile or default_portrait_profile()
    config = config or PortraitConfig.load()
    selected = dict(runs[0])
    score = _set_score(
        selected, original_rgba, subject_mask, profile, config, enable_head_detail
    )
    trace: list[dict[str, object]] = []
    for run_index, run in enumerate(runs[1:], start=2):
        for tag, candidate in run.items():
            proposal = dict(selected)
            proposal[tag] = candidate
            proposal_score = _set_score(
                proposal, original_rgba, subject_mask, profile, config, enable_head_detail
            )
            accepted = proposal_score > score + 1e-9
            trace.append({
                "run": run_index,
                "tag": tag,
                "old_score": score,
                "new_score": proposal_score,
                "accepted": accepted,
            })
            if accepted:
                selected = proposal
                score = proposal_score
    return LayerSelectionResult(layers=selected, score=float(score), trace=tuple(trace))
