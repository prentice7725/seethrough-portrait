from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .types import MaskEvidence, PortraitConfig, PortraitEvaluation, PortraitProfile


def default_portrait_profile() -> PortraitProfile:
    return PortraitProfile(
        name="upper_body_v1",
        critical_groups={
            "head_base": ("head",),
            "hair": ("front hair", "back hair", "hair"),
            "upper_body": ("topwear", "neck", "neckwear", "handwear"),
            "face_detail": ("face",),
        },
        optional_tags=(
            "headwear", "eyewear", "irides", "eyebrow", "eyewhite", "eyelash",
            "ears", "earwear", "nose", "mouth", "objects", "tail", "wings", "bottomwear",
        ),
        ignored_absent_tags=("legwear", "footwear"),
        enable_shoulder_arm_heuristic=True,
    )


def _present(layer: np.ndarray | None, threshold: float, min_ratio: float) -> bool:
    if layer is None:
        return False
    value = np.asarray(layer)
    if value.ndim != 3 or value.shape[-1] != 4:
        return False
    alpha = value[..., 3].astype(np.float32)
    if alpha.size and float(alpha.max()) > 1.0:
        alpha /= 255.0
    return float(np.mean(alpha > threshold)) >= min_ratio


def evaluate_portrait_layers(
    layer_dict: Mapping[str, np.ndarray],
    mask_evidence: MaskEvidence,
    profile: PortraitProfile | None = None,
    enable_head_detail: bool = True,
    config: PortraitConfig | None = None,
) -> PortraitEvaluation:
    profile = profile or default_portrait_profile()
    config = config or PortraitConfig.load()
    alpha_cfg = config.section("alpha")
    threshold = float(alpha_cfg["binary_threshold"])
    min_ratio = float(alpha_cfg["layer_presence_ratio_min"])
    valid_layers = tuple(sorted(
        tag for tag, layer in layer_dict.items() if _present(layer, threshold, min_ratio)
    ))
    valid_set = set(valid_layers)

    critical: dict[str, bool] = {}
    for group, tags in profile.critical_groups.items():
        if group == "face_detail" and not enable_head_detail:
            critical[group] = True
        else:
            critical[group] = any(tag in valid_set for tag in tags)
    missing = tuple(group for group, ok in critical.items() if not ok)
    return PortraitEvaluation(
        critical_groups=critical,
        missing_critical_groups=missing,
        handwear_detected="handwear" in valid_set,
        valid_layers=valid_layers,
        crop_flags=dict(mask_evidence.touches_border),
        semantic_success=not missing,
    )
