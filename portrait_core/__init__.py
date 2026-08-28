"""Pure portrait-mode analysis and recovery primitives.

This package intentionally has no ComfyUI, torch, diffusers, or OpenCV import so
its contracts can be tested without a GPU environment.
"""

from .body_remainder import build_body_remainder, composite_alpha
from .masks import compose_union_alpha, resolve_subject_mask
from .portrait_mode import default_portrait_profile, evaluate_portrait_layers
from .scoring import select_best_layer_set
from .silhouette_guard import apply_silhouette_guard
from .types import (
    CoverageMetrics,
    GuardResult,
    MaskEvidence,
    PortraitConfig,
    PortraitEvaluation,
    PortraitProfile,
)

__all__ = [
    "CoverageMetrics",
    "GuardResult",
    "MaskEvidence",
    "PortraitConfig",
    "PortraitEvaluation",
    "PortraitProfile",
    "apply_silhouette_guard",
    "build_body_remainder",
    "compose_alpha",
    "composite_alpha",
    "compose_union_alpha",
    "default_portrait_profile",
    "evaluate_portrait_layers",
    "resolve_subject_mask",
    "select_best_layer_set",
]


# Backward-friendly descriptive alias used by callers and docs.
compose_alpha = composite_alpha
