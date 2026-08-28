from __future__ import annotations

from typing import Any

from .types import GuardResult, MaskEvidence, PortraitConfig, PortraitEvaluation, dataclass_to_dict


def build_portrait_report(
    source: dict[str, Any],
    run: dict[str, Any],
    mask: MaskEvidence,
    guard: GuardResult,
    evaluation: PortraitEvaluation,
    config: PortraitConfig,
    selection_trace: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    verdict = guard.verdict
    reasons = list(guard.reasons)
    if evaluation.missing_critical_groups:
        if verdict == "PASS":
            verdict = "REWORK"
        elif verdict.startswith("SOFT_PASS"):
            verdict = "REWORK"
        reasons.append(
            "Missing critical semantic groups: " + ", ".join(evaluation.missing_critical_groups)
        )
    return {
        "schema_version": 1,
        "mode": "portrait",
        "source": {
            **source,
            "mask_source": mask.source,
            "mask_confidence": mask.confidence,
            "foreground_ratio": mask.foreground_ratio,
            "touches_border": dict(mask.touches_border),
        },
        "run": dict(run),
        "semantic": {
            "critical_groups": dict(evaluation.critical_groups),
            "missing_critical_groups": list(evaluation.missing_critical_groups),
            "handwear_detected": evaluation.handwear_detected,
            "valid_layers": list(evaluation.valid_layers),
            "semantic_success": evaluation.semantic_success,
        },
        "coverage": dataclass_to_dict(guard.metrics),
        "selection_trace": list(selection_trace),
        "verdict": verdict,
        "recovery_verdict": guard.verdict,
        "reasons": reasons,
        "warnings": list(mask.warnings),
        "config": config.to_dict(),
    }
