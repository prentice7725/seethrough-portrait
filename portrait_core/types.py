from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaskEvidence:
    alpha: np.ndarray
    binary: np.ndarray
    source: str
    confidence: str
    foreground_ratio: float
    touches_border: dict[str, bool]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageMetrics:
    subject_area_px: int
    generated_area_px: int
    intersection_area_px: int
    missing_area_px: int
    spill_area_px: int
    silhouette_coverage: float
    missing_ratio: float
    spill_ratio: float
    recovered_ratio: float
    post_recovery_coverage: float
    post_alpha_mae: float
    post_spill_ratio: float
    handwear_detected: bool
    valid_layer_count: int


@dataclass
class GuardResult:
    guarded_layers: dict[str, np.ndarray]
    body_remainder: np.ndarray
    subject_mask: np.ndarray
    generated_union_pre_guard: np.ndarray
    generated_union_post_guard: np.ndarray
    missing_mask: np.ndarray
    spill_mask: np.ndarray
    reconstruction_rgba: np.ndarray
    metrics: CoverageMetrics
    verdict: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PortraitProfile:
    name: str
    critical_groups: dict[str, tuple[str, ...]]
    optional_tags: tuple[str, ...]
    ignored_absent_tags: tuple[str, ...]
    enable_shoulder_arm_heuristic: bool = True


@dataclass(frozen=True)
class PortraitEvaluation:
    critical_groups: dict[str, bool]
    missing_critical_groups: tuple[str, ...]
    handwear_detected: bool
    valid_layers: tuple[str, ...]
    crop_flags: dict[str, bool]
    semantic_success: bool


@dataclass(frozen=True)
class LayerSelectionResult:
    layers: dict[str, np.ndarray]
    score: float
    trace: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PortraitConfig:
    raw: dict[str, Any]
    source_path: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PortraitConfig":
        if path is None:
            path = Path(__file__).resolve().parents[1] / "config" / "portrait_defaults.json"
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema_version") != 1:
            raise ValueError(f"Unsupported portrait config schema: {raw.get('schema_version')!r}")
        return cls(raw=raw, source_path=str(path))

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing portrait config section: {name}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Serialize public metric/evaluation dataclasses without ndarray payloads."""
    result = asdict(value)
    for key, item in list(result.items()):
        if isinstance(item, np.generic):
            result[key] = item.item()
    return result
