"""Swappable proposal/critic backend contracts for external models.

`SEETHROUGH_PORTRAIT_RESEARCH_ABSORPTION_PLAN_v0.1` (docs/research absorption
plan, section 6) sets one rule for every external model this project ever
wires in: it returns a *candidate*, never a canonical layer, and the
candidate is validated the same way any other repair proposal is -- accepted
only when it does not regress static fidelity. This module is that contract:
a proposal metadata type plus a `Protocol` per backend role, with the current
deterministic implementation registered as the default for each role.

This is Phase A of the plan ("structure only, zero behavior change"): nothing
in `generation.py` or `repair.py` imports this module, and the default
backends below only call the existing functions -- they do not re-implement
or alter any logic. A later phase (P1: ViTMatte ROI, EVF-SAM2 proposals, ...)
wires a real alternative backend in behind these same interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from portrait_core import resolve_subject_mask
from portrait_core.types import MaskEvidence, PortraitConfig

from .repair import fit_edge_alpha

__all__ = [
    "BackendProposal",
    "SubjectMatteBackend",
    "DecompositionBackend",
    "SemanticProposalBackend",
    "EdgeMatteBackend",
    "CurrentDeterministicMatteBackend",
    "SeeThroughDecompositionBackend",
    "CurrentEdgeAlphaBackend",
]

# Tracks this wrapper module, not the algorithm it wraps -- `fit_edge_alpha`
# and `resolve_subject_mask` carry their own version/provenance already.
CURRENT_BACKEND_VERSION = "1.0"


@dataclass(frozen=True)
class BackendProposal:
    """The common envelope every backend result carries (plan section 6.3).

    `value` is the actual candidate output (a mask, a layer dict, an alpha
    array -- whatever the role produces); everything else is the JSON
    provenance record a diagnostics file can hold as-is via `metadata()`.
    `accepted` always starts `False` here: only the deterministic validator
    that later consumes a proposal may flip it, and it does so on its own
    copy, never on the instance a backend returned.
    """

    backend: str
    backend_version: str
    target: str
    confidence: float
    value: Any
    roi: tuple[int, int, int, int] | None = None
    reason: str = ""
    source: str = "proposal"
    accepted: bool = False

    def metadata(self) -> dict[str, Any]:
        """The section 6.3 JSON contract, without the (non-serializable) payload."""
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "target": self.target,
            "confidence": self.confidence,
            "roi": list(self.roi) if self.roi is not None else None,
            "reason": self.reason,
            "source": self.source,
            "accepted": self.accepted,
        }


@runtime_checkable
class SubjectMatteBackend(Protocol):
    """Candidates: current flat-key, anime-segmentation, BiRefNet, manual mask."""

    def matte(self, image: np.ndarray) -> BackendProposal: ...


@runtime_checkable
class DecompositionBackend(Protocol):
    """Default candidate: See-through. No other heavy decomposition backend
    is added by this plan (see section 21, non-goals)."""

    def decompose(self, image: np.ndarray, subject_mask: np.ndarray) -> BackendProposal: ...


@runtime_checkable
class SemanticProposalBackend(Protocol):
    """Candidates: EVF-SAM2, SAM2 point/box mode, face parser.

    No default implementation ships in this phase -- this is P1 (plan
    section 9), added later behind the same interface.
    """

    def propose(self, image: np.ndarray, label: str,
                roi: tuple[int, int, int, int] | None = None) -> BackendProposal: ...


@runtime_checkable
class EdgeMatteBackend(Protocol):
    """Candidates: current deterministic alpha fitting, ViTMatte.

    `trimap` is intentionally untyped here: the current deterministic backend
    takes the whole layer dict it refits, while a real matting backend (P1:
    ViTMatte ROI) takes a single-channel unknown/fg/bg trimap crop. Each
    backend documents its own expected shape.
    """

    def refine(self, image: np.ndarray, trimap: Any,
               roi: tuple[int, int, int, int] | None = None) -> BackendProposal: ...


@dataclass(frozen=True)
class CurrentDeterministicMatteBackend:
    """Wraps `portrait_core.resolve_subject_mask`: today's default subject
    matte, unchanged. Registered so a caller can hold a `SubjectMatteBackend`
    reference without special-casing "no backend configured yet".
    """

    config: PortraitConfig | None = None
    provided_mask: np.ndarray | None = None
    generated_layers: dict[str, np.ndarray] | None = None

    def matte(self, image: np.ndarray) -> BackendProposal:
        evidence: MaskEvidence = resolve_subject_mask(
            image,
            provided_mask=self.provided_mask,
            generated_layers=self.generated_layers,
            config=self.config,
        )
        confidence = {"HIGH": 0.95, "MEDIUM": 0.7, "LOW": 0.4}.get(evidence.confidence, 0.5)
        return BackendProposal(
            backend="current_deterministic",
            backend_version=CURRENT_BACKEND_VERSION,
            target="subject_mask",
            confidence=confidence,
            value=evidence,
            reason=evidence.source,
        )


@dataclass(frozen=True)
class SeeThroughDecompositionBackend:
    """Wraps the current See-through diffusion pass as a `DecompositionBackend`.

    See-through's real call needs a live pipeline, device, and text
    embeddings that only the caller already running the diffusion loop has
    (see `generation.run_diffusion_stage`), so this backend is built around
    an already-bound diffusion callable -- the same shape as
    `run_portrait_pipeline`'s internal `_diffuse` closure -- rather than
    reaching for those objects itself.
    """

    diffuse: Callable[[], dict[str, np.ndarray]]
    backend_version: str = CURRENT_BACKEND_VERSION

    def decompose(self, image: np.ndarray, subject_mask: np.ndarray) -> BackendProposal:
        layers = self.diffuse()
        return BackendProposal(
            backend="see_through",
            backend_version=self.backend_version,
            target="layer_set",
            confidence=1.0,
            value=layers,
            reason="see_through_diffusion",
        )


@dataclass(frozen=True)
class CurrentEdgeAlphaBackend:
    """Wraps `repair.fit_edge_alpha`: today's deterministic edge-alpha solve.

    Runs the existing whole-canvas per-layer solve rather than a single ROI
    refine, since that is what the current implementation does. `roi` is
    accepted for interface conformance and carried through to the proposal
    metadata unused -- a future ViTMatte ROI backend (plan section 7) is what
    actually crops to it.
    """

    backend_version: str = CURRENT_BACKEND_VERSION

    def refine(self, image: np.ndarray, trimap: dict[str, np.ndarray],
               roi: tuple[int, int, int, int] | None = None) -> BackendProposal:
        layers, moved = fit_edge_alpha(trimap, image)
        return BackendProposal(
            backend="current_deterministic",
            backend_version=self.backend_version,
            target="edge_alpha",
            confidence=1.0,
            value=layers,
            roi=roi,
            reason=f"fit_edge_alpha ({len(moved)} layers touched)",
        )
