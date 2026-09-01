"""Conservative recovery of missing pixels into existing semantic owners.

This module operates only on the original still, subject matte, and semantic
layers.  It has no knowledge of animation, depth, bones, meshes, or motion.
Pixels without decisive local evidence remain unresolved for body_remainder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from portrait_core.body_remainder import build_body_remainder
from portrait_core.masks import compose_union_alpha

from .image import composite_layers
from .scale import scale_area, scale_length
from .semantic import SEMANTIC_Z_ORDER

OWNERSHIP_VERSION = "1.0"
OWNERSHIP_CANDIDATES: tuple[str, ...] = (
    "topwear", "neckwear", "bottomwear", "legwear", "footwear",
    "hair", "back hair", "front hair", "hairb", "hairf",
    "head", "face", "neck", "handwear", "handwearl", "handwearr",
)
ALPHA_THRESHOLD = 13
CONTACT_RADIUS_AT_768 = 3
LOCAL_SAMPLE_RADIUS_AT_768 = 18
MIN_RECOVERY_AREA_AT_768 = 12
COLOR_ERROR_MAX = 48
WINNER_MARGIN = 10
MAX_PALETTE_COLORS = 12
GARMENT_CANDIDATES = frozenset({"topwear", "neckwear"})
ANATOMY_COMPETITORS = ("neck", "head", "face", "ears", "earl", "earr")


@dataclass(frozen=True)
class OwnershipRecoveryResult:
    layers: dict[str, np.ndarray]
    report: dict[str, Any]


def _render_with_remainder(layers: dict[str, np.ndarray], original: np.ndarray,
                           subject: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    union = compose_union_alpha(layers)
    remainder = build_body_remainder(original, subject, union)
    canonical = {**layers, "body_remainder": remainder}
    return composite_layers(canonical, original.shape[:2], order=SEMANTIC_Z_ORDER), remainder


def _exact_error(original: np.ndarray, composite: np.ndarray, subject: np.ndarray) -> int:
    mask = subject > (ALPHA_THRESHOLD / 255.0)
    error = np.abs(original[..., :3].astype(np.int32)
                   - composite[..., :3].astype(np.int32)).sum(axis=2)
    return int(error[mask].sum())


def _palette(samples: np.ndarray) -> np.ndarray:
    if not samples.size:
        return np.empty((0, 3), np.int16)
    quantized = (samples.astype(np.int16) // 8) * 8 + 4
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    return colors[np.argsort(counts)[-MAX_PALETTE_COLORS:]]


def _color_error(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if not palette.size:
        return np.full(rgb.shape[:2], 1_000_000, np.int32)
    flat = rgb.reshape(-1, 3).astype(np.int16)
    result = np.full(flat.shape[0], 1_000_000, np.int32)
    for color in palette:
        result = np.minimum(result, np.abs(flat - color).sum(axis=1))
    return result.reshape(rgb.shape[:2])


def _connected_to_seed(eligible: np.ndarray, seed: np.ndarray,
                       minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        eligible.astype(np.uint8), 8)
    keep = np.zeros_like(eligible)
    seeded_labels = set(np.unique(labels[seed]).tolist()) - {0}
    for index in seeded_labels:
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area:
            keep |= labels == index
    return keep


def _merge_back_into_front(front: np.ndarray, back: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    """Fold back-layer remainder into a semantic layer at selected pixels."""
    patched = np.array(front, copy=True)
    fa = front[..., 3].astype(np.float32) / 255.0
    ba = back[..., 3].astype(np.float32) / 255.0
    combined_a = fa + ba * (1.0 - fa)
    premultiplied = (
        front[..., :3].astype(np.float32) * fa[..., None]
        + back[..., :3].astype(np.float32) * ba[..., None] * (1.0 - fa[..., None])
    )
    combined_rgb = premultiplied / np.maximum(combined_a[..., None], 1e-6)
    patched[..., :3][mask] = np.rint(np.clip(combined_rgb[mask], 0, 255)).astype(np.uint8)
    patched[..., 3][mask] = np.rint(np.clip(combined_a[mask] * 255, 0, 255)).astype(np.uint8)
    return patched


def recover_missing_ownership(
    layer_dict: dict[str, np.ndarray],
    original_rgba: np.ndarray,
    subject_alpha: np.ndarray,
    *,
    candidates: tuple[str, ...] = OWNERSHIP_CANDIDATES,
) -> OwnershipRecoveryResult:
    """Return high-confidence missing pixels to an existing semantic layer.

    Candidate growth starts only where a missing region touches a semantic
    owner, follows locally matching source colour, resolves competition by a
    measured margin, and is accepted only when exact static RGB error does not
    increase.  Everything else remains available to body_remainder.
    """
    original = np.asarray(original_rgba)
    subject = np.asarray(subject_alpha, np.float32)
    out = {tag: np.array(layer, copy=True) for tag, layer in layer_dict.items()}
    union = compose_union_alpha(out)
    missing_alpha = np.maximum(subject - union, 0.0)
    missing = missing_alpha > (ALPHA_THRESHOLD / 255.0)
    subject_px = max(int((subject > ALPHA_THRESHOLD / 255.0).sum()), 1)
    initial_missing_px = int(missing.sum())
    before_composite, initial_remainder = _render_with_remainder(out, original, subject)
    current_error = _exact_error(original, before_composite, subject)
    if not missing.any():
        return OwnershipRecoveryResult(out, {
            "version": OWNERSHIP_VERSION,
            "initial_missing_px": 0,
            "semantic_recovered_px": 0,
            "recovered_by_tag": {},
            "unresolved_remainder_px": 0,
            "unresolved_remainder_ratio": 0.0,
            "candidates": [],
        })

    contact_radius = scale_length(CONTACT_RADIUS_AT_768, original.shape)
    sample_radius = scale_length(LOCAL_SAMPLE_RADIUS_AT_768, original.shape)
    minimum_area = scale_area(MIN_RECOVERY_AREA_AT_768, original.shape)
    contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * contact_radius + 1,) * 2)
    sample_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * sample_radius + 1,) * 2)
    rgb = original[..., :3]
    proposals: list[tuple[str, np.ndarray, np.ndarray]] = []
    audit: list[dict[str, Any]] = []

    anatomy_support = np.zeros(missing.shape, bool)
    anatomy_error = np.full(missing.shape, 1_000_000, np.int32)
    for anatomy_tag in ANATOMY_COMPETITORS:
        anatomy = out.get(anatomy_tag)
        if anatomy is None:
            continue
        anatomy_owner = np.asarray(anatomy)[..., 3] > ALPHA_THRESHOLD
        if not anatomy_owner.any():
            continue
        near = cv2.dilate(anatomy_owner.astype(np.uint8), sample_kernel).astype(bool)
        anatomy_support |= near
        anatomy_palette = _palette(np.asarray(anatomy)[..., :3][anatomy_owner])
        anatomy_error = np.minimum(anatomy_error, _color_error(rgb, anatomy_palette))

    for tag in candidates:
        layer = out.get(tag)
        if layer is None:
            continue
        owner = np.asarray(layer)[..., 3] > ALPHA_THRESHOLD
        if not owner.any():
            continue
        seed_zone = cv2.dilate(owner.astype(np.uint8), contact_kernel).astype(bool)
        seeds = missing & seed_zone
        if not seeds.any():
            continue
        local_owner = owner & cv2.dilate(missing.astype(np.uint8), sample_kernel).astype(bool)
        # The candidate's own visible material defines its palette.  Sampling
        # the flattened original here lets adjacent skin overwrite a garment
        # boundary and incorrectly teaches topwear that skin is cloth.
        palette = _palette(np.asarray(layer)[..., :3][local_owner])
        error = _color_error(rgb, palette)
        eligible = missing & (error <= COLOR_ERROR_MAX)
        anatomy_veto = np.zeros(missing.shape, bool)
        if tag in GARMENT_CANDIDATES:
            anatomy_veto = (
                anatomy_support
                & (anatomy_error <= error + WINNER_MARGIN)
            )
            eligible &= ~anatomy_veto
        recovered = _connected_to_seed(eligible, seeds & eligible, minimum_area)
        if recovered.any():
            proposals.append((tag, recovered, error))
            audit.append({
                "tag": tag,
                "seed_px": int(seeds.sum()),
                "eligible_px": int(eligible.sum()),
                "proposed_px": int(recovered.sum()),
                "anatomy_veto_px": int((missing & anatomy_veto).sum()),
            })

    recovered_by_tag: dict[str, int] = {}
    if proposals:
        best_error = np.full(missing.shape, 1_000_000, np.int32)
        second_error = np.full(missing.shape, 1_000_000, np.int32)
        best_owner = np.full(missing.shape, -1, np.int16)
        for index, (_, proposed, error) in enumerate(proposals):
            better = proposed & (error < best_error)
            second_error[better] = best_error[better]
            best_error[better] = error[better]
            best_owner[better] = index
            runner_up = proposed & ~better & (error < second_error)
            second_error[runner_up] = error[runner_up]
        decisive = (second_error >= 1_000_000) | (best_error + WINNER_MARGIN <= second_error)

        for index, (tag, proposed, _) in enumerate(proposals):
            selected = proposed & (best_owner == index) & decisive
            if not selected.any():
                continue
            trial = dict(out)
            trial[tag] = _merge_back_into_front(
                np.asarray(out[tag]), initial_remainder, selected)
            trial_composite, _ = _render_with_remainder(trial, original, subject)
            trial_error = _exact_error(original, trial_composite, subject)
            if trial_error <= current_error:
                out = trial
                current_error = trial_error
                recovered_by_tag[tag] = int(selected.sum())
            else:
                for row in audit:
                    if row["tag"] == tag:
                        row["rejected"] = "exact static RGB error increased"

    final_union = compose_union_alpha(out)
    final_remainder = build_body_remainder(original, subject, final_union)
    unresolved = int((final_remainder[..., 3] > ALPHA_THRESHOLD).sum())
    recovered_px = initial_missing_px - unresolved
    return OwnershipRecoveryResult(out, {
        "version": OWNERSHIP_VERSION,
        "initial_missing_px": initial_missing_px,
        "semantic_recovered_px": max(recovered_px, 0),
        "recovered_by_tag": recovered_by_tag,
        "unresolved_remainder_px": unresolved,
        "unresolved_remainder_ratio": round(unresolved / subject_px, 6),
        "rgb_error_delta": int(current_error - _exact_error(
            original, before_composite, subject)),
        "candidates": audit,
    })
