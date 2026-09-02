"""Static occlusion graph: which layer covers which, and by how much.

`SEETHROUGH_PORTRAIT_RESEARCH_ABSORPTION_PLAN_v0.1` section 5 asks for a
recorded structure, not a new model: the canonical semantic z-order and each
layer's own alpha already say which layer sits in front of which and how much
of the back layer it swallows. This module turns that into an explicit graph
so a future consumer (deformation guards, `portrait-autorig`) does not have to
re-derive it from raw layer stacks.

Every edge here is *measured* on this specific portrait -- which pairs touch,
and how -- never assumed from a fixed list of named relationships (plan
section 5.3: "실제 관계는 hardcode하지 않고 계산 결과로 생성한다"). The one
sanctioned exception is `_EXPOSURE_PRIOR` below: a small confidence *booster*
over evidence that already exists, never the thing that decides whether an
edge exists.

This is diagnostics only. It does not touch `layers/`, does not change repair
order, and carries no rig, motion, or deformation-safety verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scale import scale_area, scale_length
from .semantic import SEMANTIC_Z_ORDER

__all__ = ["OCCLUSION_GRAPH_VERSION", "compute_occlusion_graph"]

OCCLUSION_GRAPH_VERSION = "1.0"

# "Any presence" -- matches the alpha_threshold convention used throughout
# repair.py/ownership.py for "this layer has something here at all".
ALPHA_THRESHOLD = 10
# "Owns this pixel in the composite" -- matches seams.py/fit_seam_residual's
# owner-map convention, used only for the boundary-contact measurement.
OWNER_ALPHA_THRESHOLD = 128
# "Fully occludes" -- matches fit_layer_tone's topmost-owner threshold. A back
# layer under a pixel this opaque cannot show through at all, which is what
# separates `hidden_extent_px` (fully swallowed) from `overlap_px` (any
# footprint overlap, including a soft feathered fringe the back layer still
# shows through).
OPAQUE_ALPHA_THRESHOLD = 200

# Below this an "overlap" is antialiasing noise at a shared boundary, not a
# real occlusion relationship worth an edge.
MIN_OVERLAP_PX_AT_768 = 30
# A contact run this long already reads as "fully touching" for the risk
# score; longer contact does not make the pair more risky, only cheaper to
# measure confidently.
BOUNDARY_SATURATE_PX_AT_768 = 50
# A Marigold depth gap (depth is normalized to [0, 1]) beyond this is treated
# as full confidence that the two layers are genuinely front/back rather than
# coplanar; depth is not scaled to the picture's own contrast because depth
# maps are already normalized per Marigold's own [0, 1] convention.
DEPTH_MARGIN_SATURATE = 0.15


def _category(tag: str) -> str:
    return {
        "front hair": "hair_front", "hairf": "hair_front",
        "back hair": "hair_back", "hairb": "hair_back", "hair": "hair_back",
        "head": "face", "face": "face",
        "ears": "face", "earl": "face", "earr": "face", "earwear": "headwear",
        "topwear": "topwear", "neckwear": "neckwear", "neck": "neck",
        "handwear": "handwear", "handwearl": "handwear", "handwearr": "handwear",
        "headwear": "headwear", "eyewear": "eyewear",
        "legwear": "legwear", "footwear": "footwear", "bottomwear": "bottomwear",
    }.get(tag, tag)


# plan section 5.4's example priors, generalized to layer *categories* so a
# left/right or v2/v3 tag variant (`earl`, `handwearr`, ...) does not need its
# own row. This table only nudges `disocclusion_risk`; it never gates whether
# an edge is emitted -- that is decided purely by measured overlap above.
_EXPOSURE_PRIOR: dict[frozenset[str], float] = {
    frozenset({"hair_front", "face"}): 0.9,
    frozenset({"hair_back", "face"}): 0.85,
    frozenset({"hair_front", "hair_back"}): 0.6,
    frozenset({"topwear", "neck"}): 0.75,
    frozenset({"neckwear", "neck"}): 0.75,
    frozenset({"headwear", "hair_front"}): 0.6,
    frozenset({"headwear", "hair_back"}): 0.6,
    frozenset({"handwear", "topwear"}): 0.5,
    frozenset({"bottomwear", "legwear"}): 0.35,
    frozenset({"legwear", "footwear"}): 0.25,
}
_DEFAULT_EXPOSURE = 0.5


def _likely_motion_exposure(front_tag: str, back_tag: str) -> float:
    pair = frozenset({_category(front_tag), _category(back_tag)})
    return _EXPOSURE_PRIOR.get(pair, _DEFAULT_EXPOSURE)


def _depth_terms(depth_dict: dict[str, np.ndarray] | None, front_tag: str, back_tag: str,
                  overlap_mask: np.ndarray) -> tuple[float | None, float | None]:
    """`(depth_margin, depth_confidence)`; both `None` when depth is unavailable.

    Marigold depth is larger-means-further (see `depth.py`). A positive
    margin -- the back layer measured further away, in the overlap region
    only -- agrees with the z-order call and is what makes depth trustworthy
    evidence here; a non-positive margin means depth contradicts or cannot
    distinguish the ordering and contributes no confidence.
    """
    if not depth_dict or front_tag not in depth_dict or back_tag not in depth_dict:
        return None, None
    if not overlap_mask.any():
        return None, None
    front_depth = float(np.median(depth_dict[front_tag][overlap_mask]))
    back_depth = float(np.median(depth_dict[back_tag][overlap_mask]))
    margin = back_depth - front_depth
    confidence = float(np.clip(margin / DEPTH_MARGIN_SATURATE, 0.0, 1.0)) if margin > 0 else 0.0
    return round(margin, 4), round(confidence, 4)


def compute_occlusion_graph(
    layer_dict: dict[str, np.ndarray],
    *,
    order: tuple[str, ...] = SEMANTIC_Z_ORDER,
    depth_dict: dict[str, np.ndarray] | None = None,
    alpha_threshold: int = ALPHA_THRESHOLD,
) -> dict[str, Any]:
    """Return the measured front/back contact graph for `layer_dict`.

    No inference model runs here (plan section 5.2): edges are derived only
    from each layer's own alpha and the existing canonical z-order. `depth_dict`
    is the same optional `{tag: float32 HxW}` shape `depth.estimate_layer_depths`
    returns; when omitted, `depth_margin` is reported as `null` and the risk
    score's depth term is left at full confidence rather than penalizing an
    edge for information nobody supplied.
    """
    if not layer_dict:
        return {"version": OCCLUSION_GRAPH_VERSION, "alpha_threshold": alpha_threshold,
                "depth_available": depth_dict is not None, "edges": []}

    shape = np.asarray(next(iter(layer_dict.values()))).shape[:2]
    min_overlap = scale_area(MIN_OVERLAP_PX_AT_768, shape)
    boundary_saturate = scale_length(BOUNDARY_SATURATE_PX_AT_768, shape)

    def rank(tag: str) -> int:
        return order.index(tag) if tag in order else -1

    tags = [tag for tag in sorted(layer_dict, key=rank)
            if np.asarray(layer_dict[tag])[..., 3].max() > alpha_threshold]

    present = {tag: np.asarray(layer_dict[tag])[..., 3] > alpha_threshold for tag in tags}
    opaque = {tag: np.asarray(layer_dict[tag])[..., 3] > OPAQUE_ALPHA_THRESHOLD for tag in tags}
    areas = {tag: int(present[tag].sum()) for tag in tags}

    owner = np.full(shape, -1, np.int16)
    for index, tag in enumerate(tags):
        owner[np.asarray(layer_dict[tag])[..., 3] > OWNER_ALPHA_THRESHOLD] = index

    edges: list[dict[str, Any]] = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            back_tag, front_tag = tags[i], tags[j]  # higher rank (j) sits in front
            overlap_mask = present[back_tag] & present[front_tag]
            overlap_px = int(overlap_mask.sum())
            if overlap_px < min_overlap:
                continue

            hidden_mask = present[back_tag] & opaque[front_tag]
            hidden_extent_px = int(hidden_mask.sum())

            contact = np.zeros(shape, bool)
            for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rolled = np.roll(owner, shift, axis=(0, 1))
                contact |= ((owner == j) & (rolled == i)) | ((owner == i) & (rolled == j))
            visible_boundary_px = int(contact.sum())
            if visible_boundary_px == 0 and hidden_extent_px == 0:
                # Footprints overlap but neither layer ever actually owns a
                # pixel over the other (e.g. both stay below OWNER/OPAQUE
                # threshold there) -- not a real contact relationship.
                continue

            back_area = max(areas[back_tag], 1)
            front_area = max(areas[front_tag], 1)
            overlap_ratio = overlap_px / back_area
            hidden_ratio = hidden_extent_px / back_area
            boundary_contact = min(1.0, visible_boundary_px / max(boundary_saturate, 1))
            # How much of the *smaller* footprint the overlap covers -- a
            # confident real pair, not two large layers grazing a corner.
            confidence = float(np.clip(overlap_px / min(front_area, back_area), 0.0, 1.0))

            depth_margin, depth_confidence = _depth_terms(
                depth_dict, front_tag, back_tag, overlap_mask)
            exposure = _likely_motion_exposure(front_tag, back_tag)
            risk = (overlap_ratio * boundary_contact * hidden_ratio * exposure
                    * (depth_confidence if depth_confidence is not None else 1.0))

            edges.append({
                "front": front_tag,
                "back": back_tag,
                "overlap_px": overlap_px,
                "visible_boundary_px": visible_boundary_px,
                "hidden_extent_px": hidden_extent_px,
                "depth_margin": depth_margin,
                "confidence": round(confidence, 4),
                "disocclusion_risk": round(float(np.clip(risk, 0.0, 1.0)), 4),
            })

    edges.sort(key=lambda edge: -edge["disocclusion_risk"])
    return {
        "version": OCCLUSION_GRAPH_VERSION,
        "alpha_threshold": alpha_threshold,
        "depth_available": depth_dict is not None,
        "edges": edges,
    }
