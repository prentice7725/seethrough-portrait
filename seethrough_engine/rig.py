"""Automatic pseudo-2.5D rig construction from Portrait Mode layers.

Stages A-D of `docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md`: normalize the
semantic tags, split `body_remainder` into head/neck/body regions, split the
both-eyes-in-one-layer v3 tags into left and right, detect anchors, and assign
each part a group, a depth, a head-follow weight, and a draw index. The result
is `rig_manifest.json`, the contract the browser runtime (Stage E) consumes.

No torch, no diffusers -- numpy and cv2 only, like `spine.py` -- so the whole
rig can be unit tested on synthetic layers without a GPU or a model download.

Two things this deliberately does differently from the Spine exporter:

* It keys on the **pre-rename semantic tags** (`back hair`, not `back-hair`).
  `DEFAULT_SPINE_NAMES` is a Spine file-naming concern and stays there.
* Coordinates stay in canvas pixels, top-left origin, Y down -- the same
  convention as `xyxy` -- rather than Spine's bottom-centre Y-up. Only the
  Spine exporter converts, and only at its own boundary.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Collection
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .spine import SEMANTIC_Z_ORDER, crop_to_alpha

__all__ = [
    "GROUP_HEAD",
    "GROUP_NECK",
    "GROUP_BODY",
    "HEAD_REMAINDER",
    "NECK_REMAINDER",
    "BODY_REMAINDER",
    "EYE_SPLIT_TAGS",
    "RIG_Z_ORDER",
    "group_for_tag",
    "depth_table",
    "split_remainder",
    "split_eyes",
    "detect_anchors",
    "RECLAIM_PAIRS",
    "reclaim_occluded",
    "fit_edge_alpha",
    "composite_layers",
    "composite_fidelity",
    "build_rig",
    "write_rig_project",
    "rebuild_run_rig",
]

GROUP_HEAD = "head"
GROUP_NECK = "neck"
GROUP_BODY = "body"

HEAD_REMAINDER = "head_remainder"
NECK_REMAINDER = "neck_remainder"
BODY_REMAINDER = "body_remainder"

# Group membership is a table over the known tag vocabulary rather than the
# centroid heuristic a PSD-based rigger has to fall back on: Portrait Mode's
# tags are fixed by the model's `tag_version`, so guessing is never necessary.
# Both the v2 and v3 spellings are listed (see `layers.VALID_BODY_PARTS_*`).
HEAD_TAGS = frozenset({
    "back hair", "front hair", "hair", "hairb", "hairf",
    "head", "headwear", "face",
    "ears", "earl", "earr", "earwear",
    "eyewhite", "eyewhitel", "eyewhiter",
    "irides", "iridesl", "iridesr",
    "eyebrow", "eyebrowl", "eyebrowr", "browl", "browr",
    "eyelash", "eyelashl", "eyelashr",
    "eyes", "eyel", "eyer", "eyewear",
    "nose", "mouth",
    HEAD_REMAINDER,
})
NECK_TAGS = frozenset({"neck", "neckwear", NECK_REMAINDER})

# v3 packs both eyes into one layer per feature, so a rig that wants to blink
# or wink has to separate them itself. `eyes` is the v2 spelling of the same
# problem. The `l`/`r` suffixes match the names `DEFAULT_SPINE_NAMES` already
# knows, so a later Spine export maps them for free.
EYE_SPLIT_TAGS = ("eyewhite", "irides", "eyelash", "eyebrow", "eyes")

# Head-follow weights. Starting values, expected to move once the preview
# exists -- see the feasibility doc's open risk on `back hair`.
HEAD_WEIGHT = 1.00
BODY_WEIGHT = 0.16

# The neck bridges the head and the body, so its gradient has to *end* on their
# weights exactly. These are not free parameters: a borrowed 0.55 at the top
# against a head at 1.00 put a 0.45 discontinuity right at the jaw, and with
# only the top quarter of the neck visible above a stand collar, the head
# visibly slid off a neck moving at half its speed.
NECK_TOP_WEIGHT = HEAD_WEIGHT
NECK_BOTTOM_WEIGHT = BODY_WEIGHT

# A stand collar (a gakuran, a turtleneck) touches the jaw, so leaving it fully
# rigid while the head tilts reads as the chin cutting into it. A garment whose
# top edge overlaps the neck therefore takes **the neck's own weight function**
# over that overlap, rather than a ramp of its own.
#
# That is not a stylistic choice. `reclaim_occluded` cuts a window in the
# garment for the neck to show through, so the window and its contents are two
# sides of one seam: give them different weights and the window's edge slices
# the neck as the head turns. An independent collar constant of 0.45 against a
# neck at 0.571 on the collar line opened a 2.05 px crack there; sharing the
# function leaves only what the two parts' different depths contribute, 0.43 px.
#
# Below the neck the gradient has already reached BODY_WEIGHT, so the rest of
# the garment is unaffected.
COLLAR_TAGS = frozenset({"topwear", "neckwear"})

# The head rotates about a point near the *bottom* of the neck, which is what
# makes a tilt read as a neck bending rather than a head sliding sideways.
NECK_PIVOT_RATIO = 0.85

# Uniform grid meshing, quoted against a 768px canvas and scaled from there.
# Parts that deform along a gradient get the finer cell, since that is where a
# coarse grid shows as faceting.
MESH_REFERENCE_SIZE = 768
MESH_CELL_PX = 42
MESH_CELL_FINE_PX = 30

# A garment layer sometimes comes back holding the skin visible through its own
# opening -- opaque, and in front of the neck, so it paints flat light skin over
# the neck's shading and hides the neck almost entirely. Measured on A-001:
# `neck` visible on 272 of its 12096 pixels behind a V-neck, 117 of 7533 behind
# a stand collar.
#
# Only these pairs are contested. Every other pairing tried either made the
# composite worse (`topwear` over `face`, `head` over `face`) or moved tens of
# thousands of pixels for no measurable gain (`back hair` over `head`), which is
# the more dangerous outcome of the two.
RECLAIM_PAIRS: tuple[tuple[str, str], ...] = (("topwear", "neck"), ("neckwear", "neck"))
# How much better the back layer has to explain a pixel before that pixel can
# seed a region, summed over RGB, and the smallest seed worth following.
RECLAIM_MARGIN = 12
RECLAIM_MIN_AREA_AT_768 = 200
# Blur applied to the handover so the garment fades into the layer behind it
# rather than stopping at a hard edge. Sub-pixel motion across a hard, jagged
# alpha cut is what made the seam glitter once per breath.
RECLAIM_FEATHER_PX = 1.0

# The same question at every layer's own boundary rather than between one named
# pair, and answered by solving rather than by voting. A layer's edge alpha is
# the one number the decomposition has no way to check: it decides how much of
# the layer shows against what is behind, and the original says exactly what
# that mixture should be. Where `face` ends at the chin its last rows composite
# to luma 80 and 162 against an original of 223 and 225 -- too much of a rim
# that should barely show. Where the neck meets the garment the two alphas sum
# to 1.64 and one row darkens -- too much of both. Where `head` fades out over
# two rows while the original's chin contour fills both, there is too little.
#
# So: a = ((O - B) . (F - B)) / |F - B|^2, the least-squares coverage that makes
# the front layer over what is behind equal the original. Clamped to [0, 1], and
# taken only where the two differ enough for the answer to mean anything.
EDGE_BAND_PX = 3          # how far inside the boundary to refit
EDGE_OUTSIDE_PX = 1       # ... and how far outside, since an edge can be short
EDGE_MIN_CONTRAST = 25    # per channel, below which the solve is noise

# A layer that is all edge is an outline, and an outline is *meant* to be dark.
# The share of a layer lying deeper than the band separates the two kinds: on
# A-001 `mouth` is 0% interior, `eyebrow` 2%, `eyelash` 13%, `nose` 15%, against
# `face` 94%, `head` 94%, `topwear` 97%. Refitting the first kind thins the
# stroke until the feature fades, which is the opposite of the fix.
#
# Three quarters rather than a half is the conservative reading of the same
# rule: an open mouth is a surface in one run and a stroke in the next.
EDGE_MIN_INTERIOR = 0.75

MANIFEST_VERSION = "0.1"
RIG_SUBDIR = "rig"

# `max_x` is measured, not chosen. Sweeping the turn on A-001 and counting the
# largest contiguous region where hair-dark pixels became skin-light -- the
# parallax sliding the hair off the head it used to cover -- the reveal stays
# scattered up to 0.8 (839 px) and merges into one visible gash by 1.0
# (2095 px). See the feasibility doc's H1 note.
DEFAULT_MOTION: dict[str, Any] = {
    "head_turn": {"max_x": 0.8, "max_y": 0.8},
    "head_tilt": {"max_deg": 2.0, "pivot": "neck_pivot"},
    "breathing": {"period_s": 4.0, "amplitude_px": 3.0},
    #  places the closed lid inside the eye opening: 1.0 is the
    # lower lid, 0.5 the centre. Closing onto the centre leaves the lash as a
    # bar floating in the socket with skin above and below, which reads as a
    # squint; a real lid comes down onto the lower one.
    "blink": {"close_s": 0.08, "hold_s": 0.34, "open_s": 0.16,
              "interval_s": [1.6, 5.4], "lid_ratio": 0.85, "lid_thickness": 0.18},
}


def _rig_z_order() -> tuple[str, ...]:
    """`SEMANTIC_Z_ORDER` with the two new remainder regions inserted.

    Each region sits directly behind the group it now moves with, so the
    recovered pixels around the head are drawn behind the head *and* follow
    it -- which is the whole point of splitting them. `body_remainder` keeps
    its position at the very back, where `SEMANTIC_Z_ORDER` already puts it.
    """
    order = list(SEMANTIC_Z_ORDER)
    for tag, anchor in ((NECK_REMAINDER, "neck"), (HEAD_REMAINDER, "head")):
        order.insert(order.index(anchor), tag)
    return tuple(order)


RIG_Z_ORDER: tuple[str, ...] = _rig_z_order()


def group_for_tag(tag: str) -> str:
    """Which movement group a tag belongs to. Unknown tags fall to `body`,
    the conservative choice: a mystery layer that fails to follow the head is
    a missed opportunity, while one that follows it can tear off the torso."""
    if tag in HEAD_TAGS:
        return GROUP_HEAD
    if tag in NECK_TAGS:
        return GROUP_NECK
    return GROUP_BODY


def depth_table() -> dict[str, float]:
    """Per-tag parallax depth, 0 near to 1 far, read straight off the
    canonical back-to-front order.

    This is the default depth source. Marigold is an override, not a
    prerequisite: parallax needs the layers' *relative* order, which the tag
    vocabulary already fixes, and estimating it costs a 3 GB model plus a pass
    per run (see `depth.estimate_layer_depths`).
    """
    last = len(RIG_Z_ORDER) - 1
    return {tag: round(1.0 - i / last, 4) for i, tag in enumerate(RIG_Z_ORDER)}


_DEPTH_TABLE = depth_table()
# Unknown tags sit at the back, matching how `spine.semantic_rank` ranks them.
UNKNOWN_DEPTH = 1.0


def _alpha_of(img: np.ndarray) -> np.ndarray:
    return np.asarray(img)[..., 3]


def _mask_of(img: np.ndarray, alpha_threshold: int) -> np.ndarray:
    return _alpha_of(img) > alpha_threshold


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """`(x1, y1, x2, y2)` around the True pixels, or None when there are none."""
    nz = cv2.findNonZero(mask.astype(np.uint8))
    if nz is None:
        return None
    x, y, w, h = cv2.boundingRect(nz)
    return int(x), int(y), int(x + w), int(y + h)


def _centroid(alpha: np.ndarray, alpha_threshold: int) -> tuple[float, float] | None:
    """Alpha-weighted centre of a layer, or None when it is empty. Weighting by
    alpha rather than by the binary mask keeps soft edges from pulling the
    centre outward, which matters for small parts like `irides`."""
    a = np.asarray(alpha, dtype=np.float32)
    a = np.where(a > alpha_threshold, a, 0.0)
    total = float(a.sum())
    if total <= 0.0:
        return None
    ys, xs = np.indices(a.shape, dtype=np.float32)
    return float((xs * a).sum() / total), float((ys * a).sum() / total)


def _union_alpha(layer_dict: dict[str, np.ndarray], tags: Collection[str],
                 alpha_threshold: int) -> np.ndarray | None:
    """Binary union of the named layers' alpha, or None when none are present."""
    out: np.ndarray | None = None
    for tag in tags:
        img = layer_dict.get(tag)
        if img is None:
            continue
        mask = _mask_of(img, alpha_threshold)
        out = mask if out is None else (out | mask)
    return out


def _group_tags(layer_dict: dict[str, np.ndarray], group: str) -> list[str]:
    return [tag for tag in layer_dict if group_for_tag(tag) == group]


def _rgba_with_alpha(source_rgba: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.array(source_rgba, dtype=np.uint8, copy=True)
    out[..., 3] = np.rint(np.clip(alpha, 0, 255)).astype(np.uint8)
    return out


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Per-pixel distance to the nearest True pixel of `mask`.

    `distanceTransform` measures distance to the nearest *zero*, so the mask is
    inverted on the way in.
    """
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def split_remainder(remainder_rgba: np.ndarray, layer_dict: dict[str, np.ndarray], *,
                    alpha_threshold: int = 10) -> dict[str, np.ndarray]:
    """Partition the Silhouette Guard's recovered pixels into head, neck, and
    body regions, returned as full-canvas RGBA under the three remainder tags.
    Empty regions are omitted.

    A single canvas-wide `body_remainder` pinned to the torso is what leaves a
    ghost head silhouette behind when the head moves: the recovered pixels
    *around* the head stay put. Splitting by region lets each piece move with
    whatever it was recovered from.

    The split is by **nearest owner**, not by bounding box. Hair falling across
    a shoulder then divides along the actual boundary between the head and body
    layers instead of at an arbitrary horizontal cut. The neck band is carved
    out first, because a neck sits between the two and would otherwise be
    arbitrarily assigned by whichever union happens to be a pixel closer.

    Note that this is a rig concern only. The Silhouette Guard's scoring and
    the portrait report keep seeing the single undivided remainder, so nothing
    about the verdict changes.
    """
    remainder = np.asarray(remainder_rgba)
    if remainder.ndim != 3 or remainder.shape[-1] != 4:
        raise ValueError(f"remainder must be HxWx4, got {remainder.shape}")
    alpha = remainder[..., 3].astype(np.float32)
    live = alpha > alpha_threshold
    if not np.any(live):
        return {}

    head_union = _union_alpha(layer_dict, _group_tags(layer_dict, GROUP_HEAD), alpha_threshold)
    body_union = _union_alpha(layer_dict, _group_tags(layer_dict, GROUP_BODY), alpha_threshold)
    neck_union = _union_alpha(layer_dict, _group_tags(layer_dict, GROUP_NECK), alpha_threshold)

    regions: dict[str, np.ndarray] = {}

    remaining = live.copy()
    neck_box = _bbox(neck_union) if neck_union is not None else None
    if neck_box is not None:
        x1, y1, x2, y2 = neck_box
        band = np.zeros_like(live)
        band[y1:y2, x1:x2] = True
        neck_part = remaining & band
        if np.any(neck_part):
            regions[NECK_REMAINDER] = _rgba_with_alpha(remainder, alpha * neck_part)
            remaining &= ~neck_part

    if not np.any(remaining):
        return regions

    have_head = head_union is not None and bool(head_union.any())
    have_body = body_union is not None and bool(body_union.any())
    if have_head and have_body:
        head_part = remaining & (_distance_to(head_union) <= _distance_to(body_union))
    elif have_head:
        head_part = remaining
    else:
        head_part = np.zeros_like(remaining)
    body_part = remaining & ~head_part

    if np.any(head_part):
        regions[HEAD_REMAINDER] = _rgba_with_alpha(remainder, alpha * head_part)
    if np.any(body_part):
        regions[BODY_REMAINDER] = _rgba_with_alpha(remainder, alpha * body_part)
    return regions


def split_eyes(layer_dict: dict[str, np.ndarray], face_center_x: float, *,
               tags: Collection[str] = EYE_SPLIT_TAGS, alpha_threshold: int = 10,
               min_area_ratio: float = 0.05, dilate_px: int = 2) -> dict[str, np.ndarray]:
    """Split each both-eyes-in-one-layer tag into `{tag}l` / `{tag}r`.

    Connected components on the layer's alpha, small ones discarded as noise,
    each remaining component assigned by its centroid X against the face
    centre. No face-detection model: at portrait framing the two eyes are
    reliably separate components on opposite sides of the face centre, and a
    detector would be a second model download to answer a question the alpha
    already answers.

    `l`/`r` are **image** left and right, not the character's -- the same sense
    `DEFAULT_SPINE_NAMES` uses.

    Returns only the tags that actually split into both sides; a tag that
    yields components on one side only is left alone for the caller to keep
    whole, which is the right outcome for a three-quarter view or a layer where
    one eye is occluded.
    """
    out: dict[str, np.ndarray] = {}
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8) if dilate_px > 0 else None

    for tag in tags:
        img = layer_dict.get(tag)
        if img is None:
            continue
        arr = np.asarray(img)
        alpha = arr[..., 3].astype(np.float32)
        mask = (alpha > alpha_threshold).astype(np.uint8)
        if not mask.any():
            continue

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 2:  # background plus at most one blob: nothing to split
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_area = float(areas.max()) * min_area_ratio
        left = np.zeros(mask.shape, dtype=bool)
        right = np.zeros(mask.shape, dtype=bool)
        for label in range(1, count):
            if float(stats[label, cv2.CC_STAT_AREA]) < keep_area:
                continue
            side = left if float(centroids[label][0]) < face_center_x else right
            side |= labels == label
        if not left.any() or not right.any():
            continue

        for suffix, side in (("l", left), ("r", right)):
            side_mask = side
            if kernel is not None:
                # Dilate, then intersect with the layer's own alpha: the point
                # is to close the seam between adjacent components, not to
                # invent coverage the layer never had.
                side_mask = cv2.dilate(side.astype(np.uint8), kernel).astype(bool) & (mask > 0)
            out[f"{tag}{suffix}"] = _rgba_with_alpha(arr, alpha * side_mask)

    return out


def detect_anchors(layer_dict: dict[str, np.ndarray], frame_size: tuple[int, int], *,
                   alpha_threshold: int = 10) -> dict[str, list[float]]:
    """Rig anchor points in canvas pixels, derived from layer alpha alone.

    Anchors that cannot be derived are **omitted rather than guessed**: a
    fabricated eye position is worse than an absent one, because the runtime
    can skip a motion it has no anchor for but cannot tell a wrong anchor from
    a right one.
    """
    canvas_h, canvas_w = int(frame_size[0]), int(frame_size[1])
    anchors: dict[str, list[float]] = {}

    def put(name: str, point: tuple[float, float] | None) -> None:
        if point is not None:
            anchors[name] = [round(point[0], 2), round(point[1], 2)]

    face_center = None
    for tag in ("face", "head"):
        img = layer_dict.get(tag)
        if img is not None:
            face_center = _centroid(_alpha_of(img), alpha_threshold)
            if face_center is not None:
                break
    if face_center is None:
        head_union = _union_alpha(layer_dict, _group_tags(layer_dict, GROUP_HEAD), alpha_threshold)
        if head_union is not None and head_union.any():
            face_center = _centroid(head_union.astype(np.float32) * 255.0, alpha_threshold)
    put("face_center", face_center)

    for name, candidates in (("eye_left", ("eyewhitel", "iridesl", "eyel")),
                             ("eye_right", ("eyewhiter", "iridesr", "eyer"))):
        for tag in candidates:
            img = layer_dict.get(tag)
            if img is not None:
                point = _centroid(_alpha_of(img), alpha_threshold)
                if point is not None:
                    put(name, point)
                    break

    mouth = layer_dict.get("mouth")
    if mouth is not None:
        put("mouth", _centroid(_alpha_of(mouth), alpha_threshold))

    neck_box = neck_bbox(layer_dict, alpha_threshold=alpha_threshold)
    if neck_box is not None:
        x1, y1, x2, y2 = neck_box
        put("neck_pivot", ((x1 + x2) / 2.0, y1 + (y2 - y1) * NECK_PIVOT_RATIO))
    elif face_center is not None:
        # No neck layer: hinge at the bottom of the head instead, which is at
        # least in the right place even if the lever arm is short.
        head_union = _union_alpha(layer_dict, _group_tags(layer_dict, GROUP_HEAD), alpha_threshold)
        head_box = _bbox(head_union) if head_union is not None else None
        if head_box is not None:
            put("neck_pivot", (face_center[0], float(head_box[3])))

    body_box = None
    topwear = layer_dict.get("topwear")
    if topwear is not None:
        body_box = _bbox(_mask_of(topwear, alpha_threshold))
    if body_box is None:
        body_union = _union_alpha(layer_dict, _group_tags(layer_dict, GROUP_BODY), alpha_threshold)
        body_box = _bbox(body_union) if body_union is not None else None
    if body_box is not None:
        put("body_pivot", ((body_box[0] + body_box[2]) / 2.0, float(body_box[3])))
    else:
        put("body_pivot", (canvas_w / 2.0, float(canvas_h)))

    return anchors


def neck_bbox(layer_dict: dict[str, np.ndarray], *,
              alpha_threshold: int = 10) -> tuple[int, int, int, int] | None:
    """Bounds of the whole neck group, which is what the neck weight gradient
    is measured against. Taken over the group rather than per part so that
    `neck` and `neck_remainder` share one gradient -- give them their own and
    the two deform differently along the seam between them."""
    union = _union_alpha(layer_dict, _group_tags(layer_dict, GROUP_NECK), alpha_threshold)
    return _bbox(union) if union is not None else None


def reclaim_occluded(layer_dict: dict[str, np.ndarray], original_rgba: np.ndarray, *,
                     pairs: tuple[tuple[str, str], ...] = RECLAIM_PAIRS,
                     alpha_threshold: int = 10, margin: int = RECLAIM_MARGIN,
                     min_area: int | None = None, feather: float | None = None,
                     ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Give a contested pixel to whichever layer explains the original better.

    Where a garment and the neck both claim a pixel, the decomposition has
    already been told which is right: the original image is ground truth here,
    and one of the two layers matches it. Resolving the tie that way is not a
    guess -- it is measured, per pixel, rather than assumed for a region.

    Returns `(layers, {"front<-back": pixels})`. The front layer's alpha is
    cleared where the back wins; nothing else is touched, and no layer's colour
    is altered.

    The margin decides **which regions** change hands, not where each region
    ends. Applying it per pixel cut a decisively-neck area into lace: on A-001
    the neck explained 93% of one band better, but only 59% cleared the margin,
    so the handover stopped in ragged mid-region and left skin-coloured garment
    behind. So a margin-qualified `core` seeds the region, and the region's
    extent then follows plain "the back layer is better" out to where that
    stops being true -- which is a real edge in the picture rather than an
    artefact of the threshold. Boundary roughness drops from 0.121 to 0.079,
    against 0.07 for a smooth blob of the same size.

    The handover is feathered, because a hard alpha cut with a jagged boundary
    glitters as sub-pixel motion moves it: 364 hard-edged boundary pixels
    become 54. Feathering is clamped by the back layer's own alpha so it can
    never open a gap where there is nothing behind to show through.
    """
    original = np.asarray(original_rgba)[..., :3].astype(np.int32)
    feather = RECLAIM_FEATHER_PX if feather is None else feather
    if min_area is None:
        scale = original.shape[0] * original.shape[1] / (768.0 * 768.0)
        min_area = max(64, int(round(RECLAIM_MIN_AREA_AT_768 * scale)))

    out = dict(layer_dict)
    moved: dict[str, int] = {}
    kernel = np.ones((5, 5), np.uint8)

    for front_tag, back_tag in pairs:
        front, back = out.get(front_tag), out.get(back_tag)
        if front is None or back is None:
            continue
        front, back = np.asarray(front), np.asarray(back)
        contested = (front[..., 3] > alpha_threshold) & (back[..., 3] > alpha_threshold)
        if not contested.any():
            continue

        front_err = np.abs(original - front[..., :3].astype(np.int32)).sum(axis=2)
        back_err = np.abs(original - back[..., :3].astype(np.int32)).sum(axis=2)

        # A seed: the back layer is decisively better here, over an area big
        # enough that it is not noise.
        core = cv2.morphologyEx(
            (contested & (back_err + margin < front_err)).astype(np.uint8),
            cv2.MORPH_OPEN, kernel).astype(bool)
        if not core.any():
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
        core = np.isin(labels, [i for i in range(1, count)
                                if stats[i, cv2.CC_STAT_AREA] >= min_area])
        if not core.any():
            continue

        # The extent: wherever the back layer is simply better, out to where it
        # stops being so. Only regions holding a seed are taken.
        region = cv2.morphologyEx(
            (contested & (back_err < front_err)).astype(np.uint8),
            cv2.MORPH_CLOSE, kernel).astype(bool) & contested
        count, labels, _, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
        seeded = set(np.unique(labels[core]).tolist()) - {0}
        take = np.isin(labels, list(seeded)) if seeded else np.zeros_like(region)
        if not take.any():
            continue

        handover = take.astype(np.float32)
        if feather > 0:
            handover = cv2.GaussianBlur(handover, (0, 0), float(feather))
            # Never hand over more than the back layer can cover, or the fade
            # becomes a hole with nothing behind it.
            handover = np.clip(handover, 0.0, 1.0) * np.clip(
                back[..., 3].astype(np.float32) / 255.0, 0.0, 1.0)

        patched = np.array(front, copy=True)
        patched[..., 3] = np.rint(
            front[..., 3].astype(np.float32) * (1.0 - handover)).astype(np.uint8)
        out[front_tag] = patched
        moved[front_tag + "<-" + back_tag] = int(take.sum())

    return out, moved


def fit_edge_alpha(layer_dict: dict[str, np.ndarray], original_rgba: np.ndarray, *,
                   order: tuple[str, ...] = RIG_Z_ORDER,
                   alpha_threshold: int = 10, band: int = EDGE_BAND_PX,
                   outside: int = EDGE_OUTSIDE_PX,
                   min_contrast: int = EDGE_MIN_CONTRAST,
                   min_interior: float = EDGE_MIN_INTERIOR,
                   ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Refit each layer's edge alpha so the stack matches the original there.

    `reclaim_occluded` asks which of two layers owns a contested pixel. This
    asks a narrower question of every layer at its own boundary -- *how much* of
    it shows -- and the original answers it: the front layer composited over
    what is behind has to equal the picture, which is one linear equation per
    pixel in the coverage.

    The solve runs in both directions, which is why it replaces trimming. Where
    a layer's outermost rows are painted toward an outline the picture does not
    have, the coverage comes back near zero and the rim goes. Where a layer's
    alpha ramp is narrower than the line it draws -- the chin, where `head`
    fades over two rows while the original's contour fills both -- it comes back
    higher and the line is whole. Where two layers overlap and their alphas sum
    past one, both are pulled down.

    Only alpha changes, only within a few pixels of a boundary, and only where
    the front and what is behind differ enough for the answer to mean anything.
    A layer that is mostly boundary is skipped: it is an outline, and an outline
    is meant to be dark.
    """
    original = np.asarray(original_rgba)[..., :3].astype(np.float32)
    canvas_h, canvas_w = original.shape[:2]
    out = dict(layer_dict)
    moved: dict[str, int] = {}

    def rank(tag: str) -> int:
        return order.index(tag) if tag in order else -1

    beneath_rgb = np.zeros((canvas_h, canvas_w, 3), np.float32)
    beneath_a = np.zeros((canvas_h, canvas_w, 1), np.float32)

    for tag in sorted(out, key=rank):
        layer = np.asarray(out[tag])
        alpha = layer[..., 3].astype(np.float32) / 255.0
        solid = (layer[..., 3] > alpha_threshold).astype(np.uint8)
        if solid.any() and beneath_a.any():
            distance = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
            interior = float((distance > band).sum()) / max(float(solid.sum()), 1.0)
            if interior >= min_interior:
                edge = (solid > 0) & (distance <= band)
                if outside > 0:
                    grown = cv2.dilate(solid, np.ones((2 * outside + 1,) * 2, np.uint8))
                    edge |= (grown > 0) & (solid == 0)
                edge &= beneath_a[..., 0] > 0.5

                delta = layer[..., :3].astype(np.float32) - beneath_rgb
                denominator = (delta * delta).sum(axis=2)
                # Two layers the same colour: the coverage cannot be recovered
                # and does not matter, since either answer draws the same pixel.
                solvable = edge & (denominator > 3.0 * float(min_contrast) ** 2)
                if solvable.any():
                    fitted = np.clip(((original - beneath_rgb) * delta).sum(axis=2)
                                     / np.maximum(denominator, 1e-6), 0.0, 1.0)
                    updated = np.where(solvable, fitted, alpha)
                    changed = int((np.abs(updated - alpha) > 0.05).sum())
                    if changed:
                        moved[tag] = changed
                    layer = np.array(layer, copy=True)
                    layer[..., 3] = np.rint(updated * 255.0).astype(np.uint8)
                    out[tag] = layer

        a = np.clip(layer[..., 3:4].astype(np.float32) / 255.0, 0.0, 1.0)
        beneath_rgb = beneath_rgb * (1.0 - a) + layer[..., :3].astype(np.float32) * a
        beneath_a = np.clip(beneath_a + a, 0.0, 1.0)

    return out, moved


def composite_layers(layer_dict: dict[str, np.ndarray], frame_size: tuple[int, int], *,
                     order: tuple[str, ...] = RIG_Z_ORDER,
                     alpha_threshold: int = 10) -> np.ndarray:
    """Alpha-blend the layers back to front into one RGBA canvas.

    `layers.make_preview` also blends, but in dict insertion order and only for
    display. This composites in the canonical z-order and returns the array, so
    it can be compared against the original -- which is the only way to see an
    RGB loss. Every coverage metric in the report is computed on **alpha**:
    a layer that is present, correctly shaped, and the wrong colour scores
    exactly as well as a right one, and the `reconstruction` diagnostic cannot
    show the difference either because it copies the original's RGB and
    replaces only its alpha.
    """
    canvas_h, canvas_w = int(frame_size[0]), int(frame_size[1])
    rgb = np.zeros((canvas_h, canvas_w, 3), np.float32)
    acc = np.zeros((canvas_h, canvas_w, 1), np.float32)

    def rank(tag: str) -> int:
        return order.index(tag) if tag in order else -1

    for tag in sorted(layer_dict, key=rank):
        img = layer_dict.get(tag)
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] != 4 or not np.any(arr[..., 3] > alpha_threshold):
            continue
        src_a = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = arr[..., :3].astype(np.float32) * src_a + rgb * acc * (1.0 - src_a)
        acc = src_a + acc * (1.0 - src_a)
        rgb = rgb / np.maximum(acc, 1e-6)

    out = np.zeros((canvas_h, canvas_w, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    out[..., 3] = np.rint(np.clip(acc[..., 0] * 255.0, 0, 255)).astype(np.uint8)
    return out


def composite_fidelity(original_rgba: np.ndarray, composite: np.ndarray,
                       subject_mask: np.ndarray, *, bad_threshold: int = 30) -> dict[str, float]:
    """How closely a composite reproduces the original inside the subject.

    `mae` is the mean per-pixel sum of absolute RGB differences; `bad_ratio` is
    the share of subject pixels off by more than `bad_threshold`, which is
    where a wrong colour stops being a soft gradient difference and starts
    being a missing feature -- a dropped `eyewhite` leaves skin where the
    sclera was and lands around 60.
    """
    original = np.asarray(original_rgba)[..., :3].astype(np.int32)
    made = np.asarray(composite)[..., :3].astype(np.int32)
    mask = np.asarray(subject_mask)
    if mask.dtype != bool:
        mask = mask > (0.5 if mask.max() <= 1.0 else 127)
    total = int(mask.sum())
    if total == 0:
        return {"mae": 0.0, "bad_ratio": 0.0, "bad_px": 0, "subject_px": 0}
    diff = np.abs(original - made).sum(axis=2)[mask]
    bad = int((diff > bad_threshold).sum())
    return {
        "mae": round(float(diff.mean()), 3),
        "bad_ratio": round(bad / total, 5),
        "bad_px": bad,
        "subject_px": total,
    }


def _weight_for(tag: str, group: str, box: tuple[int, int, int, int],
                neck_box: tuple[int, int, int, int] | None,
                gradient_tags: Collection[str]) -> dict[str, Any]:
    """Head-follow weight for one part.

    The neck is the reason this is per-vertex at all: its top has to follow the
    head and its bottom has to stay with the body, which no arrangement of two
    rigid bones produces without a visible seam.
    """
    if tag in gradient_tags:
        # Explicit caller override, so it wins: the documented `back hair` risk,
        # where hair reaching past the shoulder line tears at full head weight.
        return {"mode": "gradient_y", "top": HEAD_WEIGHT, "bottom": BODY_WEIGHT,
                "y_top": float(box[1]), "y_bottom": float(box[3])}
    if group == GROUP_NECK and neck_box is not None:
        return {"mode": "gradient_y", "top": NECK_TOP_WEIGHT, "bottom": NECK_BOTTOM_WEIGHT,
                "y_top": float(neck_box[1]), "y_bottom": float(neck_box[3])}
    if tag in COLLAR_TAGS and neck_box is not None and box[1] < neck_box[3]:
        # The garment overlaps the neck, so its top edge is a collar: it shares
        # the neck's gradient exactly, which is what keeps the reclaimed window
        # and the neck showing through it moving together.
        return {"mode": "gradient_y", "top": NECK_TOP_WEIGHT, "bottom": NECK_BOTTOM_WEIGHT,
                "y_top": float(neck_box[1]), "y_bottom": float(neck_box[3])}
    if group == GROUP_HEAD:
        return {"mode": "constant", "value": HEAD_WEIGHT}
    if group == GROUP_NECK:
        return {"mode": "constant", "value": NECK_TOP_WEIGHT}
    return {"mode": "constant", "value": BODY_WEIGHT}


def _mesh_cell(frame_size: tuple[int, int], fine: bool) -> int:
    scale = max(int(frame_size[0]), int(frame_size[1])) / MESH_REFERENCE_SIZE
    base = MESH_CELL_FINE_PX if fine else MESH_CELL_PX
    return max(8, int(round(base * scale)))


def build_rig(layer_dict: dict[str, np.ndarray], *,
              original_rgba: np.ndarray | None = None,
              body_remainder: np.ndarray | None = None,
              depth_dict: dict[str, np.ndarray] | None = None,
              frame_size: tuple[int, int] | None = None,
              alpha_threshold: int = 10,
              gradient_tags: Collection[str] = (),
              run_id: str = "", tag_version: str = "",
              image_prefix: str = f"{RIG_SUBDIR}/images",
              motion: dict[str, Any] | None = None,
              ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Stages A-D: turn `{tag: full-canvas RGBA}` into `(manifest, images)`.

    `images` maps part name to the cropped RGBA the manifest references, so the
    caller decides where they land on disk (`write_rig_project` is the default
    answer). `depth_dict`, when given, overrides the tag depth table per layer
    with the median Marigold depth over that layer's visible pixels -- the same
    statistic `spine.layers_to_parts` uses, so a run can be compared against
    its own Spine export.

    `gradient_tags` forces a head-to-body vertical falloff onto tags that would
    otherwise follow the head rigidly; `("back hair",)` is the case the
    feasibility doc calls out.
    """
    working: dict[str, np.ndarray] = {}
    for tag, img in layer_dict.items():
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] != 4 or not np.any(arr[..., 3] > alpha_threshold):
            continue
        working[tag] = arr

    if frame_size is None:
        if not working:
            raise ValueError("frame_size is required when no layer has content")
        sample = next(iter(working.values()))
        frame_size = (int(sample.shape[0]), int(sample.shape[1]))
    canvas_h, canvas_w = int(frame_size[0]), int(frame_size[1])

    # Contested pixels first: a garment holding the skin from its own opening
    # hides the neck almost completely, and the remainder split below reads the
    # group unions this changes.
    reclaimed: dict[str, int] = {}
    edge_fit: dict[str, int] = {}
    if original_rgba is not None:
        working, reclaimed = reclaim_occluded(working, original_rgba,
                                              alpha_threshold=alpha_threshold)
        working, edge_fit = fit_edge_alpha(working, original_rgba,
                                           alpha_threshold=alpha_threshold)

    # Stage B: remainder next, so the eye split and the anchors below see the
    # same layer set the manifest will describe.
    depth_parent = {tag: tag for tag in working}
    if body_remainder is not None:
        for tag, img in split_remainder(body_remainder, working,
                                        alpha_threshold=alpha_threshold).items():
            working[tag] = img
            depth_parent[tag] = tag

    anchors = detect_anchors(working, (canvas_h, canvas_w), alpha_threshold=alpha_threshold)
    face_center = anchors.get("face_center")
    if face_center is not None:
        halves = split_eyes(working, face_center[0], alpha_threshold=alpha_threshold)
        for tag, img in halves.items():
            working[tag] = img
            depth_parent[tag] = tag[:-1]
        # Drop the undivided originals -- keeping both would double-draw the eyes.
        for parent in {tag[:-1] for tag in halves}:
            working.pop(parent, None)
            depth_parent.pop(parent, None)
        if halves:
            # The eye anchors only exist once the split has run.
            anchors = detect_anchors(working, (canvas_h, canvas_w),
                                     alpha_threshold=alpha_threshold)

    # Stage D.
    neck_box = neck_bbox(working, alpha_threshold=alpha_threshold)
    depths: dict[str, float] = {}
    for tag, arr in working.items():
        parent = depth_parent.get(tag, tag)
        depth = _DEPTH_TABLE.get(tag, _DEPTH_TABLE.get(parent, UNKNOWN_DEPTH))
        if depth_dict is not None and parent in depth_dict:
            visible = arr[..., 3] > alpha_threshold
            estimated = np.asarray(depth_dict[parent])
            if np.any(visible):
                depth = float(np.median(estimated[visible]))
        depths[tag] = round(float(depth), 4)

    ordered = sorted(
        working,
        key=lambda t: (-depths[t], RIG_Z_ORDER.index(t) if t in RIG_Z_ORDER else -1),
    )

    parts: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    for z, tag in enumerate(ordered):
        cropped = crop_to_alpha(working[tag], alpha_threshold)
        if cropped is None:
            continue
        crop_img, xyxy = cropped
        name = tag.replace(" ", "_")
        group = group_for_tag(tag)
        weight = _weight_for(tag, group, tuple(xyxy), neck_box, gradient_tags)
        images[name] = crop_img
        parts.append({
            "name": name,
            "tag": tag,
            "image": f"{image_prefix}/{name}.png" if image_prefix else f"{name}.png",
            "xyxy": [int(v) for v in xyxy],
            "group": group,
            "depth": depths[tag],
            "z": z,
            "weight": weight,
            # Anything deforming along a gradient gets the finer cell: that is
            # where a coarse grid shows up as faceting.
            "mesh": {"cell": _mesh_cell((canvas_h, canvas_w),
                                        fine=weight["mode"] == "gradient_y")},
        })

    manifest = {
        "version": MANIFEST_VERSION,
        "canvas": {"width": canvas_w, "height": canvas_h},
        "source": {
            "run_id": run_id,
            "tag_version": tag_version,
            "depth": "marigold" if depth_dict else "table",
            "reclaimed": reclaimed,
            "edge_fit": edge_fit,
        },
        "anchors": anchors,
        "parts": parts,
        "motion": json.loads(json.dumps(motion if motion is not None else DEFAULT_MOTION)),
    }
    return manifest, images


def write_rig_project(output_dir: str, base_name: str, manifest: dict[str, Any],
                      images: dict[str, np.ndarray], *, subdir: str = RIG_SUBDIR) -> str:
    """Write `{output_dir}/{base_name}_rig_manifest.json` plus the part PNGs
    under `{output_dir}/{subdir}/images/`. Returns the manifest path.

    The manifest sits at the run root beside `portrait_report.json` so that
    motion can be re-tuned by editing one file, without re-running the
    decomposition; the images it names are relative to that root, so zipping
    the run carries a self-contained rig.
    """
    images_dir = os.path.join(output_dir, subdir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for name, img in images.items():
        Image.fromarray(img).save(os.path.join(images_dir, f"{name}.png"))

    manifest_path = os.path.join(output_dir, f"{base_name}_rig_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def rebuild_run_rig(run_dir: str, *, gradient_tags: Collection[str] = ()) -> str:
    """Rebuild a finished run's rig from the layers it already wrote.

    Stages A-D read `{tag: full-canvas RGBA}` and nothing else, and every one of
    those layers is on disk beside the report. So a run made before a fix to the
    remainder split, the weights or `reclaim_occluded` can pick that fix up
    without the GPU pass that produced it -- which matters because the visible
    faults in a rig are usually found long after the run that made it.

    Overwrites `{base}_rig_manifest.json` and the part PNGs under `rig/images/`.
    An expression pack attached to the old manifest does not survive and has to
    be attached again; `expression.attach_to_run` and `transplant_to_run` both
    take a run directory, so that is one command.
    """
    names = [f for f in os.listdir(run_dir) if f.endswith("_manifest.json")
             and not f.endswith("_rig_manifest.json")]
    if not names:
        raise FileNotFoundError(f"no run manifest in {run_dir}")
    with open(os.path.join(run_dir, names[0]), encoding="utf-8") as f:
        run = json.load(f)
    base_name = run["base"]

    def read(filename):
        path = os.path.join(run_dir, filename)
        return np.array(Image.open(path).convert("RGBA")) if os.path.isfile(path) else None

    layer_dict = {tag: read(name) for tag, name in run["layers"].items()}
    layer_dict = {tag: img for tag, img in layer_dict.items() if img is not None}
    if not layer_dict:
        raise FileNotFoundError(f"none of the run's layer PNGs are in {run_dir}")
    original = read(run.get("original", f"{base_name}_original.png"))
    remainder = read(f"{base_name}_body_remainder.png")

    # `report` in the run manifest is the report's *filename*, not its contents.
    report = {}
    report_name = run.get("report")
    if isinstance(report_name, str) and os.path.isfile(os.path.join(run_dir, report_name)):
        with open(os.path.join(run_dir, report_name), encoding="utf-8") as f:
            report = json.load(f)
    elif isinstance(report_name, dict):
        report = report_name

    manifest, images = build_rig(
        layer_dict, original_rgba=original, body_remainder=remainder,
        frame_size=(run["height"], run["width"]),
        gradient_tags=gradient_tags,
        run_id=os.path.basename(os.path.normpath(run_dir)),
        tag_version=str(report.get("source", {}).get("tag_version", "")),
    )
    return write_rig_project(run_dir, base_name, manifest, images)


def main(argv=None) -> int:
    """`python -m seethrough_engine.rig <run dir>` -- rebuild a run's rig from
    its own layers, with the current code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(main.__doc__, file=sys.stderr)
        return 2
    print(rebuild_run_rig(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
