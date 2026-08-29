"""Expression pack extraction: recover only what a donor image changed.

M4.1 of `docs/PORTRAIT_MODE_FORK_PLAN_v0.1.md`. The rig can turn a head, blink
by squashing the lash, and flap a closed mouth, but it cannot produce a *shut*
eye or an *open* mouth: those are drawings A-001 does not contain. The
practical answer, read from PachiPakuGen, is to let an image model draw them --
and then to refuse to trust the result as a portrait.

A generated "same character with closed eyes" always drifts: the jawline moves
a little, the hair shifts, the colours are half a step off. Used as a portrait
that is an identity failure, which is where this fork's earlier modular attempt
died. Used as a **donor** it is fine, because only the eye region is taken from
it and everything else stays the original's own pixels.

Our order is the reverse of PachiPakuGen's, and cheaper, for a reason specific
to this fork: their donor is composited into a full frame which is then
decomposed again by the model, because they have no layers yet. We already have
layers, and Portrait Mode's `face` layer is *featureless skin* behind every
feature -- measured on A-001, luma standard deviation 0.5-0.8 under the eye and
mouth boxes against 25.5 over the face as a whole. So a recovered region can be
dropped straight in as one more part over clean skin: no second decomposition,
no GPU, and nothing else in the portrait can move.

numpy and cv2 only, like `rig.py`, so the whole path is testable on synthetic
donors without a model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "EYE_TAGS_L",
    "EYE_TAGS_R",
    "MOUTH_TAGS",
    "Roi",
    "ExpressionPart",
    "expression_rois",
    "drift_level",
    "infer_edit_mask",
    "match_to_base_boundary",
    "extract_part",
    "build_expression_pack",
    "write_expression_pack",
]

# Which layer boxes define each region's extent. A closed eye is drawn where
# the open one was, plus the lid and lashes around it, so the brow is included:
# it bounds the region an eye edit may legitimately reach, not what is taken.
EYE_TAGS_L = ("eyewhitel", "iridesl", "eyelashl", "eyebrowl", "eyel")
EYE_TAGS_R = ("eyewhiter", "iridesr", "eyelashr", "eyebrowr", "eyer")
MOUTH_TAGS = ("mouth",)

# How far past those boxes an edit may reach, as a fraction of the box's own
# size. An open mouth drops the jaw well below the closed mouth's box, which is
# why its vertical margin is the largest number here.
EYE_MARGIN = (0.45, 0.90)
MOUTH_MARGIN = (0.50, 1.20)

# Hysteresis, the same shape as `rig.reclaim_occluded`: a strong core decides
# *which* regions are real, and a weaker rule decides how far each one extends.
# A single threshold cannot do both jobs -- generative drift covers the whole
# picture weakly, so anything low enough to follow an eyelid to its edge is also
# low enough to seed a region out of the drift on the cheek next to it.
CORE_DIFF = 64        # a pixel this different is an edit, not drift
EXTENT_FLOOR = 14     # ... and this is as low as the extent rule may go
EXTENT_PERCENTILE = 87
# The percentile is a fraction of the *region*, so how high it lands depends on
# how much of the region the edit happens to fill. An open mouth fills most of
# its own region and pushes the 87th percentile to 187 -- above the core -- at
# which point the extent rule is stricter than the seed and the recovered shape
# comes from the dilation rather than from the picture. The extent rule has to
# stay the weaker of the two by construction.
EXTENT_CAP_RATIO = 0.5

# Drift is measured rather than assumed, on the part of the picture we are not
# taking anything from -- the torso, the hair, the far cheek. A fixed floor is
# not enough on its own: a donor that shifts the whole image by 25 puts every
# pixel of the region over a floor of 14, the region becomes one connected
# component, the core seeds it, and the "mask" is the whole rectangle. So the
# extent rule is held above whatever drift the donor actually has.
#
# A high percentile rather than a maximum, so the estimate is a ceiling on the
# generator's noise and not a report of its worst pixel -- and low enough that a
# real edit somewhere we are not looking (a donor that also redrew a hand)
# cannot poison it and quietly raise the bar until nothing is recovered at all.
DRIFT_PERCENTILE = 90
DRIFT_MARGIN = 4

# Morphology, from PachiPakuGen's values, as starting points.
DILATE_ITERATIONS = {"eye": 3, "mouth": 4}
FEATHER = {"eye": 7, "mouth": 9}
MAX_COMPONENTS = {"eye": 6, "mouth": 3}
MIN_COMPONENT_AREA = 12

# How much of the boundary colour difference to take out of the donor. Not all
# of it: the ring is a mix of both sides, so correcting fully overshoots.
BOUNDARY_CORRECTION = 0.55

# Below this many pixels a recovered region is noise, not a drawing.
MIN_PART_AREA = 40


@dataclass(frozen=True)
class Roi:
    """Where one expression part may live, in canvas pixels."""

    name: str
    kind: str                       # "eye" | "mouth"
    side: str | None                # "l" | "r" | None
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    anchor: tuple[float, float]

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=np.uint8)
        x1, y1, x2, y2 = self.box
        out[y1:y2, x1:x2] = 255
        return out


@dataclass
class ExpressionPart:
    """A region recovered from a donor, ready to be drawn as one more layer."""

    name: str
    kind: str
    side: str | None
    image: np.ndarray                # cropped RGBA
    xyxy: tuple[int, int, int, int]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(x1 + 1, min(width, round(x2))))
    y2 = int(max(y1 + 1, min(height, round(y2))))
    return x1, y1, x2, y2


def _union_box(part_boxes, tags):
    boxes = [part_boxes[t] for t in tags if t in part_boxes]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _grow(box, margin):
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * margin[0]
    my = (y2 - y1) * margin[1]
    return (x1 - mx, y1 - my, x2 + mx, y2 + my)


def expression_rois(anchors, part_boxes, canvas) -> list[Roi]:
    """The regions a donor is allowed to change, derived from this run's own
    anchors and layer boxes.

    PachiPakuGen fixes these as fractions of the canvas -- the eye band at 18%
    to 46% of the height -- which is right for a full-body standing picture and
    wrong for a bust, where the face fills most of the frame. We do not have to
    guess: the decomposition already told us where the eyes and the mouth are.
    """
    width, height = canvas
    rois: list[Roi] = []
    eye_l = _union_box(part_boxes, EYE_TAGS_L)
    eye_r = _union_box(part_boxes, EYE_TAGS_R)
    for side, box, anchor_key in (("l", eye_l, "eye_left"), ("r", eye_r, "eye_right")):
        if box is None:
            continue
        anchor = anchors.get(anchor_key)
        if anchor is None:
            anchor = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        rois.append(Roi(f"eye_{side}", "eye", side,
                        _clip_box(_grow(box, EYE_MARGIN), width, height),
                        (float(anchor[0]), float(anchor[1]))))
    # Keep the two eyes' regions apart, so an edit to one can never be claimed
    # by the other. The split is the midpoint between the anchors, not the face
    # centre: a three-quarter view puts the nose off centre.
    if len(rois) == 2:
        mid = (rois[0].anchor[0] + rois[1].anchor[0]) / 2
        left, right = sorted(rois, key=lambda r: r.anchor[0])
        rois = [
            Roi(left.name, left.kind, left.side,
                (left.box[0], left.box[1], min(left.box[2], int(mid)), left.box[3]), left.anchor),
            Roi(right.name, right.kind, right.side,
                (max(right.box[0], int(mid)), right.box[1], right.box[2], right.box[3]), right.anchor),
        ]

    mouth = _union_box(part_boxes, MOUTH_TAGS)
    if mouth is not None:
        anchor = anchors.get("mouth") or ((mouth[0] + mouth[2]) / 2, (mouth[1] + mouth[3]) / 2)
        rois.append(Roi("mouth", "mouth", None,
                        _clip_box(_grow(mouth, MOUTH_MARGIN), width, height),
                        (float(anchor[0]), float(anchor[1]))))
    return rois


def _resize_like(image, base):
    if image.shape[:2] == base.shape[:2]:
        return image
    bh, bw = base.shape[:2]
    ih, iw = image.shape[:2]
    if abs((iw / ih) - (bw / bh)) > 0.02 * (bw / bh):
        raise ValueError(
            f"donor is {iw}x{ih} against a {bw}x{bh} base and the aspect ratios "
            "differ by more than 2%: it is a different framing, not a different "
            "expression")
    return cv2.resize(image, (bw, bh), interpolation=cv2.INTER_AREA)


def _keep_seeded_components(extent, core, anchor, max_components):
    """Components of `extent` that contain a `core` pixel, nearest the anchor
    first. The core decides what is real; the extent decides where it ends."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(extent, 8)
    if count <= 1:
        return np.zeros_like(extent), 0
    seeded = set(np.unique(labels[core > 0])) - {0}
    candidates = []
    for label in range(1, count):
        if label not in seeded:
            continue
        if stats[label, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue
        cx, cy = centroids[label]
        distance = float(np.hypot(cx - anchor[0], cy - anchor[1]))
        candidates.append((distance, -int(stats[label, cv2.CC_STAT_AREA]), label))
    if not candidates:
        return np.zeros_like(extent), 0
    candidates.sort()
    keep = [label for _, _, label in candidates[:max_components]]
    return (np.isin(labels, keep).astype(np.uint8) * 255), len(keep)


def drift_level(base, donor, rois) -> int:
    """How much the donor moved everywhere it was not asked to.

    Measured on the subject outside every region, which is exactly the part of
    the picture a donor is supposed to leave alone, so whatever difference is
    found there is the generator's noise floor for this image.
    """
    donor = _resize_like(donor, base)
    outside = base[:, :, 3] > 128
    for roi in rois:
        x1, y1, x2, y2 = roi.box
        outside[y1:y2, x1:x2] = False
    if int(outside.sum()) < 64:
        return 0
    diff = np.max(np.abs(base[:, :, :3].astype(np.int16)
                         - donor[:, :, :3].astype(np.int16)), axis=2)
    return int(np.percentile(diff[outside], DRIFT_PERCENTILE))


def infer_edit_mask(base, donor, roi: Roi, drift: int = 0):
    """Where the donor differs from the base, inside one region.

    Returns a feathered uint8 mask over the whole canvas and a diagnostics dict.
    An empty mask is a legitimate answer and means the donor did not change this
    region -- the caller must not treat it as a failure.
    """
    diff = np.max(np.abs(base[:, :, :3].astype(np.int16)
                         - donor[:, :, :3].astype(np.int16)), axis=2).astype(np.uint8)
    region = roi.mask(base.shape[:2]) > 0
    values = diff[region]
    if values.size == 0:
        return np.zeros(base.shape[:2], dtype=np.uint8), {"reason": "empty roi"}

    core_threshold = max(CORE_DIFF, 2 * drift)
    core = ((diff >= core_threshold) & region).astype(np.uint8) * 255
    core_area = int((core > 0).sum())
    diagnostics = {
        "roi": list(roi.box),
        "drift": int(drift),
        "core_threshold": int(core_threshold),
        "core_area": core_area,
        "max_diff": int(values.max()),
        "extent_threshold": None,
        "components": 0,
        "mask_area": 0,
    }
    if core_area < MIN_COMPONENT_AREA:
        # Nothing in this region is different enough to be a drawing. The donor
        # left it alone, or changed it only as much as it changed everything.
        diagnostics["reason"] = "no core"
        return np.zeros(base.shape[:2], dtype=np.uint8), diagnostics

    threshold = max(EXTENT_FLOOR, drift + DRIFT_MARGIN,
                    min(int(np.percentile(values, EXTENT_PERCENTILE)),
                        int(core_threshold * EXTENT_CAP_RATIO)))
    diagnostics["extent_threshold"] = threshold
    extent = ((diff >= threshold) & region).astype(np.uint8) * 255
    extent = cv2.medianBlur(extent, 3)
    # medianBlur can erase a thin core, so put it back before seeding.
    extent = np.maximum(extent, core)
    mask, kept = _keep_seeded_components(extent, core, roi.anchor, MAX_COMPONENTS[roi.kind])
    diagnostics["components"] = kept
    if kept == 0:
        diagnostics["reason"] = "no seeded component"
        return np.zeros(base.shape[:2], dtype=np.uint8), diagnostics

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERATIONS[roi.kind])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (FEATHER[roi.kind], FEATHER[roi.kind]), 0)
    # Dilation and blur both run past the region; the region is the contract.
    mask = np.where(region, mask, 0).astype(np.uint8)
    diagnostics["mask_area"] = int((mask > 0).sum())
    return mask, diagnostics


def match_to_base_boundary(base, donor, mask):
    """Pull the donor's colour toward the base's, measured on the ring just
    outside the mask. A generated image is usually a half-step off in exposure,
    and a seam is visible long before the colour difference is."""
    hard = (mask > 16).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ring = (cv2.dilate(hard, kernel, iterations=2) > 0) & (cv2.erode(hard, kernel, iterations=1) == 0)
    # Only where both pictures actually have the subject: outside the silhouette
    # the "colour" is whatever the background was.
    ring &= base[:, :, 3] > 128
    if int(ring.sum()) < 20:
        return donor, [0.0, 0.0, 0.0]
    delta = np.median(base[:, :, :3].astype(np.int16)[ring]
                      - donor[:, :, :3].astype(np.int16)[ring], axis=0)
    if not np.all(np.isfinite(delta)):
        return donor, [0.0, 0.0, 0.0]
    out = donor.astype(np.int16).copy()
    out[:, :, :3] = np.clip(out[:, :, :3] + delta * BOUNDARY_CORRECTION, 0, 255)
    return out.astype(np.uint8), [round(float(d), 2) for d in delta]


def extract_part(base, donor, roi: Roi, name: str, drift: int = 0) -> ExpressionPart | None:
    """One region of a donor, as a cropped RGBA sprite, or None if the donor
    did not change it."""
    donor = _resize_like(donor, base)
    mask, diagnostics = infer_edit_mask(base, donor, roi, drift)
    if not mask.any():
        return None
    adjusted, delta = match_to_base_boundary(base, donor, mask)
    diagnostics["boundary_delta"] = delta

    # Clipped to the base's own silhouette: a donor that grew the character an
    # extra pixel of hair must not add it here.
    alpha = (mask.astype(np.float32) / 255.0) * (base[:, :, 3].astype(np.float32) / 255.0)
    ys, xs = np.nonzero(alpha > 0.004)
    if xs.size == 0 or float(alpha.sum()) < MIN_PART_AREA / 255.0:
        diagnostics["reason"] = "region too small once clipped to the silhouette"
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    sprite = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    sprite[:, :, :3] = adjusted[y1:y2, x1:x2, :3]
    sprite[:, :, 3] = np.round(alpha[y1:y2, x1:x2] * 255).astype(np.uint8)
    diagnostics["sprite_area"] = int((sprite[:, :, 3] > 0).sum())
    return ExpressionPart(name=name, kind=roi.kind, side=roi.side, image=sprite,
                          xyxy=(x1, y1, x2, y2), diagnostics=diagnostics)


def build_expression_pack(base, donors, anchors, part_boxes) -> dict[str, Any]:
    """Recover every region each donor changed.

    `donors` maps a state name to an image: `{"eye_closed": rgba,
    "mouth_open": rgba}`. A donor is searched only in the regions its kind owns,
    so a mouth donor whose eyes drifted cannot rewrite the eyes.
    """
    canvas = (base.shape[1], base.shape[0])
    rois = expression_rois(anchors, part_boxes, canvas)
    parts: list[ExpressionPart] = []
    report: dict[str, Any] = {"canvas": list(canvas), "states": {}}
    for state, donor in donors.items():
        kind = "mouth" if state.startswith("mouth") else "eye"
        drift = drift_level(base, donor, rois)
        entries = {}
        for roi in rois:
            if roi.kind != kind:
                continue
            name = f"{state}_{roi.side}" if roi.side else state
            part = extract_part(base, donor, roi, name, drift)
            entries[roi.name] = part.diagnostics if part else {"recovered": False, "drift": drift}
            if part is not None:
                parts.append(part)
        report["states"][state] = entries
    return {"parts": parts, "report": report}


def write_expression_pack(out_dir, pack, subdir="rig/images") -> dict[str, Any]:
    """Write each part's PNG and return the manifest block naming them."""
    images_dir = os.path.join(out_dir, *subdir.split("/"))
    os.makedirs(images_dir, exist_ok=True)
    entries: dict[str, Any] = {}
    for part in pack["parts"]:
        rel = f"{subdir}/{part.name}.png"
        Image.fromarray(part.image, mode="RGBA").save(os.path.join(out_dir, *rel.split("/")))
        entries[part.name] = {
            "image": rel,
            "kind": part.kind,
            "side": part.side,
            "xyxy": list(part.xyxy),
        }
    return {"version": "0.1", "parts": entries, "report": pack["report"]}
