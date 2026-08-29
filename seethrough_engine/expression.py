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

import json
import os
import sys
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
    "register_donor",
    "claimable_mask",
    "drift_level",
    "infer_edit_mask",
    "match_to_base_boundary",
    "extract_part",
    "build_expression_pack",
    "write_expression_pack",
    "attach_to_run",
    "align_runs",
    "transplant_pack",
    "SKIN_TRIM",
    "transplant_to_run",
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

# What a facial edit is allowed to repaint: the face's own skin and features.
# A donor is a different generation, so its hair is drawn differently too, and
# that difference sits inside the eye region and connects to the eye through the
# brow. Left alone it is invisible at rest and wrong in motion -- the swallowed
# hair would travel at the eye's depth on the head shell while the real `front
# hair` travels on the hair shell, and the two would slide apart exactly when
# the head turns. Nothing here has to guess where the hair is: the decomposition
# already said so, which is the one thing a PSD-less pipeline cannot do.
CLAIMABLE_TAGS = frozenset({
    "face", "head", "nose", "mouth",
    "eyewhite", "eyewhitel", "eyewhiter",
    "irides", "iridesl", "iridesr",
    "eyelash", "eyelashl", "eyelashr",
    "eyebrow", "eyebrowl", "eyebrowr",
    "eyes", "eyel", "eyer",
})


def claimable_mask(rig_parts, images, shape) -> np.ndarray:
    """Where an expression may repaint, as a boolean canvas: the pixels whose
    topmost layer at rest is one of `CLAIMABLE_TAGS`.

    Topmost, not "any layer covers it": the hair in front of a cheek owns that
    pixel, and repainting it would put a second copy of the hair under the real
    one."""
    owner = np.full(shape, -1, dtype=np.int16)
    parts = sorted(rig_parts, key=lambda p: p["z"])
    for index, part in enumerate(parts):
        image = images.get(part["name"])
        if image is None:
            continue
        x1, y1, x2, y2 = part["xyxy"]
        window = owner[y1:y2, x1:x2]
        window[image[:, :, 3] > 128] = index
    claimable = np.zeros(shape, dtype=bool)
    for index, part in enumerate(parts):
        if part["tag"] in CLAIMABLE_TAGS:
            claimable |= owner == index
    return claimable


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


def _silhouette_box(alpha):
    ys, xs = np.nonzero(alpha > 128)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def register_donor(base, donor):
    """Bring a donor into the base's frame.

    A generated donor is rarely the same crop as the picture that was
    decomposed -- the model returns whatever framing it likes -- and every
    threshold here compares the two pixel for pixel. Since both are the same
    drawing, one uniform scale and one offset are enough, and the silhouette
    gives them: match the subject's bounding box.

    A donor that is already the base's size is left alone. Fitting a box to an
    image that is already registered can only move it, and the expression
    itself (a lock of hair drawn one pixel differently) is enough to move the
    box.
    """
    if donor.shape[:2] == base.shape[:2]:
        return donor
    if donor.shape[2] == 3 or int((donor[:, :, 3] > 128).all()):
        # No matte of its own: the background is whatever fills its border.
        from .matting import key_flat_background
        donor, _ = key_flat_background(donor[:, :, :3])
    src = _silhouette_box(donor[:, :, 3])
    dst = _silhouette_box(base[:, :, 3])
    if src is None or dst is None:
        raise ValueError("cannot register a donor without a subject silhouette")
    scale = (dst[2] - dst[0]) / (src[2] - src[0])
    warp = np.array([[scale, 0, dst[0] - src[0] * scale],
                     [0, scale, dst[1] - src[1] * scale]], dtype=np.float32)
    return cv2.warpAffine(donor, warp, (base.shape[1], base.shape[0]),
                          flags=cv2.INTER_AREA, borderValue=(0, 0, 0, 0))


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
    donor = register_donor(base, donor)
    outside = base[:, :, 3] > 128
    for roi in rois:
        x1, y1, x2, y2 = roi.box
        outside[y1:y2, x1:x2] = False
    if int(outside.sum()) < 64:
        return 0
    diff = np.max(np.abs(base[:, :, :3].astype(np.int16)
                         - donor[:, :, :3].astype(np.int16)), axis=2)
    return int(np.percentile(diff[outside], DRIFT_PERCENTILE))


def infer_edit_mask(base, donor, roi: Roi, drift: int = 0, claimable=None):
    """Where the donor differs from the base, inside one region.

    Returns a feathered uint8 mask over the whole canvas and a diagnostics dict.
    An empty mask is a legitimate answer and means the donor did not change this
    region -- the caller must not treat it as a failure.
    """
    diff = np.max(np.abs(base[:, :, :3].astype(np.int16)
                         - donor[:, :, :3].astype(np.int16)), axis=2).astype(np.uint8)
    region = roi.mask(base.shape[:2]) > 0
    if claimable is not None:
        region = region & claimable
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


def extract_part(base, donor, roi: Roi, name: str, drift: int = 0,
                 claimable=None) -> ExpressionPart | None:
    """One region of a donor, as a cropped RGBA sprite, or None if the donor
    did not change it."""
    donor = register_donor(base, donor)
    mask, diagnostics = infer_edit_mask(base, donor, roi, drift, claimable)
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


def build_expression_pack(base, donors, anchors, part_boxes, claimable=None) -> dict[str, Any]:
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
            part = extract_part(base, donor, roi, name, drift, claimable)
            entries[roi.name] = part.diagnostics if part else {"recovered": False, "drift": drift}
            if part is not None:
                parts.append(part)
        report["states"][state] = entries
    return {"parts": parts, "report": report}


# Which rig parts a recovered sprite stands in for while it is showing. The
# runtime fades these out rather than deleting them: without an expression pack
# they are still the whole eye, and the lash squash is still the blink.
REPLACED_TAGS = {
    ("eye", "l"): ("eyewhitel", "iridesl", "eyelashl", "eyel"),
    ("eye", "r"): ("eyewhiter", "iridesr", "eyelashr", "eyer"),
    ("mouth", None): ("mouth",),
}


def _placement(part, rig_parts):
    """Where a recovered sprite is drawn: over the parts it replaces, at the
    depth of the nearest of them, so it parallaxes with the face it belongs to
    instead of with whatever the table would have given a new tag."""
    # Two different questions. What the sprite is *made of* is what the donor
    # had -- a closed eye is a lid and a brow, and the donor run has no
    # `eyewhite` at all, because there is no white to see. What has to *stop
    # drawing* is the base's whole feature: the donor lacking an eyewhite is
    # precisely why the base's must be hidden, or the open eye's sclera stays
    # visible around the shut lid. So the table is the floor, and anything extra
    # the sprite carries is added to it.
    tags = set(REPLACED_TAGS.get((part.kind, part.side), ()))
    tags |= set(part.diagnostics.get("replaces_tags", ()))
    replaced = [p for p in rig_parts if p.get("tag") in tags]
    if not replaced:
        return {"replaces": [], "z": None, "depth": None}
    return {
        "replaces": [p["name"] for p in replaced],
        "z": max(float(p["z"]) for p in replaced) + 0.5,
        "depth": min(float(p["depth"]) for p in replaced),
    }


def write_expression_pack(out_dir, pack, rig_parts=(), subdir="rig/images") -> dict[str, Any]:
    """Write each part's PNG and return the manifest block naming them.

    `rig_parts` is the rig manifest's own `parts` list. Given it, each entry
    also carries where the sprite is drawn and what it stands in for; without
    it the block is still valid and the runtime has to place the sprite itself.
    """
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
            **_placement(part, rig_parts),
        }
    return {"version": "0.1", "parts": entries, "report": pack["report"]}


# What each expression state is made of when the donor has been decomposed in
# its own right. A closed eye is drawn as a lid and lashes, so `eyelash` is
# usually the whole of it; the brow comes along because a shut eye's brow is
# drawn differently, and taking it means the base's brow has to hand over too.
TRANSPLANT_SOURCES = {
    ("eye", "l"): ("eyewhitel", "iridesl", "eyelashl", "eyebrowl"),
    ("eye", "r"): ("eyewhiter", "iridesr", "eyelashr", "eyebrowr"),
    ("mouth", None): ("mouth",),
}

# A donor layer carries a margin of the skin it was drawn on -- more than half
# of A-001's donor `mouth` is within 20 of the `face` layer underneath it, and a
# third within 8 -- plus a darkened ring two pixels wide where its own alpha
# ends. Kept, they lay one generation's skin over another's and draw a faint
# ellipse right around the lips, which no tone matching removes because the two
# skins are shaded differently. So a layer is faded out where it is
# indistinguishable from the skin beneath it.
#
# The two populations are far apart and the threshold sits in the gap between
# them: measured outward from A-001's donor mouth, the margin differs by 7, its
# edge ring by 19, and the lips by 54 to 94. Choosing near the margin instead
# put the threshold inside the noise, and a per-pixel verdict there is speckle.
SKIN_TRIM = (25.0, 45.0)

# This is a no-op on the features that are all drawing: no `eyelash` or
# `eyebrow` pixel in A-001's donor is within 144 of the face beneath it.

# The layer whose shape does not depend on the expression, and so is what the
# two runs are aligned by. `face` is inpainted skin with no features drawn on
# it -- the same silhouette whether the eyes are open or shut -- while every
# anchor a rig has (eye centres, mouth centre) is exactly what the donor
# changed. Aligning by the thing that moved is how the parts end up misplaced.
ALIGN_TAGS = ("face", "head")


def _run_manifest(run_dir):
    names = [f for f in os.listdir(run_dir) if f.endswith("_rig_manifest.json")]
    if not names:
        raise FileNotFoundError(f"no *_rig_manifest.json in {run_dir}")
    path = os.path.join(run_dir, names[0])
    with open(path, encoding="utf-8") as f:
        return path, names[0][: -len("_rig_manifest.json")], json.load(f)


def _trim_to_drawing(layer, skin):
    """Fade `layer` out wherever it only repeats the skin it was drawn on.

    Where the skin layer has nothing to compare against, the layer is left
    alone: an absent judgement is not a judgement that it is skin.
    """
    if skin is None:
        return layer
    # Weighted by coverage, because the question is how much this layer changes
    # the picture, not how different its stored colour is. A boundary pixel at
    # alpha 10 contributes a twenty-fifth of whatever its colour says, and the
    # colour under a nearly transparent pixel is arbitrary -- comparing it
    # straight is what kept the layer's own anti-aliased rim alive as a faint
    # ellipse around the mouth.
    difference = (np.max(np.abs(layer[:, :, :3] - skin[:, :, :3]), axis=2)
                  * (layer[:, :, 3] / 255.0))
    lo, hi = SKIN_TRIM
    # A region, not a per-pixel verdict. Around the drawing the difference
    # hovers near the threshold, so deciding pixel by pixel gives speckle, and
    # softening speckle gives a faint coherent ring -- which is worse than
    # either answer, because a fifteen per cent wash of another generation's
    # skin cannot add anything and can only tint. Same rule as
    # `rig.reclaim_occluded`: a core that is unmistakably drawing seeds the
    # region, and the weaker rule decides how far it reaches.
    unjudgeable = skin[:, :, 3] <= 128
    core = ((difference >= hi) | unjudgeable) & (layer[:, :, 3] > 0)
    extent = ((difference >= lo) | unjudgeable) & (layer[:, :, 3] > 0)
    keep = _seeded_regions(extent.astype(np.uint8), core.astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=2)
    keep = cv2.GaussianBlur(keep.astype(np.float32), (3, 3), 0)
    out = layer.copy()
    out[:, :, 3] *= keep
    return out


def _seeded_regions(extent, core):
    """Every connected part of `extent` that contains a `core` pixel, as 0/1."""
    count, labels = cv2.connectedComponents(extent, 8)
    if count <= 1:
        return np.zeros(extent.shape, dtype=np.float32)
    seeded = set(np.unique(labels[core > 0])) - {0}
    if not seeded:
        return np.zeros(extent.shape, dtype=np.float32)
    return np.isin(labels, list(seeded)).astype(np.float32)


def _run_original(run_dir, base_name):
    path = os.path.join(run_dir, f"{base_name}_original.png")
    if not os.path.isfile(path):
        return None
    return np.array(Image.open(path).convert("RGBA"))


def _load_layer(run_dir, manifest, tag):
    for part in manifest["parts"]:
        if part["tag"] == tag:
            path = os.path.join(run_dir, *part["image"].split("/"))
            if os.path.isfile(path):
                return np.array(Image.open(path).convert("RGBA")), tuple(part["xyxy"])
    return None, None


def align_runs(donor_manifest, donor_dir, base_manifest, base_dir):
    """The similarity that puts the donor run's canvas onto the base run's.

    Fitted to the alignment layer's bounding box: one uniform scale and one
    offset, which is all two renders of the same drawing differ by.
    """
    for tag in ALIGN_TAGS:
        src_img, src_box = _load_layer(donor_dir, donor_manifest, tag)
        dst_img, dst_box = _load_layer(base_dir, base_manifest, tag)
        if src_box is None or dst_box is None:
            continue
        scale = (dst_box[2] - dst_box[0]) / (src_box[2] - src_box[0])
        return np.array([[scale, 0, dst_box[0] - src_box[0] * scale],
                         [0, scale, dst_box[1] - src_box[1] * scale]],
                        dtype=np.float32), tag, scale
    raise ValueError(
        "neither run has a layer to align by; expected one of " + ", ".join(ALIGN_TAGS))


def transplant_pack(base_dir, donor_dir, states) -> dict[str, Any]:
    """Take the donor run's *own layers* for the features that changed.

    The alternative to diffing, and better where a second decomposition is
    affordable: a decomposition emits a layer with the matte the model drew,
    while a diff emits a region with a matte inferred from a threshold, grown
    by a dilation and softened by a blur. The inferred one is visible as a ring
    of donor skin around the feature and as a straight cut wherever the region's
    rectangle crossed the drawing.

    `states` maps a state name to nothing but its kind, as elsewhere:
    `("eye_closed", "mouth_open")`.
    """
    _, base_name, base_manifest = _run_manifest(base_dir)
    _, donor_name, donor_manifest = _run_manifest(donor_dir)
    canvas = (base_manifest["canvas"]["width"], base_manifest["canvas"]["height"])
    warp, align_tag, scale = align_runs(donor_manifest, donor_dir, base_manifest, base_dir)

    # The two originals, in one frame, only so the boundary colour can be
    # matched: a layer carries a margin of its own skin around the feature, and
    # two generations are never quite the same tone, which draws a faint contour
    # exactly where the sprite ends.
    base_original = _run_original(base_dir, base_name)
    donor_original = _run_original(donor_dir, donor_name)
    if donor_original is not None:
        donor_original = cv2.warpAffine(donor_original, warp, canvas, flags=cv2.INTER_AREA,
                                        borderValue=(0, 0, 0, 0))

    skin_image, skin_box = _load_layer(donor_dir, donor_manifest, "face")
    skin = None
    if skin_image is not None:
        skin = np.zeros((donor_manifest["canvas"]["height"],
                         donor_manifest["canvas"]["width"], 4), dtype=np.float32)
        skin[skin_box[1]:skin_box[3], skin_box[0]:skin_box[2]] = skin_image

    base_tags = {p["tag"] for p in base_manifest["parts"]}
    parts: list[ExpressionPart] = []
    report: dict[str, Any] = {"canvas": list(canvas), "source": "transplant",
                              "aligned_by": align_tag, "scale": round(float(scale), 4),
                              "states": {}}
    for state in states:
        kind = "mouth" if state.startswith("mouth") else "eye"
        entries: dict[str, Any] = {}
        for (source_kind, side), tags in TRANSPLANT_SOURCES.items():
            if source_kind != kind:
                continue
            # Every layer of the feature, drawn back to front onto one canvas:
            # the runtime hands a feature to one part, not to four.
            #
            # In premultiplied colour throughout, and un-premultiplied only at
            # the end. Compositing straight colour into a buffer that starts at
            # zero multiplies every edge pixel by its own alpha and leaves it
            # there, which paints a dark ring one pixel wide around anything
            # with a soft edge -- and a mouth is nothing but soft edge. The same
            # reason `matting` un-premultiplies rather than thresholding.
            merged = np.zeros((canvas[1], canvas[0], 4), dtype=np.float32)
            taken = []
            for tag in tags:
                image, box = _load_layer(donor_dir, donor_manifest, tag)
                if image is None:
                    continue
                full = np.zeros((donor_manifest["canvas"]["height"],
                                 donor_manifest["canvas"]["width"], 4), dtype=np.float32)
                full[box[1]:box[3], box[0]:box[2]] = image
                full = _trim_to_drawing(full, skin)
                # Premultiply before the resample too: interpolating colour and
                # alpha independently mixes in whatever is outside the layer.
                full[:, :, :3] *= full[:, :, 3:4] / 255.0
                placed = cv2.warpAffine(full, warp, canvas, flags=cv2.INTER_AREA,
                                        borderValue=(0, 0, 0, 0))
                keep = 1.0 - merged[:, :, 3:4] / 255.0
                merged[:, :, :3] += placed[:, :, :3] * keep
                merged[:, :, 3:4] += placed[:, :, 3:4] * keep
                taken.append(tag)
            alpha_f = merged[:, :, 3:4] / 255.0
            merged[:, :, :3] = np.divide(merged[:, :, :3], alpha_f,
                                         out=np.zeros_like(merged[:, :, :3]),
                                         where=alpha_f > 1e-4)
            name = f"{state}_{side}" if side else state
            alpha = merged[:, :, 3]
            if not taken or float(alpha.sum()) < MIN_PART_AREA * 255.0:
                entries[name] = {"recovered": False, "taken": taken}
                continue
            delta = [0.0, 0.0, 0.0]
            if base_original is not None and donor_original is not None:
                _, delta = match_to_base_boundary(base_original, donor_original,
                                                  alpha.astype(np.uint8))
                merged[:, :, :3] = np.clip(
                    merged[:, :, :3] + np.array(delta) * BOUNDARY_CORRECTION, 0, 255)
            ys, xs = np.nonzero(alpha > 2)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            sprite = np.clip(merged[y1:y2, x1:x2], 0, 255).astype(np.uint8)
            parts.append(ExpressionPart(
                name=name, kind=kind, side=side, image=sprite, xyxy=(x1, y1, x2, y2),
                diagnostics={"source": "transplant", "taken": taken,
                             "boundary_delta": delta,
                             "replaces_tags": [t for t in taken if t in base_tags],
                             "sprite_area": int((sprite[:, :, 3] > 0).sum()),
                             "scale": round(float(scale), 4)}))
            entries[name] = parts[-1].diagnostics
        report["states"][state] = entries
    return {"parts": parts, "report": report}


def transplant_to_run(base_dir, donor_dir, states) -> dict[str, Any]:
    """`transplant_pack`, written into the base run's manifest."""
    manifest_path, _, manifest = _run_manifest(base_dir)
    pack = transplant_pack(base_dir, donor_dir, states)
    manifest["expressions"] = write_expression_pack(base_dir, pack, manifest["parts"])
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest["expressions"]


def attach_to_run(run_dir, donors) -> dict[str, Any]:
    """Recover `donors` against a finished run and add them to its rig manifest.

    Deliberately a separate pass over a run directory rather than a step inside
    `save_portrait_run`: the donors are drawn *after* looking at the
    decomposition, and re-running the model to attach them would be absurd.
    """
    names = [f for f in os.listdir(run_dir) if f.endswith("_rig_manifest.json")]
    if not names:
        raise FileNotFoundError(f"no *_rig_manifest.json in {run_dir}")
    manifest_path = os.path.join(run_dir, names[0])
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    base_name = names[0][: -len("_rig_manifest.json")]
    base = np.array(Image.open(os.path.join(run_dir, f"{base_name}_original.png")).convert("RGBA"))
    boxes = {p["tag"]: tuple(p["xyxy"]) for p in manifest["parts"]}
    anchors = {k: tuple(v) for k, v in manifest.get("anchors", {}).items()}

    images = {}
    for part in manifest["parts"]:
        path = os.path.join(run_dir, *part["image"].split("/"))
        if os.path.isfile(path):
            images[part["name"]] = np.array(Image.open(path).convert("RGBA"))
    claimable = claimable_mask(manifest["parts"], images, base.shape[:2])

    pack = build_expression_pack(base, donors, anchors, boxes, claimable)
    manifest["expressions"] = write_expression_pack(run_dir, pack, manifest["parts"])
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest["expressions"]


def main(argv=None) -> int:
    """Attach an expression pack to a finished run, either way round:

      python -m seethrough_engine.expression <run> eye_closed=<png> mouth_open=<png>
      python -m seethrough_engine.expression <run> --from-run <donor run> eye_closed mouth_open

    The first recovers the regions a donor image changed; the second takes the
    donor run's own layers, which needs a second decomposition and gives the
    model's matte instead of an inferred one.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[1] == "--from-run":
        block = transplant_to_run(argv[0], argv[2], argv[3:] or ["eye_closed", "mouth_open"])
    elif len(argv) >= 2 and all("=" in a for a in argv[1:]):
        run_dir, donors = argv[0], {}
        for arg in argv[1:]:
            state, path = arg.split("=", 1)
            donors[state] = np.array(Image.open(path).convert("RGBA"))
        block = attach_to_run(run_dir, donors)
    else:
        print(main.__doc__, file=sys.stderr)
        return 2
    for name, entry in block["parts"].items():
        print(f"{name:16s} {entry['image']}  replaces {entry['replaces']}")
    if not block["parts"]:
        print("nothing recovered; see the report in the rig manifest", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
