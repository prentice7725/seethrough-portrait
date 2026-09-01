"""Production fidelity repair for semantic portrait layers.

The public interface applies the complete repair sequence exactly once.  It
uses the original still image as ground truth and contains no rig, motion,
mesh, weight, or runtime policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .image import composite_fidelity, composite_layers
from .semantic import SEMANTIC_Z_ORDER

REPAIR_VERSION = "1.0"
REPAIR_ORDER = (
    "reclaim_occluded",
    "fit_layer_tone",
    "fit_edge_alpha",
    "fit_seam_residual",
)

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

# Every generated layer carries a small constant colour bias against the
# original, and each one a different bias: on A-001 `topwear` is 8/4/5 bright,
# `ears` 6/7/6, `back hair` 3/4/3, while `face` is 1/3/2 dark. Where two of them
# meet, the difference between their biases is a step, which is what draws a
# line across the neck exactly where `neck` hands over to `topwear`.
#
# `body_remainder` measures 0, which is the check on this: it is the original's
# own pixels, and it is the only layer that is not generated.
#
# One constant per layer, fitted where the layer is the topmost thing visible,
# and capped -- a layer with little showing has little to fit against.
# ... and one constant is not enough, because a layer covers more than one
# material: `topwear` is a white shirt beside the neck and a beige cardigan
# everywhere else, and fitting both at once leaves its residual at 0 overall and
# +6 red at the seam -- which is the line, still there. So the bias is fitted per
# *material*, found by clustering the layer's own colours, which is exactly what
# flat anime shading gives up easily.
#
# Two families were tried and rejected first, and both failed for the same
# reason: the correction has to be a function of colour, not of position or
# brightness. A low-frequency field over position fixes the seam and absorbs
# real shading elsewhere (mae 9.99 against 9.47). A gain-and-offset over
# brightness is worse still -- a straight line fitted across a layer that holds
# both a dark outline and bright cloth moves the outline, and bad_ratio goes
# from 5.2% to 20%. A constant per colour cluster moves neither: two pixels of
# similar colour get similar corrections, so no new step can appear between
# them, and measurably none does.
TONE_MIN_SAMPLE = 300
TONE_MAX_SHIFT = 16
TONE_CLUSTERS = 8       # at most; a layer with fewer materials uses fewer
TONE_MIN_CLUSTER = 150  # ... and a cluster this small falls back to the layer's

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


def fit_layer_tone(layer_dict: dict[str, np.ndarray], original_rgba: np.ndarray, *,
                   order: tuple[str, ...] = SEMANTIC_Z_ORDER,
                   alpha_threshold: int = 200,
                   min_sample: int = TONE_MIN_SAMPLE,
                   max_shift: int = TONE_MAX_SHIFT,
                   clusters: int = TONE_CLUSTERS,
                   min_cluster: int = TONE_MIN_CLUSTER,
                   ) -> tuple[dict[str, np.ndarray], dict[str, list[list[int]]]]:
    """Take each layer's constant colour bias out of it.

    A generated layer is a little off from the picture it was decomposed from,
    and every layer is off by a different amount. Alone that is invisible --
    nobody can see a garment three levels too bright. Together it is a seam:
    where two layers meet, the difference between their biases is a step, and a
    step along a boundary is a line. On A-001 that is the line across the neck,
    where `neck` at +1 hands over to `topwear` at -8.

    The bias is measured where a layer is the topmost thing visible, so it is
    compared against pixels it is actually responsible for, and it is capped,
    because a layer with little showing has little to fit against.

    This is the one place the pipeline changes a layer's colour rather than its
    alpha. `body_remainder` measures 0 and stays untouched, which is the check
    that the measurement means what it says: it is the original's own pixels,
    and the only layer here that was not generated.
    """
    original = np.asarray(original_rgba)[..., :3].astype(np.int16)
    canvas_h, canvas_w = original.shape[:2]

    def rank(tag: str) -> int:
        return order.index(tag) if tag in order else -1

    tags = sorted(layer_dict, key=rank)
    owner = np.full((canvas_h, canvas_w), -1, np.int16)
    for index, tag in enumerate(tags):
        owner[np.asarray(layer_dict[tag])[..., 3] > alpha_threshold] = index

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    out: dict[str, np.ndarray] = {}
    shifts: dict[str, list[list[int]]] = {}
    for index, tag in enumerate(tags):
        layer = np.asarray(layer_dict[tag])
        visible = owner == index
        sample = int(visible.sum())
        if sample < min_sample:
            out[tag] = layer
            continue

        source = layer[..., :3][visible].astype(np.float32)
        target = original[visible].astype(np.float32)
        whole = np.clip(np.median(target - source, axis=0), -max_shift, max_shift)

        # One constant per material, and the materials are found by clustering
        # the layer's own colours -- flat anime shading separates them easily.
        count = max(1, min(clusters, sample // min_cluster))
        if count > 1:
            _, labels, centres = cv2.kmeans(source, count, None, criteria, 3,
                                            cv2.KMEANS_PP_CENTERS)
            labels = labels.ravel()
        else:
            labels = np.zeros(sample, np.int32)
            centres = source.mean(axis=0, keepdims=True)
        fitted = np.stack([
            np.clip(np.median(target[labels == c] - source[labels == c], axis=0),
                    -max_shift, max_shift)
            if int((labels == c).sum()) >= min_cluster else whole
            for c in range(count)]).astype(np.float32)

        if not np.any(np.abs(fitted) >= 1):
            out[tag] = layer
            continue

        # Every pixel of the layer takes the shift of the material it belongs
        # to, including the ones currently hidden: they are the same cloth, and
        # a turn may bring them into view.
        flat = layer[..., :3].reshape(-1, 3).astype(np.float32)
        nearest = ((flat[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        patched = np.array(layer, copy=True)
        patched[..., :3] = np.clip(flat + fitted[nearest], 0, 255
                                   ).reshape(layer.shape[0], layer.shape[1], 3).astype(np.uint8)
        out[tag] = patched
        shifts[tag] = [[int(v) for v in row] for row in fitted]

    return out, shifts


def fit_edge_alpha(layer_dict: dict[str, np.ndarray], original_rgba: np.ndarray, *,
                   order: tuple[str, ...] = SEMANTIC_Z_ORDER,
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


# The last of the four. `reclaim_occluded` settles who owns a contested pixel,
# `fit_edge_alpha` how much of a layer shows, `fit_layer_tone` each material's
# colour -- and a residual survives all three exactly where two *different*
# generated layers meet, because none of them compares the two against each
# other. On A-001 that is 1 to 2 luma along the neck's boundary with the
# garment, which the seam guard ranks first and a whole-image average cannot
# see at all.
#
# It is fixed by the only thing that knows the answer: the original, read in a
# narrow band on each side of the boundary and faded to nothing a few pixels in,
# so the correction cannot introduce an edge of its own.
SEAM_PAIRS: tuple[tuple[str, str], ...] = (("topwear", "neck"), ("neckwear", "neck"))
SEAM_BAND_PX = 3
SEAM_MAX_SHIFT = 12

# This pass is intentionally static. Whether two repaired parts may move apart
# safely is a downstream deformation-guard question, not portrait repair.


def fit_seam_residual(layer_dict: dict[str, np.ndarray], original_rgba: np.ndarray, *,
                      pairs: tuple[tuple[str, str], ...] = SEAM_PAIRS,
                      order: tuple[str, ...] = SEMANTIC_Z_ORDER,
                      band: int = SEAM_BAND_PX, max_shift: int = SEAM_MAX_SHIFT,
                      alpha_threshold: int = 10,
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Take the residual out of both sides of a shared boundary.

    Each layer is corrected on its own side, by what the original says is
    missing there, faded to zero `band` pixels in. A fade rather than a cut:
    two levels spread over three pixels is a gradient, which nobody sees, while
    the same two levels at a boundary is a line, which is the thing being
    removed.
    """
    original = np.asarray(original_rgba)[..., :3].astype(np.float32)
    canvas_h, canvas_w = original.shape[:2]
    out = dict(layer_dict)
    report: dict[str, Any] = {}

    def rank(tag: str) -> int:
        return order.index(tag) if tag in order else -1

    for front_tag, back_tag in pairs:
        if front_tag not in out or back_tag not in out:
            continue
        composite = composite_layers(out, (canvas_h, canvas_w), order=order,
                                     alpha_threshold=alpha_threshold)[..., :3].astype(np.float32)
        owner = np.full((canvas_h, canvas_w), -1, np.int16)
        for index, tag in enumerate(sorted(out, key=rank)):
            owner[np.asarray(out[tag])[..., 3] > 128] = index
        names = sorted(out, key=rank)
        front, back = names.index(front_tag), names.index(back_tag)

        # Where the two actually touch, in either direction.
        contact = np.zeros((canvas_h, canvas_w), np.uint8)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rolled = np.roll(owner, shift, axis=(0, 1))
            contact |= (((owner == front) & (rolled == back))
                        | ((owner == back) & (rolled == front))).astype(np.uint8)
        if not contact.any():
            report[f"{front_tag}|{back_tag}"] = {"skipped": "no contact"}
            continue

        residual = np.clip(original - composite, -max_shift, max_shift)
        distance = cv2.distanceTransform(1 - contact, cv2.DIST_L2, 3)
        falloff = np.clip(1.0 - distance / float(band), 0.0, 1.0)
        falloff = falloff * falloff * (3.0 - 2.0 * falloff)     # smooth, so no inner edge

        moved = {}
        for tag, index in ((front_tag, front), (back_tag, back)):
            side = (owner == index) & (falloff > 0)
            if not side.any():
                continue
            layer = np.array(out[tag], copy=True)
            # Divided by the layer's own coverage: a correction of R applied to
            # a layer showing at alpha a moves the composite by a*R, and at a
            # seam the alphas are exactly where they are not 1. Clamped, so a
            # nearly transparent pixel cannot demand an enormous shift.
            coverage = np.clip(layer[..., 3].astype(np.float32) / 255.0, 0.5, 1.0)
            correction = residual / coverage[..., None] * (falloff * side)[..., None]
            # Rounded, not truncated: the corrections here are one or two
            # levels, and truncation would quietly eat most of one of them.
            layer[..., :3] = np.rint(np.clip(layer[..., :3].astype(np.float32)
                                             + np.clip(correction, -max_shift, max_shift),
                                             0, 255)).astype(np.uint8)
            out[tag] = layer
            moved[tag] = int(side.sum())
        report[f"{front_tag}|{back_tag}"] = {"px": moved}

    return out, report



@dataclass(frozen=True)
class RepairResult:
    layers: dict[str, np.ndarray]
    report: dict[str, Any]


def repair_portrait_layers(layer_dict: dict[str, np.ndarray],
                           original_rgba: np.ndarray) -> RepairResult:
    """Return production-repaired canonical semantic layers.

    Ordering is part of the interface.  Callers do not invoke the individual
    stages and downstream consumers must not invoke this module at all.
    """
    working, reclaimed = reclaim_occluded(layer_dict, original_rgba)
    working, tone_fit = fit_layer_tone(working, original_rgba)
    working, edge_fit = fit_edge_alpha(working, original_rgba)
    working, seam_fit = fit_seam_residual(working, original_rgba)
    return RepairResult(
        layers=working,
        report={
            "version": REPAIR_VERSION,
            "order": list(REPAIR_ORDER),
            "reclaim_occluded": reclaimed,
            "fit_layer_tone": tone_fit,
            "fit_edge_alpha": edge_fit,
            "fit_seam_residual": seam_fit,
        },
    )
