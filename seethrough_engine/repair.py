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
from .scale import canvas_scale, odd_kernel, scale_area, scale_length
from .semantic import SEMANTIC_Z_ORDER

REPAIR_VERSION = "1.8"
REPAIR_ORDER = (
    "reclaim_occluded",
    "fit_layer_tone",
    "fit_edge_alpha",
    "clean_garment_orphans",
    "fit_edge_alpha_final",
    "fit_mouth_contact",
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

# A garment can retain isolated skin/head fragments after contested ownership
# has been reclaimed.  These are not "small components are bad": a button,
# ribbon, or detached ornament is valid garment content.  A component is only
# removable when it is detached from the garment main mass, lies inside a
# competing head/neck semantic region, that semantic explains the original
# decisively better, and a virtual removal does not worsen reconstruction.
GARMENT_TAGS: tuple[str, ...] = ("topwear", "neckwear")
GARMENT_COMPETING_TAGS: tuple[str, ...] = (
    "neck", "head", "face", "ears", "earl", "earr",
)
ORPHAN_ALPHA_THRESHOLD = 10
ORPHAN_MIN_AREA_AT_768 = 1
ORPHAN_MAX_AREA_RATIO = 0.02
ORPHAN_MAX_DISTANCE_DIAGONAL_RATIO = 0.20
ORPHAN_SEMANTIC_SUPPORT_RATIO = 0.80
ORPHAN_BETTER_RATIO = 0.70
ORPHAN_ERROR_MARGIN = 12
ORPHAN_REDUNDANT_MAX_ERROR = 18
ORPHAN_REDUNDANT_RATIO = 0.70
ORPHAN_FRINGE_PX = 2

# Mouth is a feature layer, but model output can include a soft skin-coloured
# matte around the drawing.  Only the narrow raster contact band is eligible;
# the interior (lip/teeth/outline) is never globally trimmed.
MOUTH_CONTACT_BAND_PX = 5
MOUTH_ALPHA_DROP = 0.08
MOUTH_ORIGINAL_MARGIN = 8
MOUTH_MIN_CONTRAST = 12
MOUTH_HALO_ERROR_MIN = 15
MOUTH_HALO_MIN_LUMA = 110
MOUTH_HALO_MAX_CHROMA = 100
MOUTH_FRINGE_BASE_ERROR_MAX = 40
MOUTH_FRINGE_BAND_RATIO = 0.45

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
    feather = (max(0.5, RECLAIM_FEATHER_PX * canvas_scale(original.shape))
               if feather is None else feather)
    if min_area is None:
        min_area = scale_area(RECLAIM_MIN_AREA_AT_768, original.shape)

    out = dict(layer_dict)
    moved: dict[str, int] = {}
    kernel_size = odd_kernel(5, original.shape)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

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


def _static_reconstruction_score(layer_dict: dict[str, np.ndarray],
                                 original_rgba: np.ndarray,
                                 order: tuple[str, ...]) -> tuple[int, int]:
    """Exact RGB error and bad-pixel count for a non-regression gate.

    The public report rounds MAE for readability.  A repair gate must not use
    that rounded value: several individually hidden regressions could
    otherwise accumulate into a visible final difference.
    """
    original = np.asarray(original_rgba)
    composite = composite_layers(layer_dict, original.shape[:2], order=order)
    subject = original[..., 3] > ORPHAN_ALPHA_THRESHOLD
    error = np.abs(original[..., :3].astype(np.int32)
                   - composite[..., :3].astype(np.int32)).sum(axis=2)
    values = error[subject]
    return int(values.sum()), int((values > 30).sum())


def _merge_semantic_ownership(owner: np.ndarray, contaminant: np.ndarray,
                              mask: np.ndarray, *, owner_is_front: bool) -> np.ndarray:
    """Fold one RGBA contribution into its semantic owner without flattening.

    Alpha compositing is associative, so merging the two layers in their
    existing front/back order preserves how their pair renders over any layer
    below them.  The caller still checks the full canonical composite because
    unrelated layers may sit between the pair in the global z-order.
    """
    if not mask.any():
        return owner
    patched = np.array(owner, copy=True)
    owner_a = owner[..., 3].astype(np.float32) / 255.0
    contaminant_a = contaminant[..., 3].astype(np.float32) / 255.0
    owner_rgb = owner[..., :3].astype(np.float32)
    contaminant_rgb = contaminant[..., :3].astype(np.float32)
    if owner_is_front:
        combined_a = owner_a + contaminant_a * (1.0 - owner_a)
        premultiplied = (
            owner_rgb * owner_a[..., None]
            + contaminant_rgb * contaminant_a[..., None] * (1.0 - owner_a[..., None])
        )
    else:
        combined_a = contaminant_a + owner_a * (1.0 - contaminant_a)
        premultiplied = (
            contaminant_rgb * contaminant_a[..., None]
            + owner_rgb * owner_a[..., None] * (1.0 - contaminant_a[..., None])
        )
    combined_rgb = premultiplied / np.maximum(combined_a[..., None], 1e-6)
    patched[..., :3][mask] = np.rint(
        np.clip(combined_rgb[mask], 0, 255)).astype(np.uint8)
    patched[..., 3][mask] = np.rint(
        np.clip(combined_a[mask] * 255.0, 0, 255)).astype(np.uint8)
    return patched


def clean_garment_orphans(
    layer_dict: dict[str, np.ndarray],
    original_rgba: np.ndarray,
    *,
    garment_tags: tuple[str, ...] = GARMENT_TAGS,
    competing_tags: tuple[str, ...] = GARMENT_COMPETING_TAGS,
    order: tuple[str, ...] = SEMANTIC_Z_ORDER,
    alpha_threshold: int = ORPHAN_ALPHA_THRESHOLD,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Remove only well-supported orphan semantic contamination from garments.

    The garment main mass is never a candidate.  Detached components are
    evaluated using four independent observations: distance from that mass,
    overlap with a head/face/neck semantic, per-pixel agreement with the
    original still, and an exact virtual-composite non-regression gate.  Small
    size alone never authorizes deletion.  Ambiguous components remain in the
    canonical layer and are recorded in the report.
    """
    original = np.asarray(original_rgba)
    original_rgb = original[..., :3].astype(np.int32)
    canvas_h, canvas_w = original.shape[:2]
    min_area = scale_area(ORPHAN_MIN_AREA_AT_768, original.shape)
    max_distance = ORPHAN_MAX_DISTANCE_DIAGONAL_RATIO * float(
        np.hypot(canvas_h, canvas_w))
    out = dict(layer_dict)
    report: dict[str, Any] = {}

    support = np.zeros((canvas_h, canvas_w), bool)
    best_competing_error = np.full((canvas_h, canvas_w), 1_000_000, np.int32)
    best_competing_owner = np.full((canvas_h, canvas_w), -1, np.int16)
    available_competing_tags: list[str] = []
    for tag in competing_tags:
        layer = out.get(tag)
        if layer is None:
            continue
        arr = np.asarray(layer)
        present = arr[..., 3] > alpha_threshold
        if not present.any():
            continue
        owner_index = len(available_competing_tags)
        available_competing_tags.append(tag)
        error = np.abs(original_rgb - arr[..., :3].astype(np.int32)).sum(axis=2)
        better = present & (error < best_competing_error)
        best_competing_error[better] = error[better]
        best_competing_owner[better] = owner_index
        support |= present

    if not support.any():
        return out, report

    current_score = _static_reconstruction_score(out, original, order)
    # This fringe follows the raster antialiasing footprint, not a geometric
    # distance in the portrait. The resize kernel remains about two output
    # pixels at both 768 and 1024; scaling it to three overreaches into valid
    # cloth and makes exact ownership transfer fail on A002.
    fringe = ORPHAN_FRINGE_PX
    fringe_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * fringe + 1,) * 2)

    for tag in garment_tags:
        layer = out.get(tag)
        if layer is None:
            continue
        arr = np.asarray(layer)
        strong = arr[..., 3] > alpha_threshold
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            strong.astype(np.uint8), 8)
        if count <= 2:
            continue
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        main_area = int(stats[main, cv2.CC_STAT_AREA])
        max_area = max(min_area, int(round(main_area * ORPHAN_MAX_AREA_RATIO)))
        distance_to_main = cv2.distanceTransform(
            (labels != main).astype(np.uint8), cv2.DIST_L2, 3)
        garment_error = np.abs(
            original_rgb - arr[..., :3].astype(np.int32)).sum(axis=2)
        component_rows: list[dict[str, Any]] = []
        removed_px = 0
        ignored_tiny_components = 0

        candidates = sorted(
            range(1, count),
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        for index in candidates:
            if index == main:
                continue
            component = labels == index
            area = int(stats[index, cv2.CC_STAT_AREA])
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if area < min_area:
                ignored_tiny_components += 1
                continue
            distance = float(distance_to_main[component].min())
            semantic_ratio = float((component & support).sum()) / max(area, 1)
            decisively_competing = (
                support
                & (best_competing_error + ORPHAN_ERROR_MARGIN < garment_error)
            )
            better_ratio = float((component & decisively_competing).sum()) / max(area, 1)
            redundant_ownership = (
                support
                & (best_competing_error <= ORPHAN_REDUNDANT_MAX_ERROR)
                & (garment_error <= ORPHAN_REDUNDANT_MAX_ERROR)
            )
            redundant_ratio = float(
                (component & redundant_ownership).sum()) / max(area, 1)
            fill_ratio = float(area) / max(width * height, 1)
            row: dict[str, Any] = {
                "area_px": area,
                "bbox_xywh": [x, y, width, height],
                "distance_to_main_px": round(distance, 2),
                "semantic_support_ratio": round(semantic_ratio, 4),
                "competing_better_ratio": round(better_ratio, 4),
                "redundant_ownership_ratio": round(redundant_ratio, 4),
                "fill_ratio": round(fill_ratio, 4),
            }

            structural = (
                min_area <= area <= max_area
                and distance <= max_distance
            )
            evidence = (
                semantic_ratio >= ORPHAN_SEMANTIC_SUPPORT_RATIO
                and (
                    better_ratio >= ORPHAN_BETTER_RATIO
                    or redundant_ratio >= ORPHAN_REDUNDANT_RATIO
                )
            )
            if not structural:
                row.update(status="kept", reason="outside conservative size/distance gate")
                component_rows.append(row)
                continue
            if not evidence:
                row.update(status="ambiguous", reason="semantic/original evidence is not decisive")
                component_rows.append(row)
                continue

            fringe = cv2.dilate(component.astype(np.uint8), fringe_kernel).astype(bool)
            ownership_evidence = decisively_competing | redundant_ownership
            remove = component | (
                fringe & (arr[..., 3] > 0) & support & ownership_evidence)
            patched = np.array(arr, copy=True)
            patched[remove] = 0
            tentative = dict(out)
            tentative[tag] = patched
            tentative_score = _static_reconstruction_score(tentative, original, order)
            strategy = "clear"
            transferred_px: dict[str, int] = {}
            if not (tentative_score[0] <= current_score[0]
                    and tentative_score[1] <= current_score[1]):
                transferred = dict(tentative)
                rank = {name: index for index, name in enumerate(order)}
                for owner_index, owner_tag in enumerate(available_competing_tags):
                    owner_mask = remove & (best_competing_owner == owner_index)
                    if not owner_mask.any():
                        continue
                    transferred[owner_tag] = _merge_semantic_ownership(
                        np.asarray(out[owner_tag]), arr, owner_mask,
                        owner_is_front=(
                            rank.get(owner_tag, -1) > rank.get(tag, -1)
                        ),
                    )
                    transferred_px[owner_tag] = int(owner_mask.sum())
                transferred_score = _static_reconstruction_score(
                    transferred, original, order)
                if (transferred_score[0] <= current_score[0]
                        and transferred_score[1] <= current_score[1]):
                    tentative = transferred
                    tentative_score = transferred_score
                    strategy = "transfer"
            if (tentative_score[0] <= current_score[0]
                    and tentative_score[1] <= current_score[1]):
                previous_score = current_score
                out = tentative
                arr = patched
                current_score = tentative_score
                count_removed = int(remove.sum())
                removed_px += count_removed
                row.update(
                    status="removed",
                    strategy=strategy,
                    removed_px=count_removed,
                    transferred_px=transferred_px if strategy == "transfer" else {},
                    rgb_error_delta=int(tentative_score[0] - previous_score[0]),
                    bad_px_delta=int(tentative_score[1] - previous_score[1]),
                )
            else:
                row.update(status="ambiguous", reason="virtual removal worsened reconstruction")
            component_rows.append(row)

        if component_rows:
            report[tag] = {
                "main_area_px": main_area,
                "removed_px": removed_px,
                "ignored_tiny_components": ignored_tiny_components,
                "components": component_rows,
            }

    return out, report


def fit_mouth_contact(
    layer_dict: dict[str, np.ndarray],
    original_rgba: np.ndarray,
    *,
    mouth_tag: str = "mouth",
    order: tuple[str, ...] = SEMANTIC_Z_ORDER,
    band: int = MOUTH_CONTACT_BAND_PX,
    alpha_threshold: int = ORPHAN_ALPHA_THRESHOLD,
    alpha_drop: float = MOUTH_ALPHA_DROP,
    original_margin: int = MOUTH_ORIGINAL_MARGIN,
    min_contrast: int = MOUTH_MIN_CONTRAST,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Remove a skin-coloured matte from the mouth's local contact band.

    The raw mouth layer is allowed to contain the complete drawing, including
    antialiasing.  What is not production-safe is an opaque skin patch around
    that drawing: because ``mouth`` is in front of ``face``, it becomes a
    visible oval in the canonical composite.  The original still is the
    ground truth.  For each boundary pixel we solve the alpha that blends the
    mouth colour over the stack beneath it to reproduce the original, and only
    accept decreases that make the underlying stack explain the pixel better.

    This is deliberately local and feature-agnostic: no rig/depth knowledge,
    no global trim, and no connected-component deletion.  A full-canvas exact
    reconstruction gate rejects any ambiguous proposal.
    """
    out = dict(layer_dict)
    mouth = out.get(mouth_tag)
    if mouth is None:
        return out, {"status": "missing"}
    original = np.asarray(original_rgba)
    if original.ndim != 3 or original.shape[-1] != 4:
        raise ValueError("original_rgba must be an HxWx4 canvas")
    arr = np.asarray(mouth)
    if arr.shape != original.shape:
        raise ValueError("mouth layer and original must share an HxWx4 canvas")

    strong = arr[..., 3] > alpha_threshold
    if not strong.any():
        return out, {"status": "empty"}
    ys, xs = np.where(strong)
    band = max(1, int(band))
    # Estimate the actual drawing from original-vs-underlay contrast. A
    # percentile keeps this independent of canvas resolution and avoids a
    # fixed RGB cutoff. The remaining connected matte is eligible only in a
    # narrow neighbourhood of that drawing.
    beneath_for_core = dict(out)
    beneath_for_core.pop(mouth_tag, None)
    core_base = composite_layers(
        beneath_for_core, original.shape[:2], order=order,
        alpha_threshold=alpha_threshold)[..., :3].astype(np.float32)
    core_score = np.abs(
        original[..., :3].astype(np.float32) - core_base
    ).sum(axis=2)
    core_cut = float(np.percentile(core_score[strong], 75.0))
    drawing_core = strong & (core_score >= core_cut)
    if not drawing_core.any():
        drawing_core = strong
    # The model's matte can sit farther from a thin lip stroke than the
    # nominal raster band.  Scale the neighbourhood from the observed feature
    # extent, while keeping it a mouth-local ROI (never a canvas/global trim).
    core_y, core_x = np.where(drawing_core)
    feature_span = max(
        int(core_x.max() - core_x.min() + 1),
        int(core_y.max() - core_y.min() + 1),
    )
    band = max(band, int(round(feature_span * 0.30)))
    core_neighbourhood = cv2.dilate(
        drawing_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2),
    ).astype(bool) & strong
    contact = core_neighbourhood & ~drawing_core

    # Pixels hidden by a semantic later in the z-order cannot be explained by
    # the mouth and are not safe to edit.
    rank = {tag: index for index, tag in enumerate(order)}
    mouth_rank = rank.get(mouth_tag, -1)
    visible_owner = np.asarray(arr)[..., 3] > 128
    for tag, image in out.items():
        if rank.get(tag, -1) > mouth_rank:
            visible_owner &= ~(np.asarray(image)[..., 3] > 128)
    contact &= visible_owner
    if not contact.any():
        return out, {"status": "no_visible_contact"}

    beneath = dict(out)
    beneath.pop(mouth_tag, None)
    base = composite_layers(beneath, original.shape[:2], order=order,
                            alpha_threshold=alpha_threshold)
    current = composite_layers(out, original.shape[:2], order=order,
                               alpha_threshold=alpha_threshold)
    O = original[..., :3].astype(np.float32)
    B = base[..., :3].astype(np.float32)
    M = arr[..., :3].astype(np.float32)
    C = current[..., :3].astype(np.float32)
    alpha = arr[..., 3].astype(np.float32) / 255.0
    delta = M - B
    denominator = (delta * delta).sum(axis=2)
    solvable = contact & (denominator > 3.0 * float(min_contrast) ** 2)
    desired = np.clip(((O - B) * delta).sum(axis=2)
                      / np.maximum(denominator, 1e-6), 0.0, 1.0)
    current_error = np.abs(O - C).sum(axis=2)
    beneath_error = np.abs(O - B).sum(axis=2)
    candidate = (
        solvable
        & (desired + float(alpha_drop) < alpha)
        & (beneath_error + float(original_margin) < current_error)
    )

    # A skin-coloured halo is not mouth artwork.  Earlier versions transferred
    # it into `face` to preserve the rendered pixels, but that merely moved the
    # oval into the canonical face layer (and made the artifact survive every
    # later composite).  Clear the halo from mouth instead; the existing face
    # layer is the owner of the surrounding skin and remains untouched.
    face = out.get("face")
    original_mouth_error = np.abs(O - M).sum(axis=2)
    halo = (
        (strong & ~core_neighbourhood) & visible_owner & (face is not None)
        & (original_mouth_error > float(MOUTH_HALO_ERROR_MIN))
        & (M.mean(axis=2) >= float(MOUTH_HALO_MIN_LUMA))
        & ((M.max(axis=2) - M.min(axis=2)) <= float(MOUTH_HALO_MAX_CHROMA))
    )
    # Antialiased segmentation edges can be dark even when the broad matte is
    # skin-coloured, producing a dotted oval after compositing.  Keep this
    # check local to the observed mouth feature and require evidence that the
    # face already explains the original.  This catches both low-alpha dark
    # fringes and opaque wrong-coloured matte pixels; real lip strokes normally
    # match the original and remain untouched.
    fringe_band = max(1, int(round(feature_span * MOUTH_FRINGE_BAND_RATIO)))
    fringe_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * fringe_band + 1,) * 2,
    )
    fringe_region = cv2.dilate(
        drawing_core.astype(np.uint8), fringe_kernel,
    ).astype(bool) & strong
    fringe_visible = strong.copy()
    for tag, image in out.items():
        if rank.get(tag, -1) > mouth_rank:
            fringe_visible &= ~(np.asarray(image)[..., 3] > alpha_threshold)
    fringe = (
        fringe_region & ~drawing_core & fringe_visible
        & (original_mouth_error > np.maximum(
            float(MOUTH_HALO_ERROR_MIN),
            beneath_error + float(MOUTH_ORIGINAL_MARGIN),
        ))
        & (beneath_error <= float(MOUTH_FRINGE_BASE_ERROR_MAX))
    )
    halo |= fringe

    report: dict[str, Any] = {
        "status": "unchanged",
        "bbox_xywh": [int(xs.min()), int(ys.min()),
                       int(xs.max() - xs.min() + 1),
                       int(ys.max() - ys.min() + 1)],
        "contact_px": int(contact.sum()),
        "drawing_core_px": int(drawing_core.sum()),
        "core_cut_rgb_error": round(core_cut, 3),
        "candidate_px": int(candidate.sum()),
        "halo_candidate_px": int(halo.sum()),
    }
    if not candidate.any() and not halo.any():
        return out, report

    tentative = dict(out)
    transferred_px = 0
    if halo.any() and face is not None:
        transferred_mouth = np.array(arr, copy=True)
        transferred_mouth[halo] = 0
        tentative[mouth_tag] = transferred_mouth

    # Recompute the beneath stack after any ownership transfer, then apply the
    # original-vs-composite alpha solve only to candidates not already moved.
    remaining = candidate & ~halo
    if remaining.any():
        beneath_after = dict(tentative)
        beneath_after.pop(mouth_tag, None)
        base_after = composite_layers(
            beneath_after, original.shape[:2], order=order,
            alpha_threshold=alpha_threshold)[..., :3].astype(np.float32)
        mouth_after = np.asarray(tentative[mouth_tag])
        delta_after = mouth_after[..., :3].astype(np.float32) - base_after
        den_after = (delta_after * delta_after).sum(axis=2)
        desired_after = np.clip(
            ((O - base_after) * delta_after).sum(axis=2)
            / np.maximum(den_after, 1e-6), 0.0, 1.0)
        alpha_after = mouth_after[..., 3].astype(np.float32) / 255.0
        patched = np.array(mouth_after, copy=True)
        patched[..., 3] = np.where(
            remaining,
            np.rint(np.minimum(alpha_after, desired_after) * 255.0),
            mouth_after[..., 3],
        ).astype(np.uint8)
        tentative[mouth_tag] = patched

    before_score = _static_reconstruction_score(out, original, order)
    after_score = _static_reconstruction_score(tentative, original, order)
    # Exact gate: a local improvement must not trade for a regression elsewhere
    # or increase the count of visibly bad pixels.
    if after_score[0] > before_score[0] or after_score[1] > before_score[1]:
        report.update(status="rejected", reason="virtual ownership change worsened reconstruction",
                      rgb_error_delta=int(after_score[0] - before_score[0]),
                      bad_px_delta=int(after_score[1] - before_score[1]))
        return out, report

    out = tentative
    report.update(
        status="applied",
        changed_px=int((candidate | halo).sum()),
        transferred_px=transferred_px,
        alpha_drop_mean=round(float((alpha[remaining] - desired[remaining]).mean()), 4)
        if remaining.any() else 0.0,
        rgb_error_delta=int(after_score[0] - before_score[0]),
        bad_px_delta=int(after_score[1] - before_score[1]),
    )
    return out, report


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
# so the correction cannot introduce an edge of its own.  This is deliberately
# the *last* repair stage: orphan cleanup can transfer garment debris back to
# the neck or another face semantic, changing both alpha and colour at the
# exact boundary that needs the final correction.
SEAM_PAIRS: tuple[tuple[str, str], ...] = (("topwear", "neck"), ("neckwear", "neck"))
SEAM_BAND_PX = 3
SEAM_MAX_SHIFT = 12
# A seam correction changes the composite seen by the other side of the same
# boundary.  One solve is intentionally conservative but under-corrects
# partially-covered pixels; a second, independently clamped residual solve
# converges that handoff without turning the pass into broad colour fitting.
SEAM_ITERATIONS = 2

# This pass is intentionally static. Whether two repaired parts may move apart
# safely is a downstream deformation-guard question, not portrait repair.


def fit_seam_residual(layer_dict: dict[str, np.ndarray], original_rgba: np.ndarray, *,
                      pairs: tuple[tuple[str, str], ...] = SEAM_PAIRS,
                      order: tuple[str, ...] = SEMANTIC_Z_ORDER,
                      band: int = SEAM_BAND_PX, max_shift: int = SEAM_MAX_SHIFT,
                      iterations: int = 1,
                      alpha_threshold: int = 10,
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Take the residual out of both sides of a shared boundary.

    Each layer is corrected on its own side, by what the original says is
    missing there, faded to zero `band` pixels in. A fade rather than a cut:
    two levels spread over three pixels is a gradient, which nobody sees, while
    the same two levels at a boundary is a line, which is the thing being
    removed.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    original = np.asarray(original_rgba)[..., :3].astype(np.float32)
    canvas_h, canvas_w = original.shape[:2]
    out = dict(layer_dict)
    report: dict[str, Any] = {}
    initial_rgb = {
        tag: np.asarray(image)[..., :3].astype(np.float32)
        for tag, image in out.items()
    }

    def rank(tag: str) -> int:
        return order.index(tag) if tag in order else -1

    for front_tag, back_tag in pairs:
        if front_tag not in out or back_tag not in out:
            continue
        moved = {}
        completed = 0
        for _ in range(iterations):
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
                break

            residual = np.clip(original - composite, -max_shift, max_shift)
            distance = cv2.distanceTransform(1 - contact, cv2.DIST_L2, 3)
            falloff = np.clip(1.0 - distance / float(band), 0.0, 1.0)
            falloff = falloff * falloff * (3.0 - 2.0 * falloff)  # smooth, no inner edge

            for tag, index in ((front_tag, front), (back_tag, back)):
                side = (owner == index) & (falloff > 0)
                if not side.any():
                    continue
                layer = np.array(out[tag], copy=True)
                coverage = np.clip(layer[..., 3].astype(np.float32) / 255.0, 0.5, 1.0)
                correction = residual / coverage[..., None] * (falloff * side)[..., None]
                corrected = layer[..., :3].astype(np.float32) + np.clip(
                    correction, -max_shift, max_shift)
                # Two local solves must not allow a semantic to drift farther
                # than the documented one-pass cap from its input colour.
                corrected = np.clip(corrected,
                                    initial_rgb[tag] - max_shift,
                                    initial_rgb[tag] + max_shift)
                layer[..., :3] = np.rint(np.clip(corrected, 0, 255)).astype(np.uint8)
                out[tag] = layer
                moved[tag] = moved.get(tag, 0) + int(side.sum())
            completed += 1
        key = f"{front_tag}|{back_tag}"
        report[key] = ({"px": moved, "iterations": completed}
                       if completed else {"skipped": "no contact"})

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
    shape = np.asarray(original_rgba).shape
    working, reclaimed = reclaim_occluded(
        layer_dict, original_rgba,
        min_area=scale_area(RECLAIM_MIN_AREA_AT_768, shape),
        feather=max(0.5, RECLAIM_FEATHER_PX * canvas_scale(shape)),
    )
    working, tone_fit = fit_layer_tone(
        working, original_rgba,
        min_sample=scale_area(TONE_MIN_SAMPLE, shape),
        min_cluster=scale_area(TONE_MIN_CLUSTER, shape),
    )
    working, edge_fit = fit_edge_alpha(
        working, original_rgba,
        band=scale_length(EDGE_BAND_PX, shape),
        outside=scale_length(EDGE_OUTSIDE_PX, shape),
    )
    working, orphan_cleanup = clean_garment_orphans(working, original_rgba)
    # Cleanup can alter an ownership edge. Refit edge coverage against the
    # final owner set, then fit its colour residual. This is deliberately a
    # second narrow-band solve, not a broad re-application of tone fitting.
    working, final_edge_fit = fit_edge_alpha(
        working, original_rgba,
        band=scale_length(EDGE_BAND_PX, shape),
        outside=scale_length(EDGE_OUTSIDE_PX, shape),
    )
    working, mouth_contact = fit_mouth_contact(
        working, original_rgba,
        band=scale_length(MOUTH_CONTACT_BAND_PX, shape),
    )
    working, seam_fit = fit_seam_residual(
        working, original_rgba,
        band=scale_length(SEAM_BAND_PX, shape),
        iterations=SEAM_ITERATIONS,
    )
    return RepairResult(
        layers=working,
        report={
            "version": REPAIR_VERSION,
            "order": list(REPAIR_ORDER),
            "reclaim_occluded": reclaimed,
            "fit_layer_tone": tone_fit,
            "fit_edge_alpha": edge_fit,
            "clean_garment_orphans": orphan_cleanup,
            "fit_edge_alpha_final": final_edge_fit,
            "fit_mouth_contact": mouth_contact,
            "fit_seam_residual": seam_fit,
        },
    )
