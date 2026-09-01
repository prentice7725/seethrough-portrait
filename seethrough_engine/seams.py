"""Static seam guard: find the thin, long artifacts the composite metrics cannot see.

`composite_fidelity` averages over the subject, and a seam is one pixel wide.
On A-001 the whole-subject mae reached 8.84 while a line across the neck was
the first thing anyone noticed -- 178 pixels of boundary, 0.02% of the
silhouette, and the only thing in the frame the eye went to. An average cannot
weigh that and should not be asked to.

So this measures boundaries instead of areas. Wherever the topmost layer changes
from one pixel to the next, it compares the step the composite makes there
against the step the *original* makes at the same place. A real edge -- a
collar, a jaw, a lash -- steps in both and scores zero. A seam introduced by the
decomposition steps only in the composite.

The number that matters is not how wrong a boundary pixel is but how far the
wrongness runs: a 2-luma error over 200 contiguous pixels is a line, and the
same error scattered over 200 unrelated pixels is nothing. So the report is
ranked by the longest run, not by the mean. The input is the same full-canvas
canonical layer mapping written into Portrait Bundle v1.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .scale import scale_length

__all__ = [
    "STEP_THRESHOLD",
    "FLAT_STEP",
    "QUIET_ENERGY",
    "seam_report_layers",
    "compare_seam_report",
]

# A boundary pixel counts toward a run when the composite steps this much more
# than the original does. Below it the difference is inside the anti-aliasing of
# this art style and nothing is visible.
STEP_THRESHOLD = 1.0

# ... and only where the picture itself is flat. A boundary that steps 70 luma
# more than the original sounds alarming and is invisible when it lies along a
# drawn eyebrow, where the original steps 120: sharpening an edge that is
# already an edge changes nothing anyone can see. What the eye finds is the
# opposite -- a line drawn across skin that has none, which is why the neck seam
# was noticed at 2 luma while the brow at 70 was not. So a pixel only counts
# when the original is locally flat.
FLAT_STEP = 4.0

# ... and only where the *neighbourhood* is quiet, not just the one step across
# the boundary. The first ranking put `back hair | head_remainder` on top with a
# 125 px run, and at 3x magnification the two pictures are the same: each
# flagged pixel does sit between two locally flat values, but it sits inside a
# hair strand, and a 3-luma error beside a 100-luma strand edge is masked. On
# A-001 the local gradient energy of that region measures 187 against the neck
# seam's 3.4, and the subject's own median is 11.
#
# This is the guard correcting itself: it found something, the something turned
# out to be invisible, and the rule got sharper rather than the finding being
# waved away.
#
# Fixed rather than scaled to the picture's own median, which was tried and is
# wrong in principle: masking is local, so a busy drawing must not raise the bar
# for its own quiet parts. Scaled to the median, a picture that is loud
# everywhere calls its loud regions quiet.
QUIET_ENERGY = 40.0

# A line has a direction. Along a real seam the composite is consistently darker
# than the picture, or consistently lighter; along a boundary that merely has
# noisy pixels the sign flips from one to the next -- measured on A-001's hair
# boundary, +10.0, +10.5, -0.8, +1.7, -7.2. Only pixels agreeing with the
# boundary's own dominant sign count, which is what separates a line from a
# scatter of equally large errors.
#
# Magnitude does not separate them: ranked by error size over local activity,
# the hair boundaries come first and the neck seam -- the one anybody actually
# saw -- does not make the top eight.
SIGN_AGREEMENT = True

# `--check` compares against a recorded baseline rather than an absolute bar.
# An absolute bar set where we would like to be fails on the day it is written
# and then teaches nothing; a baseline fails when a change makes a seam worse,
# which is the question actually being asked of every future change. The
# baseline is a file in the repo, so what the pipeline achieved on a given day
# is a fact under review rather than a memory.
#
# The slack is deliberately small. These numbers are deterministic for a fixed
# run directory -- same layers, same arithmetic -- so anything past rounding is
# a real change.
RUN_SLACK_PX = 8
EXCESS_SLACK = 0.3

# How far from the boundary the band reaches, for the residual figure.
BAND_PX = 3

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _luma(rgb):
    return rgb.astype(np.float32) @ _LUMA


def _render_layers(layer_dict: dict[str, np.ndarray],
                   frame_size: tuple[int, int]):
    """Render canonical full-canvas layers and their topmost owner."""
    from .semantic import semantic_rank

    height, width = int(frame_size[0]), int(frame_size[1])
    rgb = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width, 1), np.float32)
    owner = np.full((height, width), -1, np.int16)
    tags = sorted(layer_dict, key=semantic_rank)
    for index, tag in enumerate(tags):
        image = np.asarray(layer_dict[tag]).astype(np.float32)
        if image.shape != (height, width, 4):
            raise ValueError(
                f"layer {tag!r} must match canvas {(height, width, 4)}, got {image.shape}"
            )
        a = image[..., 3:4] / 255.0
        rgb[:] = rgb * (1.0 - a) + image[..., :3] * a
        alpha[:] = np.clip(alpha + a, 0.0, 1.0)
        owner[image[..., 3] > 128] = index
    return rgb, alpha[..., 0], owner, tags


def _measure_seams(original: np.ndarray, composite: np.ndarray,
                   owner: np.ndarray, tags: list[str], *, run: str,
                   step_threshold: float = STEP_THRESHOLD,
                   flat_step: float = FLAT_STEP,
                   quiet_energy: float = QUIET_ENERGY,
                   band: int | None = None) -> dict[str, Any]:
    band = scale_length(BAND_PX, original.shape) if band is None else band
    subject = original[:, :, 3] > 128
    lc, lo = _luma(composite), _luma(original[:, :, :3])
    # How busy the picture is around each pixel, which is what decides whether
    # an error of a few levels can be seen at all.
    energy = cv2.boxFilter(np.abs(cv2.Sobel(lo, cv2.CV_32F, 1, 0, ksize=3))
                           + np.abs(cv2.Sobel(lo, cv2.CV_32F, 0, 1, ksize=3)), -1, (7, 7))
    quiet = energy < quiet_energy
    # Which way the composite is wrong at each pixel, for the sign test below.
    residual = lc - lo

    seams: dict[tuple[int, int], dict[str, Any]] = {}
    height, width = owner.shape
    # Both directions, so a boundary is found whichever way it runs.
    for axis in (0, 1):
        a = owner[:-1, :] if axis == 0 else owner[:, :-1]
        b = owner[1:, :] if axis == 0 else owner[:, 1:]
        valid = (a >= 0) & (b >= 0) & (a != b)
        valid &= (subject[:-1, :] & subject[1:, :]) if axis == 0 else \
                 (subject[:, :-1] & subject[:, 1:])
        if not valid.any():
            continue
        step_c = np.abs((lc[:-1, :] - lc[1:, :]) if axis == 0 else (lc[:, :-1] - lc[:, 1:]))
        step_o = np.abs((lo[:-1, :] - lo[1:, :]) if axis == 0 else (lo[:, :-1] - lo[:, 1:]))
        excess = step_c - step_o
        ys, xs = np.nonzero(valid)
        for y, x in zip(ys.tolist(), xs.tolist()):
            key = (min(int(a[y, x]), int(b[y, x])), max(int(a[y, x]), int(b[y, x])))
            entry = seams.setdefault(key, {"excess": [], "mask": None})
            entry["excess"].append(float(excess[y, x]))
            entry.setdefault("flat", []).append(float(step_o[y, x]))
            entry.setdefault("quiet", []).append(bool(quiet[y, x]))
            entry.setdefault("residual", []).append(float(residual[y, x]))
            entry.setdefault("where", []).append((y, x))
            if entry["mask"] is None:
                entry["mask"] = np.zeros((height, width), np.uint8)
            if (excess[y, x] > step_threshold and step_o[y, x] < flat_step
                    and quiet[y, x]):
                entry["candidate"] = entry.get("candidate", [])
                entry["candidate"].append((y, x, float(residual[y, x])))

    rows = []
    for (i, j), entry in seams.items():
        values = np.array(entry["excess"], np.float32)
        flat = np.array(entry["flat"], np.float32) < flat_step
        flat &= np.array(entry["quiet"], bool)
        if values.size < scale_length(8, original.shape):
            continue
        # A line has a direction: only the pixels agreeing with the boundary's
        # dominant sign are part of it.
        mask = entry["mask"]
        candidates = entry.get("candidate", [])
        if candidates:
            signs = np.array([c[2] for c in candidates], np.float32)
            dominant = 1.0 if float(np.median(signs)) >= 0 else -1.0
            for y, x, value in candidates:
                if (value >= 0) == (dominant >= 0):
                    mask[y, x] = 1
        # The longest contiguous stretch of boundary that steps too far. This is
        # the figure the eye responds to; the mean is what an area metric sees.
        #
        # Closed first, because the eye integrates along a line and this measure
        # should too: a seam sitting at 1 to 2 luma dips below any threshold
        # every few pixels, and counting raw runs reported 11 px for a line that
        # is plainly 150 long. One gap does not make two lines.
        join_radius = scale_length(1, original.shape)
        join_kernel = np.ones((2 * join_radius + 1,) * 2, np.uint8)
        joined = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, join_kernel,
            iterations=scale_length(2, original.shape),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        longest = int(max((stats[k, cv2.CC_STAT_AREA] for k in range(1, count)), default=0))
        band_mask = cv2.dilate(joined, np.ones((2 * band + 1,) * 2, np.uint8)) > 0
        band_mask &= subject
        residual = (float(np.abs(_luma(composite[band_mask]) - _luma(original[:, :, :3][band_mask])).mean())
                    if band_mask.any() else 0.0)
        rows.append({
            "pair": f"{tags[i]} | {tags[j]}",
            "boundary_px": int(values.size),
            "flat_px": int(flat.sum()),
            "mean_excess": round(float(values[flat].mean()) if flat.any() else 0.0, 2),
            "p95_excess": round(float(np.percentile(values[flat], 95)) if flat.any() else 0.0, 2),
            "max_excess": round(float(values[flat].max()) if flat.any() else 0.0, 2),
            "over_threshold_px": int(mask.sum()),
            "longest_run_px": longest,
            "band_residual": round(residual, 2),
        })
    rows.sort(key=lambda r: (-r["longest_run_px"], -r["mean_excess"]))
    return {"run": run,
            "step_threshold": step_threshold,
            "quiet_energy": round(quiet_energy, 1), "seams": rows}




def seam_report_layers(original_rgba: np.ndarray,
                       layer_dict: dict[str, np.ndarray], *,
                       run: str = "portrait-bundle",
                       step_threshold: float = STEP_THRESHOLD,
                       flat_step: float = FLAT_STEP,
                       quiet_energy: float = QUIET_ENERGY,
                       band: int | None = None) -> dict[str, Any]:
    """Measure static seams directly at the Portrait Bundle seam."""
    original = np.asarray(original_rgba)
    composite, _, owner, tags = _render_layers(layer_dict, original.shape[:2])
    return _measure_seams(
        original, composite, owner, tags, run=run,
        step_threshold=step_threshold, flat_step=flat_step,
        quiet_energy=quiet_energy, band=band,
    )


def compare_seam_report(report: dict[str, Any], baseline: dict[str, Any], *,
                        run_slack: int = RUN_SLACK_PX,
                        excess_slack: float = EXCESS_SLACK) -> tuple[bool, list[str]]:
    """Compare two array-based reports for regression gating."""
    before = {row["pair"]: row for row in baseline.get("seams", [])}
    complaints = []
    for row in report.get("seams", []):
        was = before.get(row["pair"])
        if was is None:
            if row["longest_run_px"] > run_slack:
                complaints.append(f"{row['pair']}: new seam, {row['longest_run_px']} px")
            continue
        if row["longest_run_px"] > was["longest_run_px"] + run_slack:
            complaints.append(
                f"{row['pair']}: {was['longest_run_px']} -> "
                f"{row['longest_run_px']} px of continuous seam"
            )
        if row["mean_excess"] > was["mean_excess"] + excess_slack:
            complaints.append(
                f"{row['pair']}: mean excess {was['mean_excess']} -> {row['mean_excess']}"
            )
    return not complaints, complaints
