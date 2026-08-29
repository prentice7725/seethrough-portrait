"""Seam guard: find the thin, long artifacts the composite metrics cannot see.

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
ranked by the longest run, not by the mean.

    python -m seethrough_engine.seams <run dir> [<run dir> ...]
    python -m seethrough_engine.seams --check <run dir>

`--check` exits non-zero when a seam is worse than the thresholds below, which
are set from what the pipeline currently achieves rather than from taste: see
`docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md` for the A-001 baseline they came
from.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "STEP_THRESHOLD",
    "FLAT_STEP",
    "seam_report",
    "check_run",
    "load_baseline",
    "write_baseline",
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


def _run_manifest(run_dir):
    names = [f for f in os.listdir(run_dir) if f.endswith("_rig_manifest.json")]
    if not names:
        raise FileNotFoundError(f"no *_rig_manifest.json in {run_dir}")
    with open(os.path.join(run_dir, names[0]), encoding="utf-8") as f:
        return names[0][: -len("_rig_manifest.json")], json.load(f)


def _render(run_dir, manifest):
    """The rig's parts at rest, back to front, with the topmost owner per pixel.

    Expression parts are left out: they are an overlay that may or may not be
    attached, and a guard has to measure the same thing every time.
    """
    height = manifest["canvas"]["height"]
    width = manifest["canvas"]["width"]
    rgb = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width, 1), np.float32)
    owner = np.full((height, width), -1, np.int16)
    parts = sorted(manifest["parts"], key=lambda p: p["z"])
    for index, part in enumerate(parts):
        image = np.array(Image.open(os.path.join(run_dir, *part["image"].split("/")))
                         .convert("RGBA")).astype(np.float32)
        x1, y1, x2, y2 = part["xyxy"]
        a = image[:, :, 3:4] / 255.0
        window = rgb[y1:y2, x1:x2]
        window[:] = window * (1.0 - a) + image[:, :, :3] * a
        alpha[y1:y2, x1:x2] = np.clip(alpha[y1:y2, x1:x2] + a, 0.0, 1.0)
        owner[y1:y2, x1:x2][image[:, :, 3] > 128] = index
    return rgb, alpha[:, :, 0], owner, [p["tag"] for p in parts]


def seam_report(run_dir: str, *, step_threshold: float = STEP_THRESHOLD,
                flat_step: float = FLAT_STEP, band: int = BAND_PX) -> dict[str, Any]:
    """Every boundary between two layers, and how much of it is a line."""
    base_name, manifest = _run_manifest(run_dir)
    composite, _, owner, tags = _render(run_dir, manifest)
    original = np.array(Image.open(os.path.join(run_dir, f"{base_name}_original.png"))
                        .convert("RGBA"))
    subject = original[:, :, 3] > 128
    lc, lo = _luma(composite), _luma(original[:, :, :3])

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
            if entry["mask"] is None:
                entry["mask"] = np.zeros((height, width), np.uint8)
            if excess[y, x] > step_threshold and step_o[y, x] < flat_step:
                entry["mask"][y, x] = 1

    rows = []
    for (i, j), entry in seams.items():
        values = np.array(entry["excess"], np.float32)
        flat = np.array(entry["flat"], np.float32) < flat_step
        if values.size < 8:
            continue
        mask = entry["mask"]
        # The longest contiguous stretch of boundary that steps too far. This is
        # the figure the eye responds to; the mean is what an area metric sees.
        #
        # Closed first, because the eye integrates along a line and this measure
        # should too: a seam sitting at 1 to 2 luma dips below any threshold
        # every few pixels, and counting raw runs reported 11 px for a line that
        # is plainly 150 long. One gap does not make two lines.
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                                  iterations=2)
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
    return {"run": os.path.basename(os.path.normpath(run_dir)),
            "step_threshold": step_threshold, "seams": rows}


def check_run(run_dir: str, baseline: dict[str, Any], *,
              run_slack: int = RUN_SLACK_PX,
              excess_slack: float = EXCESS_SLACK) -> tuple[bool, list[str]]:
    """`(passed, complaints)` against a recorded baseline.

    A seam that appears where the baseline had none is a complaint too: the
    usual way to make one boundary better is to move the fault to the next one.
    """
    report = seam_report(run_dir)
    before = {row["pair"]: row for row in baseline.get("seams", [])}
    complaints = []
    for row in report["seams"]:
        was = before.get(row["pair"])
        if was is None:
            if row["longest_run_px"] > run_slack:
                complaints.append(
                    f"{row['pair']}: new seam, {row['longest_run_px']} px")
            continue
        if row["longest_run_px"] > was["longest_run_px"] + run_slack:
            complaints.append(
                f"{row['pair']}: {was['longest_run_px']} -> "
                f"{row['longest_run_px']} px of continuous seam")
        if row["mean_excess"] > was["mean_excess"] + excess_slack:
            complaints.append(
                f"{row['pair']}: mean excess {was['mean_excess']} -> "
                f"{row['mean_excess']}")
    return not complaints, complaints


def load_baseline(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_baseline(path: str, run_dirs) -> dict[str, Any]:
    """Record what the pipeline achieves today, for future changes to beat."""
    baseline = {os.path.basename(os.path.normpath(d)): seam_report(d) for d in run_dirs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def _print(report, limit=8):
    print(f"run {report['run']}   (counted where the original is flat and the "
          f"composite steps {report['step_threshold']}+ luma more)")
    print(f"  {'pair':34s} {'bound':>6s} {'flat':>6s} {'mean':>6s} {'p95':>6s} "
          f"{'max':>6s} {'over':>5s} {'longest':>8s} {'band mae':>9s}")
    for row in report["seams"][:limit]:
        print(f"  {row['pair']:34s} {row['boundary_px']:6d} {row['flat_px']:6d} "
              f"{row['mean_excess']:6.2f} {row['p95_excess']:6.2f} {row['max_excess']:6.2f} "
              f"{row['over_threshold_px']:5d} {row['longest_run_px']:8d} "
              f"{row['band_residual']:9.2f}")


DEFAULT_BASELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "docs", "seam_baseline.json")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    record = "--record" in argv
    path = DEFAULT_BASELINE
    if "--baseline" in argv:
        path = argv[argv.index("--baseline") + 1]
        argv = [a for a in argv if a != path]
    runs = [a for a in argv if not a.startswith("--")]
    if not runs:
        print(__doc__, file=sys.stderr)
        return 2

    if record:
        write_baseline(path, runs)
        print(f"recorded {len(runs)} run(s) to {path}")
        return 0

    if check:
        if not os.path.isfile(path):
            print(f"no baseline at {path}; record one with --record", file=sys.stderr)
            return 2
        baseline = load_baseline(path)
        failed = False
        for run_dir in runs:
            name = os.path.basename(os.path.normpath(run_dir))
            if name not in baseline:
                print(f"{name}: not in the baseline, nothing to compare against")
                continue
            passed, complaints = check_run(run_dir, baseline[name])
            print(f"{name}: " + ("seams ok" if passed else "SEAM GUARD"))
            for line in complaints:
                print(f"  {line}")
            failed |= not passed
        return 1 if failed else 0

    for run_dir in runs:
        _print(seam_report(run_dir))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
