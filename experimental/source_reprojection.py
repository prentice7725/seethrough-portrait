"""Source Pixel Lock -- an offline A/B experiment, not a pipeline stage.

`SEETHROUGH_PORTRAIT_RESEARCH_ABSORPTION_PLAN_v0.1` section 4 asks a
narrow question: where a canonical layer's visible pixel already sits on top
of the original still, why regenerate it? This module answers that for a
*saved* Portrait Bundle and reports whether the answer would have helped --
it does not touch the canonical pipeline, and nothing in `seethrough_engine`
imports it (plan section 4.4: "기존 canonical pipeline에 즉시 넣지 않는다").

Three regions, exactly as specified (section 4.3):

    A. Opaque visible core (layer alpha ~= 1, this layer is the topmost
       owner): the original pixel *is* the answer. `layer.rgb = original.rgb`.
    B. Semi-transparent edge (hair AA, colour fringe, soft boundaries): a
       straight per-pixel solve, `F = (O - (1-a)B) / a`, guarded and only
       accepted where it does not regress the local composite.
    C. Fully hidden region (alpha effectively zero here): left exactly as
       See-through generated it. Nothing in this module ever writes there.

Only RGB changes. Alpha is never touched by this experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from seethrough_engine.image import composite_fidelity, composite_layers
from seethrough_engine.local_fidelity import local_fidelity_report
from seethrough_engine.seams import seam_report_layers
from seethrough_engine.semantic import SEMANTIC_Z_ORDER

__all__ = [
    "SOURCE_LOCK_VERSION",
    "SourceLockResult",
    "lock_source_pixels",
    "evaluate_source_lock",
    "load_bundle_layers",
    "run_experiment",
]

SOURCE_LOCK_VERSION = "0.1-experimental"

# Region A: how close to fully opaque a layer must be before its visible
# pixel is simply replaced outright, no solve involved.
OPAQUE_ALPHA = 0.98
# Region C boundary: below this a pixel carries no meaningful contribution
# from this layer and is left as generated content untouched.
MIN_ALPHA = 0.06
# Guard: `F = (O - (1-a)B) / a` is only trustworthy where the background it
# is being un-mixed from actually differs from the observed pixel -- this is
# the same "solve is noise below this" reasoning as `repair.EDGE_MIN_CONTRAST`,
# applied to (original - behind) rather than (front - behind).
MIN_BEHIND_CONTRAST = 15.0
# Guard: a solved colour landing this far outside the sRGB range means the
# per-pixel algebra is not well-conditioned here (behind and alpha disagree
# with what the original shows); reject rather than clip blindly.
GAMUT_SLACK = 40.0


@dataclass(frozen=True)
class SourceLockResult:
    layers: dict[str, np.ndarray]
    source_locked_mask: np.ndarray
    source_solved_edge_mask: np.ndarray
    generated_hidden_mask: np.ndarray
    changed_px: dict[str, int]


def lock_source_pixels(
    layer_dict: dict[str, np.ndarray],
    original_rgba: np.ndarray,
    *,
    order: tuple[str, ...] = SEMANTIC_Z_ORDER,
    opaque_alpha: float = OPAQUE_ALPHA,
    min_alpha: float = MIN_ALPHA,
    min_behind_contrast: float = MIN_BEHIND_CONTRAST,
    gamut_slack: float = GAMUT_SLACK,
) -> SourceLockResult:
    """Replace/solve visible-region RGB from the original; leave alpha and
    fully hidden pixels untouched. See module docstring for the three regions.
    """
    original_rgb = np.asarray(original_rgba)[..., :3].astype(np.float32)
    shape = original_rgb.shape[:2]

    def rank(tag: str) -> int:
        return order.index(tag) if tag in order else -1

    tags = sorted(layer_dict, key=rank)
    out: dict[str, np.ndarray] = {}
    changed_px: dict[str, int] = {}
    locked_total = np.zeros(shape, bool)
    solved_total = np.zeros(shape, bool)
    hidden_total = np.zeros(shape, bool)

    # A layer's own alpha alone does not say whether it is what the eye
    # actually sees: an opaque `neck` can still sit entirely under a
    # semi-transparent `front_hair` fringe, and the original pixel there is a
    # *blend* of both, not neck's true colour. So before touching anything,
    # measure how much is drawn in front of each layer -- the same cumulative
    # coverage a standard front-to-back "over" walk produces -- and only ever
    # correct a layer where it is close to the plan's literal "topmost owner"
    # (section 4.3 region A): negligible front coverage. Where it is not, the
    # pixel correctly falls through to region C (left as generated) rather
    # than being corrected against a blend it is not responsible for.
    front_alpha_of: dict[str, np.ndarray] = {}
    front_acc = np.zeros(shape, np.float32)
    for tag in reversed(tags):
        front_alpha_of[tag] = front_acc
        layer_a = np.asarray(layer_dict[tag])[..., 3].astype(np.float32) / 255.0
        front_acc = front_acc + layer_a * (1.0 - front_acc)

    beneath_rgb = np.zeros((*shape, 3), np.float32)
    beneath_a = np.zeros(shape, np.float32)

    for tag in tags:
        layer = np.asarray(layer_dict[tag])
        a = layer[..., 3].astype(np.float32) / 255.0
        rgb = layer[..., :3].astype(np.float32)
        uncontested = front_alpha_of[tag] < (1.0 - opaque_alpha)

        locked = uncontested & (a >= opaque_alpha)
        out_rgb = rgb.copy()
        out_rgb[locked] = original_rgb[locked]

        candidate = uncontested & (a >= min_alpha) & ~locked
        behind_present = beneath_a > 0.02
        denom = np.maximum(a, 1e-6)[..., None]
        solved = (original_rgb - (1.0 - a)[..., None] * beneath_rgb) / denom
        contrast = np.abs(original_rgb - beneath_rgb).sum(axis=-1)
        in_gamut = np.all((solved >= -gamut_slack) & (solved <= 255.0 + gamut_slack), axis=-1)
        eligible = candidate & behind_present & (contrast >= min_behind_contrast) & in_gamut

        solved_clipped = np.clip(solved, 0.0, 255.0)
        composite_with_solved = solved_clipped * a[..., None] + beneath_rgb * (1.0 - a)[..., None]
        composite_with_generated = rgb * a[..., None] + beneath_rgb * (1.0 - a)[..., None]
        err_solved = np.abs(original_rgb - composite_with_solved).sum(axis=-1)
        err_generated = np.abs(original_rgb - composite_with_generated).sum(axis=-1)
        accept = eligible & (err_solved <= err_generated)
        out_rgb[accept] = solved_clipped[accept]

        changed = locked | accept
        changed_px[tag] = int(changed.sum())
        locked_total |= locked
        solved_total |= accept
        hidden_total |= (a >= min_alpha) & ~locked & ~accept

        patched = np.array(layer, copy=True)
        patched[..., :3] = np.rint(np.clip(out_rgb, 0, 255)).astype(np.uint8)
        out[tag] = patched

        beneath_rgb = beneath_rgb * (1.0 - a)[..., None] + out_rgb * a[..., None]
        beneath_a = np.clip(beneath_a + a, 0.0, 1.0)

    return SourceLockResult(
        layers=out,
        source_locked_mask=locked_total,
        source_solved_edge_mask=solved_total,
        generated_hidden_mask=hidden_total,
        changed_px=changed_px,
    )


def _preservation_ratio(before: dict[str, np.ndarray], after: dict[str, np.ndarray],
                        mask: np.ndarray) -> float:
    """Fraction of `mask` where every layer's RGB is unchanged before/after."""
    if not mask.any():
        return 1.0
    identical = np.ones(mask.shape, bool)
    for tag in before:
        b = np.asarray(before[tag])[..., :3].astype(np.int16)
        a = np.asarray(after.get(tag, before[tag]))[..., :3].astype(np.int16)
        identical &= np.all(b == a, axis=-1)
    return float(identical[mask].mean())


def evaluate_source_lock(
    original_rgba: np.ndarray,
    before_layers: dict[str, np.ndarray],
    result: SourceLockResult,
    *,
    subject_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """The plan section 4.4 A/B comparison, computed from arrays only."""
    original = np.asarray(original_rgba)
    shape = original.shape[:2]
    after_layers = result.layers
    mask = subject_mask if subject_mask is not None else (original[..., 3] > 10)

    before_composite = composite_layers(before_layers, shape)
    after_composite = composite_layers(after_layers, shape)
    before_fidelity = composite_fidelity(original, before_composite, mask)
    after_fidelity = composite_fidelity(original, after_composite, mask)

    before_seams = seam_report_layers(original, before_layers, run="before")
    after_seams = seam_report_layers(original, after_layers, run="after")
    longest = lambda report: max(
        (int(row["longest_run_px"]) for row in report["seams"]), default=0)

    before_local = local_fidelity_report(original, before_composite, before_layers)
    after_local = local_fidelity_report(original, after_composite, after_layers)

    opaque_owned = np.zeros(shape, bool)
    for tag in after_layers:
        opaque_owned |= np.asarray(after_layers[tag])[..., 3] > 200
    visible_error = np.abs(
        original[..., :3].astype(np.int32) - after_composite[..., :3].astype(np.int32)
    ).sum(axis=-1)
    visible_source_preservation_ratio = (
        float((visible_error[opaque_owned] <= 6).mean()) if opaque_owned.any() else 1.0
    )

    total_changed_px = sum(result.changed_px.values())

    return {
        "version": SOURCE_LOCK_VERSION,
        "global_mae": {"before": before_fidelity["mae"], "after": after_fidelity["mae"]},
        "bad_pixel_ratio": {"before": before_fidelity["bad_ratio"], "after": after_fidelity["bad_ratio"]},
        "seam_longest_run_px": {"before": longest(before_seams), "after": longest(after_seams)},
        "eye_local_fidelity_status": {"before": before_local["status"], "after": after_local["status"]},
        "mouth_local_fidelity_status": {
            "before": (before_local["mouth"] or {}).get("status", "unavailable"),
            "after": (after_local["mouth"] or {}).get("status", "unavailable"),
        },
        # Source Pixel Lock never touches alpha; tracked explicitly so an A/B
        # reader does not mistake silence for "measured and found zero".
        "alpha_edge_error": "not_applicable_rgb_only",
        "changed_px_by_layer": result.changed_px,
        "changed_px_total": total_changed_px,
        "visible_source_preservation_ratio": round(visible_source_preservation_ratio, 6),
        "hidden_pixel_preservation_ratio": round(
            _preservation_ratio(before_layers, after_layers, result.generated_hidden_mask), 6),
        "source_locked_px": int(result.source_locked_mask.sum()),
        "source_solved_edge_px": int(result.source_solved_edge_mask.sum()),
        "generated_hidden_px": int(result.generated_hidden_mask.sum()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / Path(*relative.split("/"))).resolve()
    path.relative_to(root.resolve())  # raises ValueError if it escapes root
    return path


def _rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def load_bundle_layers(bundle_dir: str | os.PathLike[str]
                       ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read `original.png` + canonical `layers/` from a saved Portrait Bundle.

    Manifest-driven and path-validated, mirroring the defensive read pattern
    `portrait-autorig`'s bundle loader uses -- reimplemented locally rather
    than imported, since an experimental script here must not depend on a
    downstream consumer.
    """
    root = Path(bundle_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("format") != "portrait-bundle":
        raise ValueError(f"not a Portrait Bundle: {root}")
    original = _rgba(_inside(root, manifest["original"]))
    layers = {
        tag: _rgba(_inside(root, entry["path"]))
        for tag, entry in (manifest.get("layers") or {}).items()
    }
    return original, layers


def _save_mask(path: Path, mask: np.ndarray) -> None:
    u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(u8, mode="L").save(path)


def run_experiment(bundle_dir: str | os.PathLike[str],
                   output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Run the A/B pass on a saved bundle; never writes inside `bundle_dir`."""
    original, before_layers = load_bundle_layers(bundle_dir)
    result = lock_source_pixels(before_layers, original)
    report = evaluate_source_lock(original, before_layers, result)

    out_root = Path(output_dir).resolve()
    diagnostics = out_root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    with (diagnostics / "source_reprojection.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    _save_mask(diagnostics / "source_locked_mask.png", result.source_locked_mask)
    _save_mask(diagnostics / "source_solved_edge_mask.png", result.source_solved_edge_mask)
    _save_mask(diagnostics / "generated_hidden_mask.png", result.generated_hidden_mask)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", help="Path to a saved Portrait Bundle directory")
    parser.add_argument("output_dir", help="Where to write the A/B diagnostics (never bundle_dir)")
    args = parser.parse_args()
    report = run_experiment(args.bundle_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
