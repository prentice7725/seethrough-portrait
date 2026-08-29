# PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1

## Goal

Answer M4 of `PORTRAIT_MODE_FORK_PLAN_v0.1.md` -- "at least two of idle,
blink, mouth, or head motion work" -- by rigging A-001's exported layers
automatically and animating them in a browser, with no Spine, no Live2D, and
no bone hierarchy.

The deliverable is a *decision*, not a product: whether Portrait Mode's
semantic layers are good enough to drive a pseudo-2.5D talking portrait. The
fork plan's Phase 2 is explicitly feasibility-only, so anything that would
turn this into production auto-rigging is out of scope (see "Out of scope").

## Why not Spine auto-bone

The existing Spine exporter (`seethrough_engine/spine.py`) emits one bone --
`root` -- and binds every slot to it, so a rig test means hand-building bones
in the Spine editor. Doing that surfaced two problems a bone hierarchy does
not solve well:

1. **Ghost silhouette.** `body_remainder` is a single canvas-wide layer
   pinned behind everything. Rotating a hand-made head bone leaves the
   recovered pixels *around the head* sitting at the original position.
2. **The neck.** A head bone moves the neck rigidly or not at all. Neither
   looks right, and a second `bone_neck` only moves the seam.

Both have the same shape of answer: **per-vertex weights, not bones.** A neck
whose top follows the head and whose bottom stays with the body is a two-line
gradient in a vertex shader and a modelling problem in a bone rig.

Spine export is not being removed -- it stays as one export target. Its own
fix (emit `bone_head` with rotation 0, pivoted at `neck_pivot`) falls out of
Stage C below for free, and is scheduled after this milestone.

## Prior art

Two projects were read for algorithms. Neither is vendored or taken as a
dependency: our input is an in-memory RGBA layer dict with known semantic
tags, not a PSD with guessed layer names, so adapting our data to their
pipelines is more work than reimplementing the parts we want.

| Source | Taken | Rejected |
| --- | --- | --- |
| Anime2.5DRig (MIT) | neck vertex-weight gradient; depth-differential pseudo-turn; connected-component eye L/R split; anchor detection from `face`/`neck` alpha; grid mesh; blink by Y-compression | name-guessed head/body classification (we have real tags); fixed name-based depth as the *only* source; generic eyelid synthesis (deferred); `mouth` to `mouth_open` remapping (wrong for us, see Stage A) |
| PNGAL (Apache-2.0) | rig config stored as a separate JSON beside the layers, so motion can be re-tuned without re-running decomposition; sprite-sheet + JSON bake as a game-delivery Plan B | RIFE in-betweening; ComfyUI/Qwen/portable-Python stack (45-90 GiB, 12 GB+ VRAM -- this fork deliberately targets 8 GB) |
| Anime2.5DRig, izumix77 fork (MIT) | ellipsoid head/hair shell projection, blended difference-only against the parallax path (Stage E, H6); a pinned region as a *shape* rather than a number, for a weight that has to hold a seam | its input assumption (a hand-separated PSD, layers guessed by name); the profile and chest curves, which are per-character tuning rather than feasibility; the body cylinder |
| PachiPakuGen (see "Candidates" in the fork plan) | nothing yet: its answers are workflow, not algorithm | RIFE in-betweening, for the same reason as PNGAL |

Constants and coefficients quoted from any of them are starting values only,
expected to be re-tuned against A-001. See "Algorithm references".

Worth recording what the prior art does **not** contain, since it is where most
of this milestone's work went: none of these projects has a contested-pixel
problem (`reclaim_occluded`), a recovered-remainder layer to place, or a
continuous neck weight. Their input is a PSD an artist already separated, so
those questions cannot arise. Anime2.5DRig still carries the neck constant this
rig started from -- 0.55 against a head at 1.00, the 0.45 jaw step -- which is
the clearest evidence that the seam problems belong to *automatic*
decomposition rather than to 2.5D rigging.

## Input contract

The rig consumes what `seethrough_engine.export.save_portrait_run` already
produces, plus the in-memory `PortraitResult`:

- `layer_dict`: `{tag: full-canvas RGBA}`, tags from `VALID_BODY_PARTS_V3_*`
  (`seethrough_engine/layers.py`). **A-001 is v3**, which matters twice:
  - there is no `eyes` tag; `irides`, `eyewhite`, `eyelash`, and `eyebrow`
    each contain *both* eyes in one layer, so Stage B has to split them;
  - there is a `head` tag, which the depth pass does not cover.
- `guard.body_remainder`: canvas-wide RGBA of Silhouette-Guard-recovered
  pixels.
- `depth_dict` (optional): per-layer Marigold depth, only when the caller ran
  the depth pass.

## Stages

### Stage A -- semantic normalization

Rig core keys on the **pre-rename semantic tags**, not the Spine-friendly
names (`back hair`, not `back-hair`). `DEFAULT_SPINE_NAMES` stays where it
is, applied only by the Spine exporter.

`mouth` is **not** renamed to `mouth_open`. A-001's `mouth` is the neutral,
closed mouth; treating it as an open mouth inverts every mouth animation.
v0.1 animates it as `mouth_base` with a small Y scale (a lip flap). A real
open mouth needs a separate synthesized layer, which is out of scope.

### Stage B -- region-aware remainder split, and eye split

**Remainder split.** `body_remainder` is partitioned by nearest owner rather
than by bounding box, so hair falling over a shoulder splits along the actual
boundary instead of a horizontal cut:

```
head_union = union of HEAD-group layer alpha
body_union = union of BODY-group layer alpha
d_head = distanceTransform(~head_union),  d_body = distanceTransform(~body_union)
head_remainder = remainder where d_head <= d_body
body_remainder = remainder elsewhere
neck_remainder = remainder inside the neck bbox band, carved out first
```

This lives in `seethrough_engine`, **not** `portrait_core`: `portrait_core`
is numpy-only by rule and `distanceTransform` is cv2. The Silhouette Guard's
scoring keeps seeing the single undivided remainder; only the rig splits it.

**Eye split.** For each of `irides`, `eyewhite`, `eyelash`, `eyebrow`:
connected components on alpha, drop components below a noise-area threshold,
assign by component centroid X against `face_center.x`, dilate each side's
mask slightly. No face-detection model.

**Contested pixels.** A garment layer sometimes comes back holding the skin
visible through its own opening -- opaque, and in front of the neck, so it
paints flat light skin over the neck's shading and hides the neck almost
entirely: `neck` visible on 272 of its 12096 pixels behind a V-neck, 117 of
7533 behind a stand collar.

Where a garment and the neck both claim a pixel, the decomposition has already
been told which is right: the original image is ground truth here, and one of
the two layers matches it. `reclaim_occluded` clears the garment's alpha
wherever the neck explains the original better by a margin, keeping only
coherent regions. It is checked per pixel rather than assumed for a region, and
the result is measurable rather than a matter of taste -- on A-001 it improved
the composite as well as the rig:

| | reclaimed | composite mae | visible neck |
| --- | --- | --- | --- |
| V-neck cardigan | 2700 px | 18.74 -> 17.54 | 272 -> 2850 px |
| gakuran stand collar | 1764 px | 15.08 -> 14.77 | 117 -> 1196 px |

Only `topwear`/`neckwear` over `neck` is treated as contested. Every other
pairing tried either made the composite worse (`topwear` over `face`, `head`
over `face`) or moved tens of thousands of pixels for no measurable gain
(`back hair` over `head`, 21k pixels) -- which is the more dangerous outcome,
since a large silent reassignment is exactly what nobody notices until it is
wrong. The manifest records what moved.

### Stage C -- anchors

Derived from layer alpha only:

```
face_center  = centroid of face alpha
eye_left     = centroid of the left-side eyewhite components
eye_right    = centroid of the right-side eyewhite components
mouth        = centroid of mouth alpha
neck_pivot   = (neck bbox center X, neck_top + neck_height * 0.85)
body_pivot   = (topwear bbox center X, topwear bbox bottom)
```

`neck_pivot` sits near the *bottom* of the neck: that is the rotation centre
that makes a head tilt read as a neck bending rather than a head sliding. It
is also the value the Spine exporter needs to place `bone_head`.

### Stage D -- groups, depth, and movement weights

Group assignment is by tag table, not by centroid heuristics:

```
HEAD   back hair, head, ears, earwear, face, eyewhite, irides, eyebrow,
       eyelash, eyewear, nose, mouth, headwear, front hair, head_remainder
NECK   neck, neckwear, neck_remainder
BODY   topwear, handwear, bottomwear, objects, tail, wings, body_remainder
```

Weights (starting values, to be tuned):

| Group | Head-follow weight |
| --- | --- |
| HEAD | 1.00 |
| NECK | gradient, `HEAD_WEIGHT` at the top of the neck bbox to `BODY_WEIGHT` at the bottom |
| BODY | 0.16, except a collar (see below) |

The neck's endpoints are **derived, not chosen**: they have to equal the head
and body weights exactly or the deformation steps at the jaw or at the collar.
The first implementation borrowed 0.55 for the neck top against a head at 1.00
and put a 0.45 discontinuity right where the two meet -- with a stand collar
hiding three quarters of the neck, that jaw step was the only part visible.

A garment whose top edge overlaps the neck is a collar, and takes **the neck's
own weight function** over that overlap. A stand collar touches the jaw, so
leaving it rigid reads as the chin cutting into it -- but the reason it shares
the neck's gradient rather than getting one of its own is stronger than that:
`reclaim_occluded` cuts a window in the garment for the neck to show through,
so the window and its contents are two sides of one seam. Give them different
weights and the window's edge slices the neck as the head turns. An independent
collar constant of 0.45 against a neck at 0.571 on the collar line opened a
2.05 px crack there; sharing the function leaves 0.43 px, which is only what
the two parts' differing depths contribute.

The shoulders do not come along, because the neck's gradient reaches
`BODY_WEIGHT` at the bottom of the neck -- which is anatomically where the
shoulders start to broaden. Measured on A-001: the garment is 22 px wide at the
collar line where the weight is 0.571, and 260 px wide by the time the weight
is back to 0.161.

**Open risk:** `back hair` frequently extends below the shoulder line, and a
flat 1.00 will tear it away from the body there. If the preview shows that,
`back hair` gets the same `gradient_y` treatment as the neck -- or, if a
horizontal ramp turns out to cut across the strands rather than along them, the
izumix77 fork's pinned *region*, which is the same idea given a shape instead of
a scalar: an area that follows one owner exactly, feathered only on its outside
so the boundary cannot move when the region is retuned. This is the one
weight most likely to need a shape change rather than a number change, so it
is already wired as a toggle rather than a code edit: `build_rig`'s
`gradient_tags` argument, surfaced in the webui as "Rig: soften back hair".

**Depth** for pseudo-turn parallax comes from a fixed per-tag table derived
from `SEMANTIC_Z_ORDER` (`seethrough_engine/spine.py`), which is already a
canonical back-to-front ordering for this fixed tag vocabulary. Marigold
`depth_median`, when present, **overrides** the table. It is not the default:
it costs a 3 GB model and an extra pass, and per-layer parallax does not need
accuracy the table cannot supply.

### Stage E -- motions

Exactly four, matching the fork plan's M4 wording:

1. **Head pseudo-turn (X/Y).** Per-layer offset scaled by that layer's depth,
   producing parallax between `back hair`, `face`, `nose`, `front hair`.
   Blended, by a slider that is 0 by default, against an **ellipsoid shell**
   (H6): every vertex is lifted onto a head-shaped surface fitted to the `face`
   box, rotated in yaw/pitch, and reprojected. Hair rides a shell 10% larger so
   bangs float above the forehead. The shell contributes only the *difference*
   between the rotated and the rest projection, so at slider 0 -- and at any
   slider value with the head at rest -- the render is bit-identical to the
   parallax rig, and the slider isolates the turn's shape from its amount.
2. **Head tilt (Z), +/-2 degrees**, rotating about `neck_pivot`. Unlike the
   turn it cannot open a seam: it rotates about the pivot scaled by the same
   weight field the rig is built on, so parts sharing a weight turn rigidly
   together and the single gradient stays continuous. That asymmetry is why
   idle caps the turn and not the tilt.
3. **Breathing.** One continuous vertical displacement field, not a per-group
   transform: full lift above the chest line, falling linearly to zero at
   `body_pivot`, plus a slight ribcage widening. Giving the head a translation
   and the torso a scale -- the first attempt -- made the two disagree at the
   collar and the neck visibly stretched. Every part reads the same field, so
   the seams cannot drift apart.
4. **Blink.** Y-compression of `eyewhite` / `irides` / `eyelash` toward the
   eye centre line, but **the lash keeps a fraction of its height** and lands
   on the lid line while the white and iris vanish behind it. Compressing
   everything to zero deletes the eye outright: a mesh of zero height draws
   nothing, and the `face` layer underneath is inpainted skin with no eye on
   it -- correctly so, which is the same property H1 depends on -- so a fully
   collapsed eye leaves a blank cheek. A shut anime eye is a dark line, and the
   lash is the only layer that can be it without synthesizing a closed-eye
   layer, which stays out of scope.

Mesh is a uniform grid per part (~42 px cells at 768), vertex positions
recomputed per frame. No mesh fitting.

## `rig_manifest.json`

The contract between the Python rig core and the JS runtime. Written as
`{base}_rig_manifest.json` in the run directory beside `portrait_report.json`,
with the part PNGs it names under `rig/images/`, so motion can be re-tuned by
editing one file without re-running decomposition -- the one structural idea
taken from PNGAL. Part **names** replace spaces with underscores
(`back hair` -> `back_hair`) since they become file names; the semantic `tag`
is carried alongside and is what the group, depth, and weight tables key on.

**Coordinates are canvas pixels, top-left origin, Y down** -- the same
convention as `xyxy` in `layers_to_parts`. This is deliberately *not* Spine's
bottom-centre Y-up convention; the Spine exporter converts at its own
boundary and nothing else should have to know about that.

```jsonc
{
  "version": "0.1",
  "canvas": { "width": 1280, "height": 1280 },
  "source": { "run_id": "...", "tag_version": "v3", "depth": "table" },
  "anchors": {
    "face_center": [x, y], "eye_left": [x, y], "eye_right": [x, y],
    "mouth": [x, y], "neck_pivot": [x, y], "body_pivot": [x, y]
  },
  "parts": [
    {
      "name": "face", "tag": "face", "image": "rig/images/face.png",
      "xyxy": [x1, y1, x2, y2],
      "group": "head", "depth": 0.52, "z": 12,
      "weight": { "mode": "constant", "value": 1.0 },
      "mesh": { "cell": 42 }
    },
    {
      "name": "neck", "tag": "neck", "image": "rig/images/neck.png",
      "xyxy": [x1, y1, x2, y2],
      "group": "neck", "depth": 0.70, "z": 6,
      "weight": { "mode": "gradient_y", "top": 0.55, "bottom": 0.0,
                  "y_top": 512, "y_bottom": 648 },
      "mesh": { "cell": 30 }
    }
  ],
  "motion": {
    "head_turn": { "max_x": 1.0, "max_y": 1.0 },
    "head_tilt": { "max_deg": 2.0, "pivot": "neck_pivot" },
    "breathing": { "period_s": 4.0, "amplitude_px": 3.0 },
    "blink": { "close_s": 0.08, "hold_s": 0.34, "open_s": 0.16,
               "interval_s": [1.6, 5.4] }
  }
}
```

`z` is the back-to-front draw index; `depth` is normalized 0 (near) to 1
(far) and drives parallax only. Weight modes in v0.1: `constant` and
`gradient_y`.

## Implementation status

Stages A-D are implemented in `seethrough_engine/rig.py` (numpy and cv2 only,
no torch) with unit tests in `tests/unit/seethrough_engine/test_rig.py`.
`export.save_portrait_run(export_rig=True)` writes the manifest, and the webui
exposes it as a checkbox, on by default.

Stage E is implemented in `webui/rig_preview/index.html`. Its deformation math
has headless checks in `webui/rig_preview/check_deformation.mjs`
(`node webui/rig_preview/check_deformation.mjs`), which cover the weight
gradient, depth parallax, the shell projection, the H2 and H3 toggles, and the
blink envelope. `webui/rig_preview/measure_disocclusion.mjs <run dir>` answers
H1 and H6 numerically: it loads the page's own `deform`, rasterizes each part's
alpha in z order on the CPU, and counts what a turn exposes. Both run on plain
`node`, with no dependencies and no GPU. What
is *not* verified without a GPU run is the only thing that matters: whether a
real A-001 decomposition looks right when it moves. That is what the
hypotheses below are for.

## Runtime

`webui/rig_preview/index.html` -- a single self-contained page that loads a run
directory's `rig_manifest.json` plus its images and animates them on a WebGL
canvas. The manifest is the only coupling between Python and JS, so the same
page works on any run directory, including ones produced by the ComfyUI graph.

It is **opened from disk, not served**. A directory picker (or a drag-dropped
folder) supplies the run, which means no Gradio static-path configuration, no
server, and an unzipped run animating on a machine that has never seen this
repo. `?manifest=<url>` is accepted as well for anyone already serving a run
over HTTP.

Vertex positions are recomputed on the CPU each frame rather than in a vertex
shader. At a few thousand vertices the cost is irrelevant, and keeping the
deformation as plain arithmetic means it can be stepped through and unit
tested -- which for a feasibility rig is worth more than the frame budget.

Controls map onto the hypotheses rather than onto a character sheet: manual
turn and tilt sliders that reach well past the manifest's limits (H1), a
`head_remainder -> body` toggle that reproduces the Spine ghost (H2), a neck
weight selector with `rigid` and `detached` as the degenerate comparisons
(H3), per-eye wink buttons (H4), and a blend from the parallax rig to the
ellipsoid shell (H6).

## Hypotheses this milestone tests

The point of building it is to answer these:

| # | Hypothesis | How the preview answers it |
| --- | --- | --- |
| H1 | Our layers are occlusion-complete (LayerDiff reconstructs what was hidden), so parallax does not open holes | Push the head turn past where a plain-PSD rig would tear; find our actual usable limit |
| H2 | Splitting the remainder by region removes the ghost silhouette seen in the Spine test | Toggle `head_remainder` between the HEAD and BODY groups and compare |
| H3 | A neck weight gradient beats any two-bone arrangement | Compare the gradient against `top = bottom = 1.0` (rigid) and `= 0.0` (detached) |
| H4 | Connected-component eye splitting is reliable on A-001 | Blink and wink each side independently |
| H5 | The fixed depth table is sufficient and Marigold is only a refinement | Run both, compare parallax quality |
| H6 | The rotation limit H1 found belongs to *rigid depth offsets*, not to the layer set: rotating a shell instead should raise it | Sweep the turn with the shell blend at 0 and 1 and count the revealed region at equal apparent turn |

H1 matters most and is the one we are least sure of: `body_remainder` exists
precisely because generated coverage is imperfect. Its answer sets the
maximum believable head rotation, which is what decides whether any of this
reads as 2.5D at all.

## Field notes

**2026-08-29 -- a REWORK run that was an input problem.** Run
`20260829_123918_34908fb4` reported "Recovery succeeded, but BODY_REMAINDER
dependency is too high" at `recovered_ratio = 0.405`. The cause was not layer
quality: the uploaded PNG was an opaque, pillarboxed picture, so its subject
mask resolved to a filled 516x768 rectangle and the Silhouette Guard recovered
the *background* into `body_remainder`. The semantic layers themselves were
fine -- per-tag areas and bounding boxes matched the passing run
`20260829_023110_37ca4fc3` (`recovered_ratio = 0.014`) almost exactly.

Two consequences:

* `portrait_core.masks` now rejects a source alpha whose foreground fills its
  own bounding box (`alpha.informative_bbox_fill_max`), and the webui rejects
  such an upload before the model loads rather than after the diffusion pass.
  Without that check a provided subject mask was silently ignored, because the
  padding alpha counted as "informative" and short-circuited the fallback.
* **`recovered_ratio` gates this milestone.** `body_remainder` is original
  source pixels with nothing reconstructed behind it (see
  `TEST_PROTOCOL_A001.md`), so wherever it carries the image, H1's premise --
  occlusion-complete layers -- is false by construction. A run's usable
  rotation limit is bounded by its remainder dependency, and H1 should only be
  measured on a run that passes the guard cleanly.

**2026-08-29 -- a PASS run that still cannot be rigged.** Re-running the same
character with a transparent background gave `20260829_125615_01ad9a33`: PASS,
`recovered_ratio = 0.029`, `silhouette_coverage = 0.997`, and a clean 18-part
rig manifest with the eyes split. The `face` layer came back as eyeless skin,
which is the see-through behaviour working and the first real evidence for H1.

But the run generated no `eyewhite` layer at all, and compositing the layers
back to front reproduces the original only to `mae 15.1`, with 9.6% of subject
pixels off by more than 30: the sclera renders as skin. Nothing in the report
caught it, for three compounding reasons -- `eyewhite` is in the profile's
`optional_tags`, every coverage metric is computed on **alpha**, and the
`reconstruction` diagnostic copies the original's RGB and replaces only its
alpha, so it can never disagree with the original.

`export.save_portrait_run` now also writes `{base}_layer_composite.png` and
`{base}_composite_error.png` and records `report["composite"]`
(`mae`, `bad_ratio`), and the webui warns above 3%. Verdict thresholds are
unchanged -- whether a colour-fidelity term belongs in the verdict is a
decision for the fork plan, not a diagnostic.

For M4 this is a second gate alongside `recovered_ratio`: a run can preserve
the silhouette perfectly and still be missing the feature layers a blink needs.

**2026-08-29 -- flat-background keying as a preprocessing step.** The image
model in this workflow does not emit RGBA, so portraits arrive opaque on a
solid background and the alpha rejection above turns every one of them away.
`seethrough_engine/matting.py` keys a flat background into alpha before the
model loads, and the webui does it automatically when the upload has no matte.

Two details decide whether the result is usable:

* **Anti-aliased edges are un-premultiplied, not thresholded.** Every boundary
  pixel is a blend, so a threshold leaves each hair strand ringed with
  background colour -- and hair already accounts for a quarter of the composite
  error. Against synthetic ground truth the recovered alpha is within 0.001 MAE
  and the recovered foreground colour within 0.75 of 255.
* **The background colour is the dominant border cluster, not the median.** A
  bust portrait always runs off the bottom edge, so the border ring reliably
  contains hair and clothing; a median names a colour that is nowhere in the
  picture. Bin counts alone are not enough either -- a noisy background spreads
  across neighbouring bins while a tight artifact (the anti-aliased seam
  between a pillarbox bar and the picture) sits in one and outvotes it -- so
  candidate bins are ranked by how much of the border falls within tolerance.

On the run that originally failed, keying takes the subject's bounding-box fill
from 1.000 to 0.647, which is the range a real silhouette occupies (the passing
run measured 0.642), and `resolve_subject_mask` then accepts it as
`source_alpha` at HIGH confidence.

**2026-08-29 -- H1 answered: the limit is disocclusion, but not the kind
expected.** Sweeping the head turn on A-001 and counting the largest contiguous
region where hair-dark pixels turn skin-light:

| turnX | max slide (back hair vs front hair) | largest revealed region |
| --- | --- | --- |
| 0.20 | 3.1 px | 195 px |
| 0.40 | 6.2 px | 523 px |
| 0.60 | 8.2 px | 718 px |
| 0.80 | 12.3 px | 839 px |
| 1.00 | 20.5 px | **2095 px** |
| 1.50 | 30.8 px | 2966 px |

Up to 0.8 the reveal stays scattered along edges; between 0.8 and 1.0 it merges
into one coherent gash down the temple. `DEFAULT_MOTION.head_turn.max_x` is now
0.8. Scaling idle by that limit was the first answer and not enough of one: the
reveal does not *begin* at 0.8, it merges there, and idle plays unattended. So
idle is capped at 0.3 -- below where 1027 px of scattered edge become one 3275
px gash -- whatever the manifest allows. The manual sliders still reach +/-1.5,
which is what found the edge.

The nature of the failure is worth recording, because it revises H1 rather than
confirming it. Our layers **are** occlusion-complete -- the `face` layer is
eyeless skin, the `head` layer is a full skull-shape -- so parallax never opens
a *hole*. What it does instead is reveal correctly-painted content that should
still be covered: slide `head` 20 px out from behind `back hair` and you see
skin, not background, exactly where hair belongs. Completeness changes the
artifact from a hole into wrongly-exposed content; it does not remove the
limit, and the limit is still set by how far one layer overhangs the next.

Breathing was ruled out by measurement rather than by inspection: its vertical
differential inside any one part is at most 0.86 px across `back hair`'s full
261 px height, and 0.14 px across the face. It cannot produce a visible tear.

**2026-08-29 -- H3 is not answerable by eye, and the blink closed on the wrong
line.** Two findings from driving the preview.

*H3.* The three neck modes do separate numerically -- over the visible neck at
max turn, the gradient displaces 2.4-9.8 px, rigid a flat 13.2 px, detached
0 px, so up to 11 px apart. But the weight multiplies a head transform, so with
the turn and tilt sliders at zero all three are identical, and the visible neck
is small enough that the comparison needs the tilt slider specifically. H3 is
therefore recorded as **not decidable from a still**: the gradient's real
justification is continuity at the two seams, which is verified by construction
(the endpoints are the head and body weights) rather than by eye.

*Blink.* The close target was the eye's centre, taken from the `eye_left` /
`eye_right` anchors. An eye closes by the upper lid coming *down* onto the
lower one; closing onto the centre leaves the lash as a short bar floating in
the socket with skin above and below it, which reads as a squint. The lid line
is now placed inside the eye *opening* -- the `eyewhite` layer's box, not the
lash's, which includes the upper lashes and sits too high -- at
`blink.lid_ratio` (0.85, where 1.0 is the lower lid). Exposed as a slider,
since how far down a closed eye should sit is a matter of the art style.

**2026-08-29 -- the reclaimed seam, and where the neck's shadow went.** Hiding
the neck in the preview showed the garment's window as a torn edge that
glittered once per breath, with flat pink skin left along the collar. Three
faults, one cause each.

*The tear.* The margin was being applied per pixel, so it decided not only
which regions changed hands but where each one ended. On A-001 the neck
explained 93% of one band better while only 59% cleared the margin, and the
handover stopped in ragged mid-region. A margin-qualified core now seeds the
region and the extent follows plain "the back layer is better" out to where
that stops -- a real edge in the picture rather than an artefact of the
threshold. Boundary roughness 0.121 -> 0.079, against 0.07 for a smooth blob
of the same size.

*The glitter.* The alpha was cut hard. A jagged hard edge moved by sub-pixel
amounts resamples differently every frame. The handover is feathered now,
clamped by the back layer's own alpha so it can never open a gap: 364 hard-
edged boundary pixels become 54, and those are where the neck itself runs out.

*The pink.* It was the jaw's shadow on the neck, flattened. `topwear` had
absorbed a copy of that skin and painted it as an even light tone
([246,217,201] against the original's [231,200,180], error 62.1), while the
`neck` layer had the shadow all along (error 18.9). Handing the pixels back
restores it -- down the centre of the neck, luma 227-228 before and 184-197
after, against the original's 182-194.

Composite over the whole subject: mae 18.74 -> 17.46, bad 8.83% -> 7.75%.

**2026-08-29 -- H6: the limit was the rigid offset, not the layers.** H1 found
that past turnX 0.8 the scattered edge reveals merge into one gash down the
temple, and concluded the limit is how far one layer overhangs the next. That
is true of a rig that moves each layer *rigidly* by its depth. An ellipsoid
shell moves the same layers by where they sit on a head instead, and
`webui/rig_preview/measure_disocclusion.mjs` counts the difference. A reveal is
a pixel hair covered at rest and skin covers after the turn -- alpha coverage
and the manifest's z order only, no colour heuristic -- and `nose px` is the
control column: the shell's max yaw is tuned so the nose travels the same
distance at the same slider value, so the two columns compare equal turns.

| turnX | nose px | largest revealed region, parallax | ... with the shell |
| --- | --- | --- | --- |
| 0.20 | 8.5 | 584 | 437 |
| 0.40 | 16.9 | 1027 | 636 |
| 0.60 | 25.4 | **3275** | 767 |
| 0.80 | 33.9 | 4157 | 805 |
| 1.00 | 42.3 | 4866 | 1090 |
| 1.50 | 63.5 | 6162 | 634 |

The parallax column has a merge event between 0.4 and 0.6 -- 1027 px to 3275 --
and grows monotonically after it. The shell column has none: across the whole
sweep, including turns half again past the manifest's limit, the reveal stays
scattered along edges (110-140 separate regions, largest under 1100 px) and
never coalesces. Total revealed area only halves (5941 to 2815 at turnX 1.0),
which is the honest reading: the shell does not remove disocclusion, it stops
disocclusion from becoming one coherent tear, and a tear is what the eye finds.

The mechanism is visible in the `slide` column, which drops to exactly 0: on one
shell, two layers no longer part by their depth difference at all. That is also
the cost. `front hair` and `back hair` share the hair shell, so the parallax
between them is gone and their depth is expressed only by draw order. The fork
this was read from pushes back hair deeper with a separate term; giving
`back hair` its own shell is the obvious refinement and is not needed to answer
H6.

What this does **not** establish is that it looks right. Every number here is
computed from alpha, and a shell that foreshortens a flat painted face can be
geometrically continuous and still read as a mask deforming. The blend is
therefore a slider defaulting to 0, and `DEFAULT_MOTION.head_turn.max_x` stays
at the parallax rig's 0.8 until someone has watched the shell turn on a GPU.

**2026-08-29 -- the line across the jaw was a layer's own edge.** Watching the
rig move turned up two strokes the portrait does not have: a heavy one under
the jaw and a faint one where the neck meets the garment.

The heavy one is a decomposition artifact, not a rig one, and it is the same
fault as the ring the expression pack's donor `mouth` layer drew. A layer's
outermost pixels are painted toward its outline, so where `face` ends at the
chin its last two rows composite to luma 80 and 162 against an original of 223
and 225 -- a black stroke on light skin, two pixels above the chin line the
picture actually has. The layer behind it, `head`, has 215 and 236 there: the
right answer was already in the stack.

`rig.trim_layer_edges` asks `reclaim_occluded`'s question at every layer's own
boundary rather than between one named pair -- where the edge is decisively
wrong and what is behind is right, the edge hands over. Same core-seeded region
and clamped feather, two differences forced by the shape: no morphological
opening, because a 3 px band does not survive a 3x3 open, and a feather of 0.4
rather than 1.0, because here the handover *is* a boundary and a sigma of 1
leaves half the rim behind (the jaw stroke came back at -56 instead of -13).

| | composite mae | bad ratio |
| --- | --- | --- |
| raw layers | 18.30 | 8.57% |
| after `reclaim_occluded` | 17.02 | 7.48% |
| ... and `trim_layer_edges` | **15.89** | **6.98%** |

The second run moves the same way, 15.07 to 13.90 and 6.84% to 6.16%, which is
the only reason to believe the band and margin are not fitted to one picture.
At the jaw the two bad rows go from -63 and -143 to +8 and -13.

The faint line remains: where the neck's bottom meets the garment the two
alphas sum to 1.64 and the overlap darkens one row by 4 luma, with the garment's
flat 230 sitting 3 above the original below it. That is an overlap to normalize
rather than an edge to hand over, and 4 luma over two rows is where this stopped
being worth another pass.

## Out of scope

Deferred deliberately, not forgotten:

- **Hair strand detection and spring physics.** The single largest piece of
  Anime2.5DRig, and not needed to answer M4.
- **Mouth opening.** Requires synthesizing a layer A-001 does not produce.
- **Generic closed-eye synthesis** (resize and recolor a stock eyelid).
- **RIFE in-betweening.** Another model and more VRAM, for frames a 3-step
  open/half/close does not need.
- **Sprite-sheet / WebM bake and game-engine export.** A good Plan B for
  shipping, irrelevant to feasibility.
- **Spine auto-bone emission.** Cheap once Stage C exists; scheduled after.
- **PSD input.** Our input is the in-memory layer dict.
- **Live2D-class deformation**, forbidden by the fork plan's Phase 2.

## M4 acceptance

1. `rig_manifest.json` is produced automatically from an A-001 run with no
   manual layer editing.
2. At least two of head motion, blink, breathing, and mouth flap run in the
   browser preview.
3. The ghost silhouette observed in the Spine test is measurably reduced
   (H2), or the reason it is not is documented.
4. A maximum believable head rotation is recorded (H1), and the artifact that
   appears past it is named.
5. The milestone ends with a written verdict -- viable, conditionally viable,
   or not viable -- per the fork plan's Phase 2 wording.

## Algorithm references

Implemented from published behaviour, not copied source:

- **Anime2.5DRig** (MIT) -- neck vertex weight gradient, depth-differential
  pseudo head turn, connected-component eye separation, alpha-derived face
  and neck anchors, uniform grid meshing, blink by Y-compression.
- **Anime2.5DRig, izumix77 fork** (MIT) -- ellipsoid head and hair shells fitted
  to the face box, their empirical radii, the short-focal reprojection, and the
  difference-only blend that keeps the rest pose exact.
- **PNGAL** (Apache-2.0) -- animation configuration persisted separately from
  layer data; sprite-sheet + JSON as a game-delivery format.

Any file that ends up containing a substantial verbatim excerpt from either
project must carry that project's license header. Per the above, none should.
