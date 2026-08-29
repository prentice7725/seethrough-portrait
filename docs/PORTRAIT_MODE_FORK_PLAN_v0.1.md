# PORTRAIT_MODE_FORK_PLAN_v0.1

## Purpose

Fork tk_seethrough into a portrait-focused derivative that can decompose
front-facing upper-body characters without losing arms, shoulders, or sleeves.
Perfect semantic separation is desirable, but preservation of the original
silhouette is mandatory.

## Scope

### Phase 1 — implementation

- Portrait Mode
- Silhouette Guard
- BODY_REMAINDER
- portrait-aware coverage scoring and auto-fill selection
- WebUI and batch workflow
- A-001 validation

### Phase 2 — feasibility only

- test the exported layers in a 2.5D rig chain
- check idle sway, blink, mouth motion, and slight head motion
- conclude viable, conditionally viable, or not viable
- do not implement production auto-rigging or Live2D-class deformation

## Output profile

The first usable target is `hair_back`, `body`, `face`, and `hair_front`, with
eyes, mouth, eyebrows, and accessories when available. Internal diagnostic and
safety layers include `topwear`, `handwear`, `body_remainder`, `coverage_mask`,
and `missing_mask`.

`BODY_FINAL = TOPWEAR + HANDWEAR + BODY_REMAINDER`

## Milestones

| Milestone | Deliverable | Exit condition |
| --- | --- | --- |
| M1 | Portrait core and A-001 validation | silhouette recovered; face/hair/body usable |
| M2 | single-image WebUI | A-001 can be run and exported from the UI |
| M3 | batch processing | multiple portraits produce isolated outputs and reports |
| M4 | 2.5D feasibility | at least two of idle, blink, mouth, or head motion work |

## Success criteria

1. Missing model pixels return in the final result through original-pixel recovery.
2. FACE/HAIR/BODY form a stable minimum layer set.
3. The WebUI supports repeatable single and batch runs.
4. A minimal 2.5D motion test can consume the exported layers.

## Current focus

M1 and M2 are implemented. See `M1_IMPLEMENTATION_SPEC.md` and
`TEST_PROTOCOL_A001.md` for M1's executable contract and validation
procedure, and `M2_IMPLEMENTATION_SPEC.md` for the standalone single-image
webui, including what it could not verify without GPU access.

M4 is being taken before M3: a manual Spine rig test on A-001's exported
layers was informative enough to justify going straight at the 2.5D
feasibility question. See `PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md`, which
answers M4 with an automatic rig plus a browser preview rather than a bone
hierarchy, and stays inside Phase 2's "feasibility only" limit.

## Candidates

Recorded so they are not rediscovered, and deliberately not scheduled: each one
is outside M4, and M4 is a decision rather than a product.

**M4.1, expression pack -- started.** PachiPakuGen (kazuya-bros), which runs the
upstream See-Through implementation as a fixed-commit runtime, asks the user for
seven extra images -- `eyes-closed`, `mouth-closed`, and the five vowel mouths --
drawn from the same character, and decomposes all eight together. This is the
practical answer to the two drawings this fork cannot synthesize: A-001's
`mouth` is a closed mouth and there is no closed eye at all, which is why the
rig currently closes an eye by keeping a fraction of the lash.

What makes it work is that the generated image is a **donor, not a portrait**.
Only the region it actually changed is taken from it; everything else stays the
original's own pixels. That is the difference between this and the fork's
earlier modular attempt, which used the generated image as the portrait and lost
the character's identity to the generator's drift.

`seethrough_engine/expression.py` implements the recovery, and takes the two
steps in the opposite order to PachiPakuGen. They composite the donor into a
full frame and decompose it again, because at that point they have no layers.
We already have layers, and Portrait Mode's `face` layer is featureless skin
behind every feature -- luma standard deviation 0.5-0.8 under A-001's eye and
mouth boxes, against 25.5 over the face as a whole -- so a recovered region
drops in as one more part over clean skin. No second decomposition, no GPU, and
no chance of the rest of the portrait coming back different.

Two departures from their algorithm, both forced by that ordering:

* Their regions are fixed fractions of the canvas, which suits a full-body
  standing picture and not a bust. Ours come from the run's own anchors and
  layer boxes, which is information they do not have at that stage.
* Their single threshold is a percentile of the region, so it tracks how much of
  the region the edit fills rather than how different the edit is. Recovery uses
  the same core/extent hysteresis as `rig.reclaim_occluded`, over a drift level
  measured on the part of the picture nothing is taken from.

Scope is three images -- `base`, `eyes-closed`, `mouth-open` -- and the vowels
are a later promotion.

The pack is attached to a finished run rather than produced inside it
(`python -m seethrough_engine.expression <run dir> eye_closed=<png> ...`):
the donors are drawn *after* looking at the decomposition, and re-running the
model to attach them would be absurd. It writes an `expressions` block into the
run's existing `rig_manifest.json`, placing each sprite over the parts it stands
in for -- their front z, their nearest depth, and their own weight, so it moves
as the feature it covers rather than as a new tag.

The runtime gives each feature one owner. A short crossfade centred on the
half-closed pose hands the eye from the layers to the art and back: a long
dissolve shows an open eye and a shut one at once, an instant swap pops. The
open eye keeps squashing until the art has taken over, so the lid is still seen
coming down. With no pack -- or with "Use expression art" off, which is the
comparison the toggle exists for -- every part draws at full and the blink is
the v0.1 lash squash exactly.

Remaining: the webui does not collect donors yet, so the pack is attached from
the command line. Eyebrow transform and iris gaze need no new drawings at all
and are the cheapest thing left.

**Seed selection before the batch.** The same tool decomposes a single image
with the depth pass and PSD assembly skipped, purely so the user can look at
which parts came out and re-roll the seed. Portrait Mode's `auto_fill` already
reruns up to five times and keeps the best combination, but it scores alpha, and
the run that reached PASS with no `eyewhite` at all is exactly the failure a
person would have caught by looking. A fast layers-only preview in the webui
would cost little and would shorten every A-001 experiment.

**Seamless loop export.** For the sprite-bake Plan B: align every driver's
period to the export length and record after several warm-up cycles, so the
first frame and the last one agree.

**8 GB profile.** Their low-VRAM path is CPU offload plus quantized weights plus
freeing VRAM between inference stages -- the third of those is the one this fork
has not tried.

