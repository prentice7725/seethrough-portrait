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

**Expression sheets as decomposition input.** PachiPakuGen (kazuya-bros), which
runs the upstream See-Through implementation as a fixed-commit runtime, asks the
user for seven extra images -- `eyes-closed`, `mouth-closed`, and the five vowel
mouths -- drawn from the same character, and decomposes all eight together. This
is the practical answer to the two layers this fork cannot synthesize: A-001's
`mouth` is a closed mouth and there is no closed eye at all, which is why the
rig currently closes an eye by keeping a fraction of the lash. The cost is a new
problem this fork does not have today -- registering several generations of one
character against each other -- which that tool solves with a manual alignment
step. Belongs to Phase 1 scope, if anywhere.

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

