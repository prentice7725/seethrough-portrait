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

M1 is the only active implementation milestone. See
`M1_IMPLEMENTATION_SPEC.md` and `TEST_PROTOCOL_A001.md` for its executable
contract and validation procedure.

