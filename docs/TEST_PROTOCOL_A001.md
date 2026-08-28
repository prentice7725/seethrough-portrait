# A-001 Test Protocol

## Asset

`PORTRAIT_d01_001_school_neutral_v01.png`

The asset is a single-character, front-facing upper-body portrait with a
transparent background. It is treated as test input, not as a checked-in repo
fixture unless redistribution rights are confirmed.

### Preflight baseline

| Property | Observed value |
| --- | --- |
| dimensions | 1212 x 1300 |
| color mode | RGBA, 8-bit |
| foreground ratio | 41.0432% |
| alpha edges | binary/hard-edged |
| foreground bounding box | `(199, 80)` to `(1029, 1300)` |
| border contact | bottom only |
| subject-mask source | source alpha |
| mask confidence | HIGH |

Bottom contact is expected for this cropped upper-body portrait and must not be
reported as missing legs or lower body.

## Test A — baseline decomposition

Run the upstream settings with `portrait_mode=false` and
`silhouette_guard=false`. Save semantic layers and a reconstruction preview.
This records the unmodified baseline and is not expected to pass M1 criteria.

## Test B — portrait evaluation

Run the same seed and inference settings with `portrait_mode=true` and
`silhouette_guard=false`. Verify that crop-aware critical groups are applied and
that absent legs or handwear alone do not create a semantic failure.

## Test C — guarded portrait

Run the same seed and inference settings with both features enabled. Verify:

1. generated layers are clipped to the trusted source silhouette;
2. `missing_mask` matches the positive subject-minus-layer difference;
3. `body_remainder` contains original source pixels, not generated pixels;
4. the reconstruction reaches at least 99.9% silhouette coverage;
5. the verdict reflects remainder dependency rather than coverage alone.

## CPU guard proof on A-001

Before a model run, validate the recovery math by deterministically removing the
lower outer 22% of both sides of the subject box after 52% of its height. This
simulates a severe shoulder/arm omission; it is not a model-quality result.

| Metric | Result |
| --- | ---: |
| simulated omitted pixels | 161,498 |
| simulated omitted ratio | 24.9735% |
| pre-recovery coverage | 75.0265% |
| recovered pixels | 161,498 |
| post-recovery coverage | 100.0000% |
| post alpha MAE | 0.000000 |
| expected verdict | REWORK |

The REWORK verdict is correct: the original silhouette is safe, but a quarter
of the subject relying on `body_remainder` is too large to call a clean semantic
decomposition.

## Evidence to retain per run

- exact commit and config JSON
- model/checkpoint identifiers
- resolution, steps, seed, and auto-fill state
- all PNG layers and PSD
- coverage, missing, and reconstruction previews
- JSON report and runtime log

## M1 decision

- **Guard PASS:** post-recovery coverage is at least 99.9% with no material spill.
- **Semantic PASS:** FACE, HAIR, and BODY critical groups are usable on review.
- **M1 PASS:** both conditions hold on Test C.
- **REWORK:** guard succeeds but semantic layers or remainder dependency are not
  usable.
- **FAIL:** trusted silhouette is not reconstructed after recovery.

