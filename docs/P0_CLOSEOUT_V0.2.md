# SeeThrough P0 Closeout v0.2

This note freezes the producer contract for the Portrait Bundle v1.

## Frozen boundary

SeeThrough produces a validated semantic source asset: decomposition, hidden
region completion, canonical ownership, static reconstruction, seam/local
fidelity, and diagnostic occlusion measurements. It does not produce final
character draw order, recipes, bake/merge decisions, meshes, anchors, weights,
or animation/runtime policy.

`semantics.z_order` is the source reconstruction order only. The occlusion
graph is measured evidence (`front`, `back`, overlap/boundary/hidden extent,
depth margin, confidence, and disocclusion risk); it never emits `bake`,
`merge`, `rig`, or `motion_safe` decisions.

## Reproducibility

- `regression` mode preserves seed `42` for fixtures and bug reproduction.
- `deterministic_auto` derives a stable seed from `source_identity` and
  `attempt_index`.
- Bundle `generation` metadata records `seed_mode`, `attempt_index`, `seed`,
  and `canonical_regression_seed`.

## Repair policy

Cleanup repair is producer-owned and may run on every attempt. Semantic
recovery is lower priority: current-attempt extracted semantic, another
deterministic attempt, source-backed conservative derivation, then heuristic
emergency fallback. A missing tag is reported as a diagnostic and is not a
rig-readiness verdict.

`experimental/source_reprojection.py` remains opt-in and is not imported by
the production generation or export path.
