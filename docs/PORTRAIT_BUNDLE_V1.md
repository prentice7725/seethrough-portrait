# Portrait Bundle v1

`Portrait Bundle` is the only file seam between `seethrough-portrait` and a
downstream consumer such as `portrait-autorig`.

## Directory layout

```text
<name>.portrait/
├─ manifest.json
├─ original.png
├─ layers/                 # canonical; downstream may consume
│  ├─ face.png
│  ├─ neck.png
│  ├─ topwear.png
│  └─ body_remainder.png   # optional when empty
├─ raw_layers/             # optional; diagnostic only
└─ diagnostics/
   ├─ portrait_report.json
   ├─ semantic_ownership.json
   ├─ local_fidelity.json
   ├─ fidelity.json
   ├─ seams.json
   ├─ coverage_mask.png
   ├─ missing_mask.png
   ├─ spill_mask.png
   ├─ reconstruction.png
   ├─ layer_composite.png
   ├─ composite_error.png
   └─ occlusion_graph.json    # optional
```

## Invariants

- Every image is full-canvas RGBA PNG unless a diagnostic explicitly declares
  grayscale.
- Coordinates use a top-left origin with Y increasing downward.
- RGB is sRGB and alpha is straight (unpremultiplied).
- `layers/` contains `production_repaired` canonical layers.
- `semantics.z_order` is the producer's reconstruction order for this source
  portrait. It is not a downstream character's final draw order; consumers may
  adapt it for composition.
- `generation` records the reproducibility source: `regression` keeps the
  canonical seed (42), while `deterministic_auto` derives a stable seed from
  `source_identity` and `attempt_index`. Recommended production modes are one
  attempt for NORMAL, three for QUALITY, and five for HARVEST.
- After existing semantic layers complete fidelity repair, missing subject
  pixels pass through conservative semantic ownership recovery. Only
  unresolved residual becomes `body_remainder`. Keeping recovery after fitted
  repair prevents newly transferred source pixels from biasing tone/seam fits.
- The fidelity-repair order is fixed: `reclaim_occluded`, `fit_layer_tone`,
  `fit_edge_alpha`, `clean_garment_orphans`, `fit_edge_alpha_final`,
  `fit_mouth_contact`, `fit_seam_residual`. The mouth stage is a local static
  ownership/alpha solve for skin-coloured mouth mattes; it has no rig or
  motion knowledge, and the final seam fit evaluates the published ownership
  boundaries.
- `diagnostics/local_fidelity.json` measures eyes, mouth, and the local
  neck/garment contact band. It reports source-visible loss and static seam
  evidence; it is not a motion or rig-readiness policy.
- `semantics.warnings` records observable producer-side semantic omissions,
  such as `missing_eyewhite`. It is not a rig-readiness or motion verdict.
- `diagnostics/occlusion_graph.json`, when present, records which canonical
  layers touch, how much one hides of the other, and a static
  `disocclusion_risk` score derived only from measured alpha overlap and
  z-order (no rig, motion, or deformation-safety verdict). A consumer must
  treat its absence as "not computed", never as "no occlusion".
- A consumer must reject an unknown major format version.
- A consumer must never run fidelity repair when
  `layer_contract.canonical_stage` is `production_repaired`.
- `raw_layers/`, when present, is forensic data and is not a fallback source
  for missing canonical layers.
- Rig-specific subdivisions such as `head_remainder`, `neck_remainder`, and
  left/right eye splits are forbidden in the canonical layer set.

## Repair and recovery policy

Cleanup repair (edge/seam fitting, alpha fringe cleanup, and conservative
ownership cleanup) may run on every producer result. Semantic recovery that
invents or derives a missing tag is lower priority and follows this ladder:

1. real semantic from the current attempt;
2. real semantic from another deterministic attempt;
3. source-backed conservative derivation;
4. heuristic emergency fallback.

The producer reports these observations but does not emit rig-readiness or
motion policy. This is the Portrait Bundle v1 P0 freeze contract.

The machine-readable schema is
[`portrait-bundle-v1.schema.json`](portrait-bundle-v1.schema.json).

## Publishing rule

A producer writes into a temporary directory and publishes the bundle only
after static fidelity and seam validation have completed. The manifest records
the validation result and repair provenance even when policy allows a
non-passing diagnostic bundle to be retained.

## Compatibility

The pre-v1 flat run directory is not a Portrait Bundle. Compatibility belongs
to a downstream legacy adapter, which may perform the old repair sequence once
and then hand an in-memory `production_repaired` portrait to the AutoRig
compiler.
