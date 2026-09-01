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
   └─ composite_error.png
```

## Invariants

- Every image is full-canvas RGBA PNG unless a diagnostic explicitly declares
  grayscale.
- Coordinates use a top-left origin with Y increasing downward.
- RGB is sRGB and alpha is straight (unpremultiplied).
- `layers/` contains `production_repaired` canonical layers.
- After existing semantic layers complete fidelity repair, missing subject
  pixels pass through conservative semantic ownership recovery. Only
  unresolved residual becomes `body_remainder`. Keeping recovery after fitted
  repair prevents newly transferred source pixels from biasing tone/seam fits.
- The fidelity-repair order is fixed: `reclaim_occluded`, `fit_layer_tone`,
  `fit_edge_alpha`, `fit_seam_residual`, `clean_garment_orphans`.
- `semantics.warnings` records observable producer-side semantic omissions,
  such as `missing_eyewhite`. It is not a rig-readiness or motion verdict.
- A consumer must reject an unknown major format version.
- A consumer must never run fidelity repair when
  `layer_contract.canonical_stage` is `production_repaired`.
- `raw_layers/`, when present, is forensic data and is not a fallback source
  for missing canonical layers.
- Rig-specific subdivisions such as `head_remainder`, `neck_remainder`, and
  left/right eye splits are forbidden in the canonical layer set.

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
