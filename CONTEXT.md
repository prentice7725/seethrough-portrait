# Domain context

## Portrait Bundle

A versioned, directory-based file contract containing validated,
production-ready semantic portrait layers. `layers/` is the canonical consumer
surface. `raw_layers/` is optional forensic output and must not be consumed by
downstream tools.

## Canonical layer

A full-canvas, straight-alpha, sRGB RGBA image after Silhouette Guard and the
complete fidelity-repair sequence. Canonical layers are safe for another
program to consume without running repair again.

## Raw layer

The selected model output before Silhouette Guard and fidelity repair. Raw
layers exist only for debugging, provenance, and regression analysis.

## Fidelity repair

The ordered static reconstruction pass: reclaim occluded pixels, remove
well-supported orphan semantic contamination, fit material tone, fit edge
alpha, then fit residual seams. It compares layers with the original still
image and contains no animation policy. A proposed cleanup is accepted only
when canonical composite fidelity does not regress.

## Semantic warning

A producer-side observation that a portrait visibly contains a semantic
feature which has no independent canonical tag. It describes semantic output
completeness only; it is not a downstream readiness or motion-policy verdict.

## Static validation

Checks that canonical layers reconstruct the source portrait while still. It
includes composite fidelity and seam detection.

## AutoRig compiler

The downstream transformation from a Portrait Bundle into derived motion
parts, anchors, depth, meshes, weights, expressions, and a Rig Bundle. It never
modifies canonical Portrait Bundle layers and never performs fidelity repair.

## Legacy run

The flat directory format emitted before Portrait Bundle v1. A legacy adapter
may repair and normalize it once before passing an in-memory portrait to the
AutoRig compiler.
