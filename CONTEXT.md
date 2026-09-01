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

The ordered static reconstruction pass: reclaim occluded pixels, fit material
tone, fit edge alpha, fit residual seams, then remove or transfer
well-supported orphan semantic contamination. It compares layers with the original still
image and contains no animation policy. A proposed cleanup is accepted only
when canonical composite fidelity does not regress.

## Semantic ownership recovery

The conservative static pass after existing semantic fidelity repair and
before `body_remainder` construction. It returns high-confidence missing pixels to an
existing semantic layer using local connectivity, source colour, competing
ownership, and an exact reconstruction gate. Pixels without decisive evidence
remain unresolved residual.

## Unresolved residual

Subject pixels left after semantic ownership recovery. Only this residual may
be published as `body_remainder`; the term no longer means every pixel omitted
by model decomposition.

## Local fidelity

Static reconstruction measured in feature-anchored face regions rather than
averaged over the whole subject. Left/right eye and mouth results expose small
critical losses, such as sclera disappearance, that global MAE can hide.

## Resolution regression fixture

A recorded same-input A/B comparison across generation resolutions. It tracks
semantic tags, unresolved residual, garment contamination, global fidelity,
local eye fidelity, and warnings.

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
