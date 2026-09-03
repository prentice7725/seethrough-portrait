# Research Absorption Status

Tracks what `SEETHROUGH_PORTRAIT_RESEARCH_ABSORPTION_PLAN_v0.1` decided against
this repository's actual state, so a later session does not have to re-read
all 24 sections to find out what already happened.

The plan's own priority call: **BUILD NOW** is Backend Protocol, Occlusion
Graph, and the Source Pixel Lock experiment; everything else in P1 is **ADD
LATER**; Qwen-Image-Layered, replacing See-through's core with SAM2+LaMa, and
a hard LayerStyle dependency are **DO NOT DO** (plan section 21/"결정 요약").

## Implemented

- **Backend Protocol** (plan §6) — [`seethrough_engine/backends.py`](../seethrough_engine/backends.py).
  `BackendProposal` + `SubjectMatteBackend` / `DecompositionBackend` /
  `SemanticProposalBackend` / `EdgeMatteBackend` protocols, with the current
  deterministic implementation wrapped as the default for each role except
  `SemanticProposalBackend` (no default exists yet; see P1 below).
  **Not wired into the production pipeline.** `generation.py`/`repair.py` do
  not import this module — Phase A ("structure only, zero behavior change")
  stops at defining the interface. Wiring a real alternative backend behind
  it is P1 work.
- **Occlusion Graph** (plan §5) — [`seethrough_engine/occlusion_graph.py`](../seethrough_engine/occlusion_graph.py),
  wired into [`export.py`](../seethrough_engine/export.py) and published as
  the optional `diagnostics/occlusion_graph.json` (documented in
  [`PORTRAIT_BUNDLE_V1.md`](PORTRAIT_BUNDLE_V1.md)). Computed from existing
  alpha/z-order data only, no inference model. Depth (`depth_margin`) is
  accepted as an optional input and is `null` in every bundle produced today,
  since nothing currently hands `export.py` a Marigold depth dict.
- **Source Pixel Lock experiment** (plan §4) — [`experimental/source_reprojection.py`](../experimental/source_reprojection.py),
  a standalone, offline A/B script over a *saved* Portrait Bundle directory.
  **Not imported by `seethrough_engine`, not wired into the canonical
  pipeline.** Run it manually (`python -m experimental.source_reprojection
  <bundle_dir> <output_dir>`) to get a comparison report; nothing about
  production output changes until a future phase promotes this.
- **P0 closeout contract** — [`P0_CLOSEOUT_V0.2.md`](P0_CLOSEOUT_V0.2.md)
  freezes producer-order semantics, diagnostic-only occlusion output, the
  regression seed `42` versus the deterministic-auto schedule, and the
  cleanup/recovery repair priority. These policies add metadata and guards;
  they do not add Composer or AutoRig responsibilities.

## Deferred (P1 — plan §17 Phase D–F)

Not started this session: ViTMatte ROI edge refinement, complex-background
neural matte fallback, EVF-SAM2 semantic proposal, anime face parser critic.
`SemanticProposalBackend` in `backends.py` has no implementation yet for the
same reason.

## Explicit non-goals (plan §21)

Qwen-Image-Layered integration, any heavy decomposition backend replacing
See-through, SAM2+LaMa as the core pipeline, LayerDivider as a semantic
splitter, a hard LayerStyle repository dependency, Cubism rigging / mesh /
bone / weight / expression / runtime animation work, and
`portrait-autorig` becoming a producer into this repository.
