# M2 Implementation Specification

## Goal

Give Portrait Mode a single-image webui that runs without ComfyUI, so A-001
can be run and its layers/diagnostics exported from a browser instead of a
node graph. M2 does not add batch processing (M3) or PostProcess/Spine
export -- it is a decomposition + verdict inspection tool, not a
replacement for the full ComfyUI pipeline.

## Why a separate app instead of extending the ComfyUI web extension

The original M2 scope (per `PORTRAIT_MODE_FORK_PLAN_v0.1.md`) was read two
ways during implementation: exposing the portrait report/diagnostics that
`SeeThrough_SavePSD` already writes to disk inside the existing ComfyUI node
graph UI, or a standalone app with no ComfyUI underneath at all (modeled on
reference tools like `BeamManP/see-through-webui`: drag-and-drop upload, a
resolution slider, a generate button, no node graph). The user picked the
standalone-app reading. This spec covers that.

## Shared core: `seethrough_engine/`

Building a second app that runs the same diffusion pipeline as `nodes.py`
created a real risk of the two silently drifting (a fix landing in one and
not the other). Rather than accept that, the reusable, non-ComfyUI-specific
pieces of `nodes.py` were extracted into a new `seethrough_engine/` package
that both `nodes.py` and `webui/app.py` import:

| Module | Contents | Needs torch? |
| --- | --- | --- |
| `vendor.py` | sys.path bootstrap into `see-through/common`; lazy re-export of vendored classes/functions | only for the pipeline-class group |
| `paths.py` | `resolve_model_path`, `scan_model_dirs`, `default_models_dir`, repo-id constants | no |
| `layers.py` | tag lists, `crop_head`, `layer_similarity`, `make_preview`, `align_subject_mask_to_canvas` | no |
| `device.py` | device/offload selection, cache clearing (replaces ComfyUI's `model_management`) | yes |
| `model_loading.py` | `load_layerdiff_model`, `load_depth_model`, text-encoder placeholder guard | yes |
| `generation.py` | `run_diffusion_stage` (single diffusion pass), `run_portrait_pipeline` (full Portrait Mode orchestration for a non-ComfyUI caller) | yes |
| `export.py` | `save_portrait_run`: write layers/diagnostics/report to a plain directory | yes (via PIL/numpy only, no torch, but imports `generation`) |

`nodes.py` was refactored to delegate rather than duplicate: `_resolve_model_path`,
`SeeThrough_LoadLayerDiffModel.load_model`, `SeeThrough_LoadDepthModel.load_model`,
`SeeThrough_GenerateLayers_Custom._run_diffusion`, `._layer_similarity`, and
`_prepare_portrait_subject_mask` are now thin wrappers around
`seethrough_engine`. The full auto-fill/portrait orchestration *loop* inside
`SeeThrough_GenerateLayers_Custom.generate` was **not** touched -- it stays
nodes.py's own control flow, interleaved with ComfyUI VRAM-offload
bookkeeping that isn't safe to restructure without the ability to run it
against a real GPU. `seethrough_engine.generation.run_portrait_pipeline` is
an independent implementation of that same orchestration, built from the
identical `run_diffusion_stage` primitive, for callers with no ComfyUI graph.
`tests/unit/test_node_engine_contract_static.py` pins the delegation calls so
this doesn't quietly regress back into two copies.

`paths.py` and `layers.py` have no torch import specifically so they stay
unit-testable in a lightweight environment (mirrors the existing rule for
`portrait_core`); `vendor.py` is split into two lazily-loaded groups
(`utils.cv` helpers vs. the diffusers pipeline classes) for the same reason.

## A deliberate divergence from nodes.py: real source alpha

`nodes.py`'s portrait branch always zeroes the alpha it hands to
`resolve_subject_mask`, because ComfyUI's `IMAGE` type is RGB-only -- any
alpha on it is synthesized, not real evidence, which is why that branch
leans on a separately-supplied `subject_mask` (MASK) input instead.
`seethrough_engine.generation.run_portrait_pipeline` does **not** do this: a
caller with no ComfyUI type system in the way can hand it a true RGBA image,
and the real alpha channel is passed straight through to
`resolve_subject_mask`, which already knows how to tell an informative
alpha channel from an opaque one. This is what lets the webui reproduce the
A-001 preflight baseline (`mask_source: source_alpha`, `mask_confidence:
HIGH`) directly from an uploaded transparent PNG, the way the protocol
describes, without requiring a separate mask upload.

`provided_subject_mask` (optional, for opaque-background uploads) is expected
at the original image's resolution and is aligned to the model's padded
square canvas via `align_subject_mask_to_canvas` -- the same transform
`nodes.py`'s `_prepare_portrait_subject_mask` applies to a ComfyUI MASK
input, now shared.

## Runtime contract (`webui/app.py`)

### Inputs (UI)

| Input | Meaning |
| --- | --- |
| portrait image | RGBA upload; real alpha used as trusted silhouette when informative |
| subject mask (optional) | Foreground-positive mask for opaque backgrounds |
| model | Local `models/SeeThrough/<name>` folder or a HuggingFace repo id |
| seed, resolution, steps | Same meaning as the ComfyUI node's inputs |
| enable_head_detail, silhouette_guard, auto_fill | Same meaning as the ComfyUI node's inputs; `portrait_mode` is implicitly always on -- this app has no non-portrait use case |

### Outputs (UI)

- verdict badge (PASS / SOFT_PASS / SOFT_PASS_LOW_CONFIDENCE / REWORK / FAIL)
- coverage metrics table and reasons list (from the same `build_portrait_report`
  used by nodes.py)
- layer gallery and diagnostics gallery (coverage/missing/spill/reconstruction/
  body_remainder)
- a downloadable `.zip` (layers, `*_portrait_report.json`, `*_manifest.json`,
  diagnostic PNGs) via `seethrough_engine.export.save_portrait_run`
- run log

Runs also persist under `webui/outputs/<run_id>/` (gitignored) so a person
can compare two runs on disk without re-downloading.

## Out of scope (explicitly deferred, not forgotten)

- PostProcess stage (hair L/R split, depth, un-padding to the original
  aspect ratio) and Spine export -- layers are exported in the model's
  padded square working resolution, matching what `portrait_core`'s guard
  and report already compute against (`fullpage`), not the original image's
  size.
- PSD export (the ComfyUI extension builds this client-side via `ag-psd` in
  the browser; this webui exports PNG + JSON instead).
- Batch processing (M3).
- `install.bat`/`run.bat`-style one-click installer automation, or bundled
  model auto-download UX beyond what `resolve_model_path` already does.
  `python webui/app.py` after `pip install -r webui/requirements.txt` is the
  whole install story for now.

## Verification performed

This change was made without GPU access or the real LayerDiff/Marigold
checkpoints, so it could not be run end to end. What was actually verified:

1. `tests/unit` (38 tests, no torch/GPU required) passes, including new
   tests for `seethrough_engine.layers` (`layer_similarity`, `crop_head`,
   `align_subject_mask_to_canvas`, `make_preview`), `seethrough_engine.paths`
   (`resolve_model_path`, `scan_model_dirs`), and the nodes.py delegation
   contract.
2. `nodes.py` still parses and its refactored methods were checked line by
   line against the pre-refactor version for exact parameter/behavior
   parity -- only the call sites moved, not the logic.
3. `webui/app.py`'s Gradio `Blocks` app was actually built, `.queue()`d,
   launched on a local port, and hit with a real HTTP request (200 OK), then
   closed cleanly -- this exercises every UI component and event-wiring call
   in the file. What it does **not** exercise is the `run_a001` callback
   itself (needs torch + the real checkpoints).

## M2 acceptance (to be confirmed on real hardware)

1. Uploading the A-001 asset with `silhouette_guard` on reproduces the same
   verdict/coverage numbers as running the same settings through the ComfyUI
   node graph (same seed, resolution, steps).
2. The exported zip's `*_portrait_report.json` validates against the same
   schema `build_portrait_report` already produces for nodes.py (schema_version 1).
3. `python webui/app.py` starts and serves the UI with no ComfyUI process
   running.
4. Existing ComfyUI node behavior is unchanged for a real GPU run (guarded
   by manual regression, since `tests/unit` cannot load the diffusion
   pipeline).
