# SeeThrough Portrait -- standalone webui (M2)

A single-image Portrait Mode webui that runs without ComfyUI. Upload one
upper-body portrait, run A-001, and get the decomposed layers, the Silhouette
Guard diagnostics, and the PASS / SOFT_PASS / REWORK / FAIL verdict -- all
downloadable as a zip.

This is the M2 milestone from `PORTRAIT_MODE_FORK_PLAN_v0.1.md`: "single-image
WebUI" whose exit condition is "A-001 can be run and exported from the UI."
See `docs/M2_IMPLEMENTATION_SPEC.md` for the full contract and known
limitations.

## Install

```bash
# 1. Install a CUDA-matched torch build first:
#    https://pytorch.org/get-started/locally/
# 2. Then everything else:
pip install -r webui/requirements.txt
```

Models download automatically from HuggingFace on first use, into
`models/SeeThrough/` at the repo root (or drop a pre-downloaded checkpoint
folder there yourself -- see the main [README](../README.md#models)).

## Run

```bash
python webui/app.py
```

Opens `http://127.0.0.1:7860`.

1. Upload a portrait. A transparent-background PNG is strongly recommended --
   the app reads the real alpha channel as the trusted subject silhouette
   (`mask_confidence: HIGH`, same as the A-001 protocol). For an
   opaque-background image, also upload a **subject mask** (white = subject).
2. Pick resolution / steps / seed, and whether to enable head detail,
   Silhouette Guard, and auto-fill.
3. Click **Run A-001**.
4. Read the verdict badge and reasons, browse the layer/diagnostic
   thumbnails, and download the zip (layers, `*_portrait_report.json`,
   `*_manifest.json`, coverage/missing/spill/reconstruction PNGs).

Runs are also kept on disk under `webui/outputs/<timestamp>_<id>/` (gitignored)
in case you want to inspect them without re-downloading the zip.

## Known limitations (M2 scope)

- **No PostProcess stage.** Layers are exported at the model's padded square
  working resolution (`fullpage`), not un-padded back to the original image's
  aspect ratio/size, and there is no hair L/R splitting, depth estimation, or
  Spine export here -- that whole stage (`SeeThrough Post Process` and
  downstream nodes) stays ComfyUI-only for now. This tool is for running and
  inspecting A-001, not for producing a final rigging-ready PSD.
- **No PSD export.** The ComfyUI extension builds a PSD client-side in the
  browser via `ag-psd`; this webui exports plain PNGs + JSON instead.
- **One model resident at a time.** Switching the model dropdown unloads the
  previously loaded one.
- Not verified end-to-end on real GPU hardware as part of this change (see
  `docs/M2_IMPLEMENTATION_SPEC.md`) -- the Gradio app itself was smoke-tested
  (builds, serves, responds to HTTP), but no real diffusion run.
