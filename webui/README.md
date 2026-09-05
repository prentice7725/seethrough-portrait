# SeeThrough Portrait — standalone producer WebUI

This WebUI turns one source portrait into a validated, production-ready
**Portrait Bundle v1**. It owns semantic decomposition, silhouette protection,
static fidelity repair, and diagnostics. The downstream flow is:

```text
Source Portrait
  → SeeThrough Portrait
  → Portrait Bundle v1
  → Portrait Composer
  → Assembly Bundle v0.2
  → Portrait AutoRig
  → Rig Bundle
```

Rigging, expression runtime, Spine export, and deformation are intentionally
outside this repository. The WebUI never imports `portrait-autorig`; the Bundle
file contract is the boundary.

## Install

```bash
# Install a CUDA-matched torch + torchvision build first:
# https://pytorch.org/get-started/locally/
python -m pip install -r webui/requirements.txt
```

Models download automatically from HuggingFace on first use into
`models/SeeThrough/`, or a pre-downloaded LayerDiff checkpoint can be placed
there.

## Run

```bash
python webui/app.py
```

On Windows, use the same interpreter for installation and launch:

```powershell
cd C:\workspace\seethrough-portrait
.\.venv\Scripts\python.exe -m pip install -r webui\requirements.txt
.\.venv\Scripts\python.exe webui\app.py
```

Open `http://127.0.0.1:7860`, upload a Source Portrait, and optionally provide a
subject mask (white = subject). A transparent-background PNG is recommended;
for an opaque flat background, leave **단색 배경을 투명 처리** enabled.

### Production Profile

- **NORMAL** — one deterministic candidate; the default production path.
- **QUALITY** — three candidates, ranked by static coverage and fidelity.
- **HARVEST** — five candidates for producer-side candidate generation. This is
  not Composer donor harvesting.

Silhouette Guard is enabled for production profiles. It can only be disabled in
the Research / Debug accordion. Face detail generation is enabled by default.

### Advanced and reproducibility

Resolution, head-detail resolution, inference steps, and the model checkpoint
are in **Advanced**. Seed controls are in **Reproducibility**:

- `deterministic_auto` derives a stable seed from the source identity and
  attempt index.
- `regression` uses the displayed regression seed (42 by default) for repeatable
  fixtures.

The UI does not expose an auto-fill count. Profile semantics own the number of
candidate attempts.

## Results

The result panel reports three independent validation axes:

- **Static Reconstruction** — does the canonical stack reproduce the source?
- **Seams** — are thin, continuous contact artifacts present?
- **Local Fidelity** — do eye, mouth, and neckline ROIs preserve important
  details even when global MAE looks good?

The old PASS/REWORK/FAIL value remains available below as **Diagnostic Summary**;
it is not the Bundle's sole acceptance signal. Semantic completeness warnings,
such as `missing_eyewhite`, are producer-side observations for downstream
review, not an AutoRig readiness verdict.

The default galleries show only `CANONICAL` assets and static `DIAGNOSTICS`.
`raw_layers/` is preserved inside the downloaded archive for Forensics and must
not be consumed as canonical input.

Runs are also kept under `webui/outputs/<timestamp>_<id>.portrait/` (gitignored).

## Portrait Bundle v1 layout

```text
A002.portrait/
├─ manifest.json
├─ original.png
├─ layers/                 # canonical production-repaired assets
├─ raw_layers/             # forensic model output; downstream use prohibited
└─ diagnostics/
   ├─ portrait_report.json
   ├─ semantic_ownership.json
   ├─ local_fidelity.json
   ├─ fidelity.json
   ├─ seams.json
   ├─ occlusion_graph.json
   ├─ coverage_mask.png
   ├─ missing_mask.png
   ├─ spill_mask.png
   ├─ reconstruction.png
   ├─ layer_composite.png
   └─ composite_error.png
```

`layers/` is the `production_repaired` canonical stage. See
[`docs/PORTRAIT_BUNDLE_V1.md`](../docs/PORTRAIT_BUNDLE_V1.md) for the schema and
invariants.

## Architecture boundaries

- **SeeThrough Portrait**: image decomposition, semantic ownership, local
  contact repair, static validation, and Portrait Bundle export.
- **Portrait Composer**: donor/VariantSet/ExpressionPreset selection, final draw
  order, and Assembly Bundle `RigIntent`.
- **Portrait AutoRig**: derived rig parts, mesh/weights, deformation, and
  runtime binding.

The standalone producer exports full-canvas PNG layers by contract. PSD export
and the original ComfyUI compatibility nodes remain separate optional paths;
they are not prerequisites for this WebUI.

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

Install `opencv-python` into the exact Python shown in the error:

```powershell
python -m pip install opencv-python
```

### The process dies mid-run with the GPU pegged at 100%

On Windows this is commonly Timeout Detection and Recovery (TDR), not a Python
exception. Check Event Viewer for `nvlddmkm` event 153. If confirmed, raise
`TdrDelay` and reboot, following your organization's GPU administration policy.
The first CUDA call on a new GPU/torch shape can also spend several minutes in
kernel autotuning before the sampling progress bar appears.

### 8GB GPU or slow high-resolution runs

The runtime measures available VRAM immediately before body/head diffusion,
prefers untiled VAE decode, and retries once with tiled decode on CUDA OOM.
See the main [README](../README.md) and
[`docs/VAE_RUNTIME_POLICY.md`](../docs/VAE_RUNTIME_POLICY.md) for measured
trade-offs.
