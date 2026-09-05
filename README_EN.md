# SeeThrough Portrait

[한국어](README.md)

A ComfyUI extension and standalone webui that turns one anime-style portrait
into a **validated, production-ready semantic Portrait Bundle**.

It builds on [@jtydhr88](https://github.com/jtydhr88)'s
[ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through), adding
Portrait Mode, Silhouette Guard, fidelity repair, and static validation.

## Project responsibility

```text
portrait.png
    ↓
seethrough-portrait
    ↓
Portrait Bundle v1
    ↓
Portrait Composer
    ↓
Assembly Bundle v0.2
    ↓
portrait-autorig
    ↓
Rig Bundle
```

This repository ends at `Image → validated, production-ready semantic portrait
bundle`. Portrait Composer owns donor/VariantSet/ExpressionPreset selection,
final draw order, and `RigIntent`. AutoRig owns mesh, weights, deformation, and
runtime binding in the separate
[`portrait-autorig`](https://github.com/prentice7725/portrait-autorig)
repository. The projects share the Portrait Bundle file contract and no Python
imports.

## Portrait Mode

- **Silhouette Guard** clips spill and recovers unexplained original pixels as
  `body_remainder`.
- **Production profiles** map to deterministic candidate generation: `NORMAL`
  runs once, `QUALITY` compares three attempts, and `HARVEST` compares five.
  HARVEST is a SeeThrough producer profile, not Composer donor harvesting.
- **Fidelity repair** runs `reclaim_occluded → fit_layer_tone → fit_edge_alpha
  → clean_garment_orphans → fit_edge_alpha_final → fit_mouth_contact
  → fit_seam_residual` against
  the original still
  image and conservatively removes isolated garment semantic contamination.
- **Static validation** measures whole-composite fidelity and thin, continuous
  seam artifacts separately.
- **Semantic ownership recovery** returns high-confidence missing pixels to an
  existing canonical layer before unresolved residual becomes
  `body_remainder`.
- **Local fidelity** checks left/right eye, mouth, and neckline-contact ROIs so
  sclera loss or a horizontal neck/topwear seam cannot hide behind a good global MAE.
- **Three validation axes** report Static Reconstruction, Seams, and Local
  Fidelity independently; the legacy PASS/REWORK/FAIL value is shown only as a
  Diagnostic Summary.

## Portrait Bundle v1

The standalone webui's canonical output:

```text
A001.portrait/
├─ manifest.json
├─ original.png
├─ layers/                 # canonical production-repaired assets
├─ raw_layers/             # optional forensic output
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

Downstream consumers use only `layers/`. `raw_layers/` records model output for
forensics and is not a fallback asset source or canonical input. See
[`docs/PORTRAIT_BUNDLE_V1.md`](docs/PORTRAIT_BUNDLE_V1.md) for the invariants
and JSON Schema.

## Install for ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/prentice7725/seethrough-portrait.git
cd seethrough-portrait
pip install -r requirements.txt
```

Models download into `models/SeeThrough/` on first use.

| Model | Repository | Purpose |
| --- | --- | --- |
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | semantic layers |
| Marigold Depth | `layerdifforg/seethroughv0.0.1_marigold` | depth and PostProcess |

## 8GB GPUs and processing time

The standalone webui measures available VRAM and switches to **leaf-level
block streaming** when the UNet cannot reside on the GPU as a whole, allowing
it to run on an 8GB-class GPU. This is not something that lowering
`resolution` or `steps` alone can solve: this model's UNet is about 7.58 GiB
even in bf16, so a card that cannot hold all weights can fail before inference
begins. VAE tiling is not fixed at model load: immediately before each body or
head diffusion call, the runtime prefers untiled from the stage resolution and
live free VRAM, then retries exactly once with tiling on CUDA OOM. With the
512px VAE tile, 768 becomes a serial 2×2 decode and 1024 a 3×3 decode.

The measurement protocol and A002 768/1024 tiled-vs-untiled results are in
[VAE runtime policy](docs/VAE_RUNTIME_POLICY.md).

These are reference measurements for a single generation pass on an RTX 5060
Laptop 8GB (A-001, seed 42, 30 steps). Actual time varies with the GPU,
available VRAM, input, and the number of Portrait Mode auto-fill attempts.

| Settings | Wall time | Peak VRAM | Layers |
| --- | ---: | ---: | ---: |
| res 512, head off | 60.7s | 2.03 GiB | 13 |
| res 512, head on | 109.6s | 2.03 GiB | 24 |
| res 1280, head off | 333.9s | 3.66 GiB | 13 |
| res 1280, head on | 609.9s | 3.66 GiB | 24 |

The head stage generates extra frames at the same resolution, so in these
measurements it adds time rather than peak VRAM.

For repeated runs, the fixed body/head semantic prompt embeddings are retained
in a CPU cache on the pipeline instance. This skips text-encoder inference and
GPU transfers after the first run. The engine path also skips the unused
checkerboard previews. Per-call `input_encode_seconds`, `unet_denoise_seconds`,
`transparent_decode_seconds`, and total time are recorded in the Bundle
manifest under `run.pipeline_timing`.

Streaming A/B knobs are available as `run_portrait_pipeline` options
`offload_non_blocking` and `offload_record_stream`. They default to false to
preserve the stable 8GB path; compare wall time, peak VRAM, and fidelity on a
freshly loaded pipeline before enabling them for a host.
On the 8GB profile, requesting both options together automatically disables
both as a safety measure.

### ComfyUI nodes

| Node | Purpose |
| --- | --- |
| SeeThrough Load Source | image, alpha mask, and source name |
| SeeThrough Load LayerDiff Model | load LayerDiff |
| SeeThrough Load Depth Model | load Marigold |
| SeeThrough Generate Layers | basic semantic decomposition |
| SeeThrough Generate Layers (Custom) | Portrait Mode, head detail, auto-fill |
| SeeThrough Generate Depth | per-layer depth maps |
| SeeThrough Post Process | crop, hair/part split, colour restoration |
| SeeThrough Save PSD | PNG, metadata, and PSD download artifacts |

For Portrait Mode, read a transparent PNG with **Load Source** and connect its
`subject_mask` to the Custom node. Opaque backgrounds need a foreground-positive
mask.

## Standalone webui

```bash
pip install -r webui/requirements.txt
python webui/app.py
# http://127.0.0.1:7860
```

Upload one source portrait, choose a production profile, inspect the three
validation axes (Static Reconstruction, Seams, Local Fidelity), and download a
Portrait Bundle zip containing canonical layers and diagnostics. See
[`webui/README.md`](webui/README.md).

Transparent PNGs can still retain the former light background in their RGB
channels. The upload path now repairs only soft-alpha edge pixels near local
opaque foreground (hair and arm/body gaps are typical cases); it does not
alter the alpha mask or raw layers. Existing Bundles need one regeneration to
receive this correction.

## Tests

```bash
python -m pytest tests -q
```

The vendored research UI under `see-through/ui` has separate optional
dependencies; the project migration gate is the root `tests/` suite.

## Documentation

- [`docs/PORTRAIT_BUNDLE_V1.md`](docs/PORTRAIT_BUNDLE_V1.md) — cross-repository file contract
- [`docs/M1_IMPLEMENTATION_SPEC.md`](docs/M1_IMPLEMENTATION_SPEC.md) — Portrait Mode contract
- [`webui/README.md`](webui/README.md) — standalone producer WebUI usage
- [`docs/TEST_PROTOCOL_A001.md`](docs/TEST_PROTOCOL_A001.md) — regression protocol
- Auto-rig design and experiment history live in
  [`portrait-autorig`](https://github.com/prentice7725/portrait-autorig).

## Acknowledgements

Thanks to [@jtydhr88](https://github.com/jtydhr88) and the See-Through research
project contributors.

## License

MIT
