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
portrait-autorig
    ↓
Rig Bundle
```

This repository ends at `Image → validated, production-ready semantic portrait
bundle`. Auto-rigging, eye subdivision, anchors, meshes, weights, expression
packs, Spine export, and the browser runtime live in the separate
[`portrait-autorig`](https://github.com/prentice7725/portrait-autorig)
repository. The projects share a file contract and no Python imports.

## Portrait Mode

- **Silhouette Guard** clips spill and recovers unexplained original pixels as
  `body_remainder`.
- **Portrait-aware auto-fill** selects the best layer set from up to five runs.
- **Fidelity repair** runs `reclaim_occluded → fit_layer_tone → fit_edge_alpha
  → fit_seam_residual` against the original still image.
- **Static validation** measures whole-composite fidelity and thin, continuous
  seam artifacts separately.
- **Two verdicts** distinguish silhouette recovery from semantic completeness.

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
   ├─ fidelity.json
   ├─ seams.json
   └─ coverage/missing/spill/composite PNGs
```

Downstream consumers use only `layers/`. `raw_layers/` records model output for
debugging and is not a fallback asset source. See
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

Upload one image, run Portrait Mode, inspect the verdict, and download a
Portrait Bundle zip containing canonical layers and diagnostics. See
[`webui/README.md`](webui/README.md).

## Tests

```bash
python -m pytest tests -q
```

The vendored research UI under `see-through/ui` has separate optional
dependencies; the project migration gate is the root `tests/` suite.

## Documentation

- [`docs/PORTRAIT_BUNDLE_V1.md`](docs/PORTRAIT_BUNDLE_V1.md) — cross-repository file contract
- [`docs/M1_IMPLEMENTATION_SPEC.md`](docs/M1_IMPLEMENTATION_SPEC.md) — Portrait Mode contract
- [`docs/M2_IMPLEMENTATION_SPEC.md`](docs/M2_IMPLEMENTATION_SPEC.md) — standalone webui
- [`docs/TEST_PROTOCOL_A001.md`](docs/TEST_PROTOCOL_A001.md) — A-001 validation
- Auto-rig design and experiment history live in
  [`portrait-autorig`](https://github.com/prentice7725/portrait-autorig).

## Acknowledgements

Thanks to [@jtydhr88](https://github.com/jtydhr88) and the See-Through research
project contributors.

## License

MIT
