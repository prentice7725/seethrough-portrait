# SeeThrough Portrait

[한국어](README.md)

An independent producer that converts a single anime-style portrait into a
validated **Portrait Bundle v1**. Its supported interfaces are the standalone
WebUI and Python engine; this repository does not include a ComfyUI custom-node
adapter.

```text
source portrait
      ↓
SeeThrough Portrait
      ↓
Portrait Bundle v1
      ↓
Portrait Composer → portrait-autorig
```

This repository owns image decomposition, canonical semantic-layer repair,
static quality validation, and Bundle publishing. Composer and AutoRig are
separate projects; the boundary between them is the Portrait Bundle file
contract, not Python imports.

## Features

- **Portrait Mode and Silhouette Guard** — checks missing subject regions and
  preserves unexplained residual pixels as diagnostic-only `body_remainder`.
- **Production profiles** — `NORMAL` compares one deterministic attempt,
  `QUALITY` compares three, and `HARVEST` compares five. HARVEST is a
  SeeThrough candidate-generation profile.
- **Fidelity repair and diagnostics** — repairs canonical layers against the
  source, then reports Static Reconstruction, Seams, and Local Fidelity
  independently.
- **Optional left/right derivatives** — supported semantics include
  handwear, legwear, and footwear. Best-effort splitting writes to
  `derived/left_right/`; if there are fewer than two alpha-connected
  components, that derivative is omitted. Canonical PNGs in `layers/` are
  never changed.
- **Optional depth derivatives** — written only when validated depth maps are
  supplied through the Python API. The WebUI does not run Marigold implicitly.

## Install and run

Python 3.10 or newer is required. For CUDA, install a matching PyTorch and
torchvision build first.

```bash
python -m pip install -r webui/requirements.txt
python webui/app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) and upload an image. Models
download from Hugging Face into `models/SeeThrough/` on first use. For complete
setup, options, and troubleshooting, see [`webui/README.md`](webui/README.md).

Supported models:

| Model | Hugging Face repository | Purpose |
| --- | --- | --- |
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | semantic layer generation |
| Marigold Depth | `layerdifforg/seethroughv0.0.1_marigold` | optional depth estimation |

An 8GB-class GPU can use UNet block streaming. VAE tiling is selected from the
diffusion-stage resolution and available VRAM; an untiled CUDA OOM is retried
once with tiling. Measurements and policy are documented in
[`docs/VAE_RUNTIME_POLICY.md`](docs/VAE_RUNTIME_POLICY.md).

## Portrait Bundle v1

`layers/` contains the canonical `production_repaired` layers for downstream
consumers. `raw_layers/` is forensic output and must not be used for authoring
or downstream layer input. `body_remainder` is also diagnostic-only and is
written to `diagnostics/body_remainder.png`.

```text
A001.portrait/
├─ manifest.json
├─ original.png
├─ layers/                 # canonical semantic layers
├─ raw_layers/             # optional forensic output; not consumer input
├─ derived/                # optional left/right and depth derivatives
└─ diagnostics/            # report, fidelity, seams, masks, composites
```

The manifest uses `left` and `right` for **geometric canvas sides**, not the
character's anatomical sides.

```json
{
  "derived": {
    "source_stage": "production_repaired",
    "left_right": {
      "status": "computed",
      "paths": {
        "handwear": {
          "left": "derived/left_right/handwear_left.png",
          "right": "derived/left_right/handwear_right.png"
        }
      }
    },
    "depth": { "status": "not_computed", "paths": {} }
  }
}
```

Derivatives are generated only when explicitly requested and never replace
canonical `layers/`. The left/right API supports handwear, legwear, footwear,
and eye- and ear-related semantics. See [`docs/PORTRAIT_BUNDLE_V1.md`](docs/PORTRAIT_BUNDLE_V1.md)
for invariants, manifest validation, and the JSON Schema.

Python callers can request left/right output with
`save_portrait_bundle(..., stratify_left_right=True)` and optionally pass
`depth_maps={tag: float32_HxW}`. These are producer derivatives for downstream
review, not automatic replacements for canonical layers.

## Next steps and project docs

- [Portrait Bundle v1 contract](docs/PORTRAIT_BUNDLE_V1.md) — file contract and schema
- [WebUI guide](webui/README.md) — setup, generation options, results, troubleshooting
- [Portrait Mode specification](docs/M1_IMPLEMENTATION_SPEC.md)
- [Regression protocol](docs/TEST_PROTOCOL_A001.md)
- [VAE runtime policy](docs/VAE_RUNTIME_POLICY.md)
- [P0 closeout](docs/P0_CLOSEOUT_V0.2.md)

Auto-rigging is maintained in the separate
[`portrait-autorig`](https://github.com/prentice7725/portrait-autorig) repository.

## Tests

```bash
python -m pytest tests -q
```

The vendored `see-through/ui` is a research UI with separate optional
dependencies. This repository's tests are under `tests/`.

## License

MIT
