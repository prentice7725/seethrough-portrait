# SeeThrough Portrait (ComfyUI Fork)

[한국어](README.md)

![Preview](https://raw.githubusercontent.com/tackcrypto1031/tk_seethrough/main/workflows/sample.png)

A fork of [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through)
by [@jtydhr88](https://github.com/jtydhr88) that decomposes an anime-style
illustration into transparent semantic layers (hair, face, topwear, etc.),
plus a **Portrait Mode** built on top for upper-body character art: a
Silhouette Guard that recovers pixels the model omits (missing sleeves,
shoulders, arms) so the final composite never loses more of the subject than
the model itself generated.

Two ways to run it:

| | ComfyUI node graph | Standalone webui |
| --- | --- | --- |
| Requires ComfyUI | yes | no |
| Best for | full pipeline: depth, hair L/R split, PSD/Spine export | quickly running Portrait Mode on one image and reading the verdict |
| Get started | [Installation](#installation-comfyui) below | [`webui/README.md`](webui/README.md) |

Both paths run the same underlying model-loading and layer-generation code
(`seethrough_engine/`), so a result from one is directly comparable to the
other given the same seed and settings.

## Portrait Mode

Portrait Mode is an upper-body recovery profile for front-facing character
portraits. It does not retrain the model or guarantee a semantic arm layer;
instead it compares the subject's real silhouette against the union of
generated layers and recovers whatever pixels are missing into a
`body_remainder` safety layer, so the composited result never loses more of
the subject than the model actually omitted.

- **Silhouette Guard** clips any layer spill to the trusted subject mask and
  recovers unexplained original pixels as `body_remainder`.
- **Crop-aware critical groups** mean absent legs or handwear alone don't
  fail an otherwise-complete upper-body decomposition.
- **Silhouette-aware auto-fill** re-runs inference (up to 5 times) and keeps
  whichever run's layers give the best coverage/similarity, not just the
  first result.
- **Two verdicts, reported separately** — a `recovery_verdict` (did the
  Silhouette Guard reconstruct the full silhouette?) and an overall
  `verdict` that also folds in semantic completeness (PASS / SOFT_PASS /
  SOFT_PASS_LOW_CONFIDENCE / REWORK / FAIL). A 100%-reconstructed
  silhouette is not by itself evidence that FACE/HAIR/BODY were separated
  correctly.

Every run produces `body_remainder`, coverage/missing/spill mask PNGs, a
reconstruction preview, and a `*_portrait_report.json` with the full
diagnostics. See [`docs/M1_IMPLEMENTATION_SPEC.md`](docs/M1_IMPLEMENTATION_SPEC.md)
for the exact contract and verdict rules, and
[`docs/TEST_PROTOCOL_A001.md`](docs/TEST_PROTOCOL_A001.md) for the validation
procedure (A-001) both entry points are checked against.

**In ComfyUI:** use **SeeThrough Load Source** for transparent PNG input and
connect its `subject_mask` output into **Generate Layers (Custom)**, then
enable `portrait_mode` (and keep `silhouette_guard` on). For an opaque
background, provide a separate foreground mask. See
[SeeThrough Generate Layers (Custom)](#seethrough-generate-layers-custom)
below for the full parameter table.

**In the standalone webui:** just upload a transparent-background PNG --
Portrait Mode is always on, and the real alpha channel is read directly as
the trusted subject mask. See [`webui/README.md`](webui/README.md).

## Installation (ComfyUI)

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tackcrypto1031/tk_seethrough.git
```

Install dependencies:

```bash
cd tk_seethrough
pip install -r requirements.txt
```

Restart ComfyUI. The nodes will appear under the `SeeThrough` category.

### Models

Models are downloaded automatically from HuggingFace on first use:

| Model | HuggingFace Repo | Purpose |
|-------|-------------------|---------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | SDXL-based transparent layer generation |
| Marigold Depth | `layerdifforg/seethroughv0.0.1_marigold` | Fine-tuned monocular depth for anime |

You can also download models manually and place them in `ComfyUI/models/SeeThrough/`.

## Usage (ComfyUI)

1. Add **SeeThrough Load LayerDiff Model** and **SeeThrough Load Depth Model**
2. Add **SeeThrough Generate Layers (Custom)** — connect both models and a **Load Image** node
3. Uncheck `enable_head_detail` if you want faster processing without head detail layers
4. Connect to **SeeThrough Generate Depth** → **SeeThrough Post Process** → **SeeThrough Save PSD**
5. Run the workflow and click **Download PSD** to export

**For upper-body portraits:** use **SeeThrough Load Source**, connect
`subject_mask`, enable `portrait_mode`, and keep `silhouette_guard` enabled.

**For Spine export:** Replace step 4's **Save PSD** with **Layer Rename** → **Layer Filter** → **Export Spine**. Open the output JSON in Spine editor.

## All Nodes

| Node | Description |
|------|-------------|
| **SeeThrough Load LayerDiff Model** | Load the LayerDiff SDXL pipeline |
| **SeeThrough Load Depth Model** | Load the Marigold depth estimation pipeline |
| **SeeThrough Generate Layers** | Original layer generation (all stages, all layers) |
| **SeeThrough Generate Layers (Custom)** | Layer generation with `enable_head_detail` toggle, `auto_fill`, and Portrait Mode |
| **SeeThrough Generate Depth** | Depth map estimation per layer |
| **SeeThrough Post Process** | Left/right splitting, hair clustering, color restoration |
| **SeeThrough Save PSD** | Export layers as PNGs + metadata; download Best PSD, Depth PSD, or All Runs PSD via browser |
| **SeeThrough Layer Rename** | Rename layer tags to Spine-friendly names (customizable) |
| **SeeThrough Layer Filter** | Include/exclude specific layers before export |
| **SeeThrough Export Spine** | Export layers as a Spine 2D skeleton project (JSON + images) |

### SeeThrough Generate Layers (Custom)

A new node `SeeThrough_GenerateLayers_Custom` that adds parameters compared to the original `SeeThrough Generate Layers`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_head_detail` | true | v3 models only: toggle the head detail inference stage on/off |
| `auto_fill` | false | Auto-fill missing layers: re-runs inference (up to 5 times) until all expected layers are generated (v3+head=24, v3 body=13, v2=19) |
| `min_alpha_coverage` | 0.01 | Minimum alpha coverage ratio to consider a layer valid. Only used when `auto_fill` is enabled |
| `portrait_mode` | false | Use upper-body critical groups and silhouette-aware run selection instead of requiring full-body parts |
| `silhouette_guard` | true | In Portrait Mode, clip spill and recover unexplained source pixels as `body_remainder` |
| `subject_mask` | optional | Foreground-positive mask (`white = subject`). Strongly recommended for opaque-background portraits |

#### How It Works

The v3 See-through model runs in **two inference stages**:

1. **Body stage** — Generates 13 body-level layers (front hair, back hair, head, neck, neckwear, topwear, handwear, bottomwear, legwear, footwear, tail, wings, objects)
2. **Head stage** — Crops the head region from stage 1, upscales it, and runs a second inference pass to generate 11 fine-grained head layers (headwear, face, irides, eyebrow, eyewhite, eyelash, eyewear, ears, earwear, nose, mouth)

Each stage is a full diffusion pipeline call. By setting `enable_head_detail = false`, the entire head stage is **skipped** (no GPU computation), saving approximately **50% of the total inference time**.

This is useful when you only need body-level decomposition and don't require fine-grained facial features.

#### Multi-Run Auto-Fill

The diffusion model is stochastic — each run may produce slightly different results. Sometimes a layer (e.g., face or hand) is missing or nearly empty in one run but present in another.

Enable `auto_fill` to automatically re-run inference until all expected layers are generated with good quality:

1. **Run 1** uses the original seed — this is the primary result
2. Each layer is compared against the original image to compute a **similarity score** (0~1)
3. Layers that are **missing** (alpha coverage below threshold) or have **low similarity** (< 0.85) trigger additional runs
4. **Run 2** uses `seed + 1`, **Run 3** uses `seed + 2`, etc.
5. For each layer, the version with the **highest similarity to the original** is kept
6. The process repeats up to **5 runs** or until all layers have good similarity

This means even if Run 1 generates a face layer, if Run 2 produces a face that better matches the original image, Run 2's version will be used automatically.

In Portrait Mode, auto-fill instead re-runs until silhouette coverage and
critical semantic groups both pass, keeping whichever run's layer *set*
scores best overall (similarity + silhouette coverage + shoulder/arm
region, minus spill), not just per-layer similarity.

Models are loaded to GPU only once across all runs — the overhead is only the additional diffusion time, not model loading.

> **Note:** For v2 models, this toggle has no effect since v2 uses a single-stage inference.

### Spine Export Workflow

For [Spine](http://esotericsoftware.com/) animation preparation, connect:

```
PostProcess → Layer Rename (optional) → Layer Filter (optional) → Export Spine
```

#### Layer Rename

Maps internal tags to Spine-friendly names. Has built-in defaults for all tags. The `custom_mapping_json` field is **optional** — leave it empty to use defaults.

**When to use it:**
- You want clean, readable names in Spine (e.g. `front-hair` instead of `hairf`)
- Your team has a naming convention and you need custom names

**Built-in default mapping (partial list):**

| Original tag | → Renamed to |
|-------------|-------------|
| `hairf` | `front-hair` |
| `hairb` | `back-hair` |
| `eyel` | `eye-left` |
| `eyer` | `eye-right` |
| `handwearl` | `handwear-left` |
| `handwearr` | `handwear-right` |
| `earl` | `ear-left` |
| `earr` | `ear-right` |
| `topwear` | `topwear` (unchanged) |
| `face` | `face` (unchanged) |

> Tags that already have clean names (e.g. `face`, `head`, `nose`) are kept as-is.

**Custom mapping example:** To override specific names, enter a JSON object in `custom_mapping_json`:

```json
{
  "hairf": "bangs",
  "hairb": "back-hair",
  "topwear": "shirt",
  "bottomwear": "skirt",
  "handwearl": "left-glove",
  "handwearr": "right-glove"
}
```

Only the tags you specify in the JSON will be overridden — all other tags still use the built-in defaults. Invalid JSON is ignored with a warning.

#### Layer Filter

Removes unwanted layers using include or exclude mode. All available tags are pre-filled by default — delete the ones you don't need. Enter one tag per line.

> **Tip:** If Layer Rename is connected before Layer Filter, use the **renamed** tag names (e.g. `front-hair`). If not using Layer Rename, use original tags (e.g. `hairf`).

#### Export Spine

Outputs a folder with a configurable output path (defaults to ComfyUI output directory):

- `{prefix}.json` — Spine skeleton file (open directly in Spine editor)
- `images/` — cropped PNG files for each layer
- Set `output_path` to export to a custom directory (e.g. `D:/my_project/spine_assets`)

Coordinates are automatically converted from image space (Y-down) to Spine space (Y-up, origin at bottom-center). Draw order follows depth ordering from PostProcess.

#### PSD Import vs JSON Export — Which Should I Use?

Spine Professional (3.6+) can import PSD files directly, so you may wonder whether this JSON export is needed. Here's the comparison:

| | Save PSD → Spine PSD Import | Export Spine (JSON + images) |
|---|---|---|
| **Spine version** | Professional 3.6+ only | **All versions** (Essential + Professional) |
| **Layer positioning** | Automatic | Automatic (coords pre-converted) |
| **Layer naming** | Depends on PSD layer names | Controllable via LayerRename |
| **Layer filtering** | Must hide/delete in PSD first | Built-in LayerFilter node |
| **Iteration** | Re-import PSD to update images | Re-export to update |
| **Bone hierarchy** | Not auto-created | Not auto-created |
| **Best for** | Spine Professional users who want a quick start | Spine Essential users, or teams wanting pre-filtered/renamed layers in an automated pipeline |

**Recommendation:**
- **Spine Professional users** → Use **Save PSD** and import via Spine's built-in PSD import. It's the simplest workflow.
- **Spine Essential users** → Use **Export Spine**, as Essential does not support PSD import.
- **Automated pipelines** → Use **Export Spine** with LayerRename + LayerFilter for consistent, pre-processed output.

<details>
<summary>Available layer tags (after LayerRename, 38 tags)</summary>

| Category | Tags |
|----------|------|
| Hair | `front-hair`, `back-hair` |
| Head | `head`, `headwear` |
| Face | `face`, `nose`, `mouth` |
| Eyes | `eye-left`, `eye-right`, `eyewear` |
| Eye detail | `irides`, `irides-left`, `irides-right`, `eyebrow`, `eyebrow-left`, `eyebrow-right`, `eye-white`, `eye-white-left`, `eye-white-right`, `eyelash`, `eyelash-left`, `eyelash-right` |
| Ears | `ears`, `ear-left`, `ear-right`, `earwear` |
| Body | `neck`, `neckwear`, `topwear`, `bottomwear` |
| Limbs | `handwear`, `handwear-left`, `handwear-right`, `legwear`, `footwear` |
| Other | `tail`, `wings`, `objects` |

If not using LayerRename, use original tags: `hairf`, `hairb`, `eyel`, `eyer`, `handwearl`, `handwearr`, `earl`, `earr`, etc.

</details>

## Standalone webui

`webui/app.py` runs Portrait Mode on a single image with no ComfyUI process
involved -- upload, run, read the verdict, download the layers and report as
a zip. It shares its model-loading and layer-generation code
(`seethrough_engine/`) with the ComfyUI nodes above, so it is not a
second, drifting implementation.

```bash
pip install -r webui/requirements.txt
python webui/app.py
# open http://127.0.0.1:7860
```

It can export a **Spine project** -- tick the box and the run's zip gains a
skeleton JSON plus the cropped layer PNGs it references. Draw order follows
Portrait Mode's fixed tag order by default; turning on depth ordering runs
Marigold and sorts the way the ComfyUI graph does (downloading a 3 GB model on
first use).

It does not (yet) do hair L/R splitting or PSD -- those stay ComfyUI-only for
now. Full install/usage instructions,
known limitations, and the implementation contract are in
[`webui/README.md`](webui/README.md) and
[`docs/M2_IMPLEMENTATION_SPEC.md`](docs/M2_IMPLEMENTATION_SPEC.md).

## 2.5D rig preview (M4)

An **automatic rig** built from Portrait Mode's layers, animated in the browser.
No Spine, no Live2D, no bone hierarchy -- vertex weights and one manifest. A
neck whose top follows the head while its bottom stays with the body is awkward
to express with two bones and is two lines as a weight gradient.

- A run writes `{base}_rig_manifest.json` and `rig/images/` (a webui checkbox,
  on by default).
- Open [`webui/rig_preview/index.html`](webui/rig_preview/index.html), pick the
  run directory, and it animates: head turn and tilt, breathing, blink, and an
  ellipsoid shell rotation. No server; it runs from `file://`.
- A run made before any of this can be re-rigged without the GPU pass that
  produced it: `python -m seethrough_engine.rig <run dir>`. Stages A-D read
  `{tag: full-canvas RGBA}` and nothing else, and those layers are already on
  disk beside the report.

### Expression pack

**A shut eye and an open mouth are drawings the decomposition cannot produce.**
A-001's `mouth` is a closed mouth and there is no closed eye at all. An image
model can draw them -- used as a **donor rather than as a portrait**: only the
region it changed is taken, and every other pixel stays the original's. That is
the difference from the earlier modular attempt, where the generated image was
the portrait and the character's identity went with the generator's drift.

```bash
# recover just the regions a donor image changed (no GPU)
python -m seethrough_engine.expression <run> eye_closed=<png> mouth_open=<png>

# or decompose the donor in its own right and transplant its layers -- the
# matte is then the one the model drew, rather than one inferred from a diff
python -m seethrough_engine.expression <run> --from-run <donor run> eye_closed mouth_open
```

The donor need not share the base's crop; it is registered by silhouette. What
an expression may repaint is decided **by the layers**: only pixels whose
topmost layer is the face's own skin or features, so a differently-drawn lock of
hair cannot come along. Turning off `Use expression art` in the preview falls
back to the v0.1 blink (the lash squashed onto the lid line) for comparison.

### Composite fidelity

Rendering the rig is what made **decomposition faults visible** for the first
time. All three were already in the composite metric; nothing had been looking:

| | what | how it showed |
| --- | --- | --- |
| `reclaim_occluded` | pixels a garment and the neck both claim | the garment holding the skin from its own opening, hiding the neck |
| `fit_edge_alpha` | solves a layer's edge alpha against the original | a black stroke under the jaw; a double chin |
| `fit_layer_tone` | each layer's colour bias, one constant per material | a step where two layers meet, drawn as a line across the neck |

Composite mae **18.30 → 9.21**, bad ratio **8.57% → 5.42%** -- on layers the
model had already produced, with no GPU pass.

The design and the measurements are in
[`docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md`](docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md).
M4's verdict is **conditionally viable**.

## What's New in This Fork

### v1.4.0.dev4 — Portrait Mode M4: the 2.5D rig and the expression pack

- **Automatic 2.5D rig** — `seethrough_engine/rig.py` turns Portrait Mode
  layers into `rig_manifest.json`: vertex weights instead of bones (the neck a
  gradient from head to body), a region-aware remainder split, left/right eye
  separation, and anchors derived from alpha.
- **Browser preview** — `webui/rig_preview/index.html`: head turn and tilt,
  breathing, blink, ellipsoid shell rotation. No server, runs from `file://`.
- **Expression pack** — `seethrough_engine/expression.py`: a generated image
  used as a donor rather than as a portrait, either by recovering the regions
  it changed or by transplanting the layers of its own decomposition.
- **Composite fidelity** — the faults rendering the rig turned up: contested
  pixels returned, edge alpha solved against the original, per-material colour
  bias removed. mae 18.30 → 9.21, bad ratio 8.57% → 5.42%.
- **`python -m seethrough_engine.rig <run>`** — re-rig an old run, no GPU.

### v1.4.0.dev3 — Spine export from the standalone webui

The webui can now produce a Spine 2D project -- skeleton JSON plus the cropped
layer PNGs it references -- delivered as `spine/` inside the run's zip.

- **New `seethrough_engine/spine.py`**: the coordinate conversion and skeleton
  JSON are one implementation now. The `SeeThrough_ExportSpine` node delegates
  to it, which deleted 110 lines, and the tag-to-Spine name mapping is no
  longer duplicated.
- **New `seethrough_engine/depth.py`**: the batched Marigold call and the
  v2-slot folding are shared with `SeeThrough_GenerateDepth`.
- **Two draw orders**: the default is `SEMANTIC_Z_ORDER`, a fixed back-to-front
  order over Portrait Mode's tag vocabulary, needing no extra model and no
  extra time. Ticking "depth-based draw order" runs Marigold and matches the
  ComfyUI graph (3 GB download on first use, ~15s per run).

One layer is beyond depth estimation either way: the depth batch is indexed by
the v2 tag list, which has no `head`. ComfyUI drops such a layer outright; this
interpolates it from its semantic neighbours instead, since losing the head is
the worse outcome for a rig.

### v1.4.0.dev2 — actually running on an 8GB GPU

The UNet this fork uses is 4.07B parameters -- **7.58 GiB even in bf16**. On
an RTX 5060 Laptop (8151 MiB), once the CUDA context and the desktop have
taken their 0.75 GiB, torch can hold 7.19 GiB, so **the weights could not be
put on the card at all**: `unet.to(device)` raised OutOfMemoryError before any
compute started. Lowering `resolution` or `steps` could not help, because the
failure came before either of them mattered. (With Windows' CUDA system-memory
fallback enabled it silently spills to host RAM instead of failing -- it does
not die, it just runs ~90x slower.)

- **UNet block streaming** — when the weights will not fit whole, they stay on
  the CPU and are streamed to the GPU a leaf at a time (diffusers group
  offloading with CUDA-stream prefetch). The decision is made by measuring
  free VRAM, so a larger card behaves exactly as before, and the ComfyUI node
  path is untouched -- it has ComfyUI's own VRAM scheduler.
- **Load at the target dtype** — `from_pretrained` was instantiating at the
  default fp32 and upcasting the bf16 checkpoint on the way in, only to cast
  it straight back. Peak system RAM for a model load fell from
  **17.51 GiB to 0.82 GiB**, and UNet load time from 20.2s to 8.0s.
- **VAE tiling** — the SDXL VAE encodes/decodes in 512px tiles. Inactive at
  `resolution` 512 or below.

Measured on that card (A-001, seed 42, 30 steps, all `PASS`):

| settings | wall time | peak VRAM | layers |
|---|---|---|---|
| res 512, head off | 60.7s | 2.03 GiB | 13 |
| res 512, head on | 109.6s | 2.03 GiB | 24 |
| res 1280, head off | 333.9s | 3.66 GiB | 13 |
| res 1280, head on | 609.9s | 3.66 GiB | 24 |

The res-512/head-off row reproduces the previously recorded coverage table
field for field -- streaming the weights does not change results. Peak VRAM
depends only on resolution: the head stage runs at the same size with fewer
frames, so it costs time and no extra memory.

### v1.4.0.dev1 — Portrait Mode M2: standalone webui

- New `webui/app.py`: a single-image Portrait Mode webui that runs without
  ComfyUI (see [Standalone webui](#standalone-webui) above).
- New `seethrough_engine/` package: the ComfyUI-independent model-loading and
  layer-generation core, shared by `nodes.py` and the webui so there is one
  implementation of the GPU-facing pipeline, not two. `nodes.py`'s node
  behavior is unchanged; see `docs/M2_IMPLEMENTATION_SPEC.md` for what moved
  where and why.
- Fixed a spec/config mismatch from M1: the post-recovery coverage floor
  that gates a hard `FAIL` verdict was `0.995` in
  `config/portrait_defaults.json` but documented (and intended) as `0.999`
  in `M1_IMPLEMENTATION_SPEC.md` / `TEST_PROTOCOL_A001.md`. Runs between
  99.5% and 99.9% post-recovery coverage now correctly fail instead of
  passing.

### v1.3.0.dev1 — Portrait Mode M1

- Upper-body Portrait Mode with crop-aware semantic groups
- Silhouette Guard that clips layer spill to a trusted subject mask
- `body_remainder` recovery layer for unexplained original pixels
- Silhouette-aware multi-run selection and JSON diagnostics
- Separate recovery and semantic verdicts

### v1.2.8 — Issue #5

- New node **SeeThrough Load Source**: same dropdown as ComfyUI's LoadImage, plus outputs `source_filename` for preserving the original filename in PSD output.
- **SeeThrough Save PSD** now accepts optional `original_image` + `source_filename` inputs, automatically includes the original input image as a visible base layer in the generated PSD, and uses the source filename in output filenames when available.
- PSD layer structure is now grouped: `Original` (visible, bottom), `Parts` (hidden), `Runs` (hidden, grouped-PSD mode only) — opens the PSD on the original so you can toggle groups to edit specific parts.

### v1.2 — Spine Export, Auto-Fill & All Runs PSD

- **Spine Export Nodes** — New `Layer Rename`, `Layer Filter`, and `Export Spine` nodes for Spine 2D animation preparation.
- **Auto-Fill Missing Layers** — Enable `auto_fill` on GenerateLayers (Custom) to automatically re-run inference up to 5 times, filling missing layers and upgrading low-quality layers by comparing against the original image.
- **All Runs PSD** — When `auto_fill` is enabled, a new "Download All Runs PSD" button appears on Save PSD. It creates a single PSD with group folders for each run, so you can manually compare and pick layers.
- **PSD Download Buttons** — Save PSD now has 3 buttons:
  - **Download PSD** (green) — best layers after auto-fill selection
  - **Download Depth PSD** (purple) — depth maps
  - **Download All Runs PSD** (orange) — all runs grouped by folder (requires `auto_fill`)

## Project docs

For the full Portrait Mode design/implementation record:

- [`docs/PORTRAIT_MODE_FORK_PLAN_v0.1.md`](docs/PORTRAIT_MODE_FORK_PLAN_v0.1.md) — overall scope and milestones (M1–M4)
- [`docs/M1_IMPLEMENTATION_SPEC.md`](docs/M1_IMPLEMENTATION_SPEC.md) — Portrait Mode core contract and verdict rules
- [`docs/TEST_PROTOCOL_A001.md`](docs/TEST_PROTOCOL_A001.md) — the A-001 validation procedure both entry points are checked against
- [`docs/M2_IMPLEMENTATION_SPEC.md`](docs/M2_IMPLEMENTATION_SPEC.md) — standalone webui contract, what's shared with `nodes.py` and why, and the measured VRAM numbers for an 8GB card
- [`docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md`](docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md) — M4: the automatic 2.5D rig, the browser preview, the expression pack, and the composite-fidelity faults the rig turned up on the way

## Acknowledgements

This project is a fork of [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) by [@jtydhr88](https://github.com/jtydhr88). Huge thanks for creating the original ComfyUI integration.

The underlying research is [See-through](https://github.com/shitagaki-lab/see-through) by [shitagaki-lab](https://github.com/shitagaki-lab).
Paper: [arxiv:2602.03749](https://arxiv.org/abs/2602.03749) (Conditionally accepted to ACM SIGGRAPH 2026)

PSD generation uses [ag-psd](https://github.com/nicasiomg/ag-psd) in the browser.

## License

The ComfyUI fork code is marked MIT. The bundled upstream `see-through`
research code retains its Apache-2.0 license in `see-through/LICENSE`.
