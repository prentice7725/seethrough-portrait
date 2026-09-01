# SeeThrough Portrait -- standalone webui (M2)

A single-image Portrait Mode webui that runs without ComfyUI. Upload one
upper-body portrait, run A-001, and get the decomposed layers, the Silhouette
Guard diagnostics, and the PASS / SOFT_PASS / REWORK / FAIL verdict. The
downloadable zip is a versioned Portrait Bundle containing canonical,
production-repaired layers.

This is the M2 milestone from `PORTRAIT_MODE_FORK_PLAN_v0.1.md`: "single-image
WebUI" whose exit condition is "A-001 can be run and exported from the UI."
See `docs/M2_IMPLEMENTATION_SPEC.md` for the full contract and known
limitations.

## Install

```bash
# 1. Install a CUDA-matched torch + torchvision build first (both from the
#    same command, so their CUDA builds match):
#    https://pytorch.org/get-started/locally/
# 2. Then everything else:
pip install -r webui/requirements.txt
```

Models download automatically from HuggingFace on first use, into
`models/SeeThrough/` at the repo root (or drop a pre-downloaded checkpoint
folder there yourself -- see the main [README](../README_EN.md#models)).

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
4. Read the verdict badge and reasons, browse the layer/diagnostic thumbnails,
   and download the `.portrait` bundle zip.

Runs are also kept under `webui/outputs/<timestamp>_<id>.portrait/` (gitignored)
in case you want to inspect them without re-downloading the zip.

## Known limitations (M2 scope)

- **Partial PostProcess stage.** Layers are exported at the model's padded
  square working resolution (`fullpage`), not un-padded back to the original
  image's aspect ratio/size, and there is no hair L/R splitting -- that part of
  `SeeThrough Post Process` stays ComfyUI-only for now.
- **No rig or Spine export.** Those consume Portrait Bundle v1 in the separate
  [`portrait-autorig`](https://github.com/prentice7725/portrait-autorig)
  project, keeping torch/diffusers out of animation tools and runtimes.
- **No PSD export.** The ComfyUI extension builds a PSD client-side in the
  browser via `ag-psd`; this webui exports plain PNGs + JSON instead.
- **One model resident at a time.** Switching the model dropdown unloads the
  previously loaded one.

## Troubleshooting

### The process just dies mid-run with no Python error, after the GPU was pegged at 100% for a long time

This is almost always **Windows TDR** (Timeout Detection and Recovery), not a
bug in this code. If a single CUDA kernel runs longer than the driver's
timeout (default 2 seconds), Windows force-resets the GPU driver, and
whatever process was using CUDA at that moment is killed outright -- below
Python's exception handling, so nothing gets logged.

**How to confirm:** Event Viewer (`eventvwr.msc`) -> Windows Logs -> System
-> look for an **Error** from source `nvlddmkm`, event ID **153**, timestamped
around when it died.

**Fix:** raise the timeout (admin PowerShell), then **reboot** (required):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" `
  -Name "TdrDelay" -PropertyType DWord -Value 60 -Force
```

This is especially likely to hit on a very new GPU generation (e.g. an RTX
50-series / Blackwell card): the **first** CUDA call on a shape/architecture
combination the driver hasn't seen before can trigger a slow cuDNN/cuBLAS
kernel autotune search that legitimately takes several minutes -- comfortably
past the default 2-second TDR window, even though nothing is actually hung.
Once `TdrDelay` is raised, a first run in the tens of minutes for a brand-new
GPU + very recent torch/CUDA build is plausible; subsequent runs in the same
process should be much faster since the kernel search doesn't repeat for
shapes already seen.

If the console shows literally nothing after `[SeeThrough] ... model loaded
to CPU` for a long time with GPU usage pegged near 100%, that alone isn't a
red flag: the pipeline's own diffusion-step progress bar (`tqdm`, enabled by
default) only starts printing once the sampling loop itself begins, so long
setup-time silence upstream of that (text encoding, moving weights to GPU,
first-call kernel autotuning) is expected to be quiet.
