# VAE runtime tiling policy

VAE tiling is selected immediately before each body and head diffusion call;
it is not a model-load property. The policy prefers untiled when CUDA's live
free VRAM is at least the stage's quadratic reserve:

```text
reserve = 256 MiB + (resolution² × 1024 bytes) + 512 MiB
```

That is approximately 1.31 GiB at 768 and 1.75 GiB at 1024. It is a policy
threshold, not an OOM guarantee: an untiled CUDA OOM clears the cache and
retries the same stage once with tiled VAE. Tiled calls and non-OOM failures do
not retry.

## A002 tiled/untiled A/B

Measured on the RTX 5060 Laptop 8GB, same A002 source, seed 42, 30 steps,
body plus head enabled. “Peak” is the largest allocation above each stage's
baseline; it is intentionally separated from the 7.58 GiB streamed UNet
weights.

| Canvas | Mode | Wall time | Pipeline call time* | Peak delta | Output delta vs. other mode | Eye-local result |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 768 | tiled (2×2) | 265.95s | 218.53s | 1.18 GiB | RGB MAE 0.498 | pass; eyewhite present |
| 768 | untiled | 247.90s | 212.92s | 1.35 GiB | RGB MAE 0.498 | pass; eyewhite present |
| 1024 | tiled (3×3) | 419.91s | 390.31s | 2.09 GiB | RGB MAE 0.679 | **review**; visible sclera lost |
| 1024 | untiled | 401.93s | 375.34s | 2.39 GiB | RGB MAE 0.679 | **pass**; sclera retained |

No measured untiled stage OOMed. At 1024, the head stage began with 1.80 GiB
driver-visible free VRAM and completed untiled, so a larger 3 GiB reserve would
unnecessarily choose tiling and reproduce the A002 eye-local regression. The
policy therefore uses the measured 1.75 GiB reserve and retains its one-time
tiled fallback for hosts with less available memory.

\* The historical “VAE-stage time” values above wrap the complete diffusion
call (input VAE encode, UNet denoising, and transparent-layer decode), not only
the VAE. New runs split these sections in `run.pipeline_timing` as
`input_encode_seconds`, `unet_denoise_seconds`, and
`transparent_decode_seconds`.

The VAE's encoder samples from CUDA's global RNG, so tiled and untiled outputs
are not bit-identical even with the same seed. The benchmark records both
cross-mode RGB delta and static source-composite fidelity for future runs.

## Reproduction

```powershell
.\.venv\Scripts\python.exe tools\benchmark_vae_tiling.py `
  webui\outputs\20260902_061821_4fdd3ec8.portrait\original.png `
  --output webui\outputs\vae_tiling_ab_a002.json
```
