"""Per-diffusion-call VAE tiling policy and CUDA telemetry.

VAE tiling is a decode-time trade-off, not a property of a loaded checkpoint.
The LayerDiff VAE has a 512px tile sample size: a 768 canvas already becomes a
2 x 2 tiled decode and a 1024 canvas becomes 3 x 3.  Repeating that serial
work for every semantic layer is expensive, so callers choose the cheaper
untiled path whenever the *current* CUDA budget supports it and retry once
with tiling only after a CUDA OOM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Literal, TypeVar

import torch

from .device import empty_cache, free_vram_bytes

VAEMode = Literal["untiled", "tiled", "unsupported"]

# This reserve covers the VAE's non-image activations and leaves a margin for
# the streamed UNet group's resident working set.  The image term is
# deliberately quadratic: doubling a square canvas quadruples activation
# storage.  An OOM still gets exactly one tiled retry, so this estimate is a
# conservative preference rather than an assumed hardware truth.
VAE_UNTILED_BASE_RESERVE_BYTES = 256 * 2**20
VAE_UNTILED_BYTES_PER_PIXEL = 1024
VAE_UNTILED_SAFETY_BYTES = 512 * 2**20
DEFAULT_TILE_SAMPLE_SIZE = 512
DEFAULT_TILE_OVERLAP = 0.25

_Result = TypeVar("_Result")


@dataclass(frozen=True)
class VAERuntimeDecision:
    stage: str
    resolution: int
    mode: VAEMode
    reason: str
    free_vram_bytes: int | None
    untiled_reserve_bytes: int
    tile_sample_size: int | None
    estimated_tiles_per_axis: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tile_properties(vae: Any) -> tuple[int, float]:
    tile_size = int(getattr(vae, "tile_sample_min_size", DEFAULT_TILE_SAMPLE_SIZE))
    overlap = float(getattr(vae, "tile_overlap_factor", DEFAULT_TILE_OVERLAP))
    return max(1, tile_size), min(max(overlap, 0.0), 0.95)


def estimated_tiles_per_axis(resolution: int, tile_size: int,
                             overlap: float) -> int:
    """Match diffusers' overlapping tile stride well enough for telemetry."""
    if resolution <= tile_size:
        return 1
    stride = max(1, int(tile_size * (1.0 - overlap)))
    return 1 + (int(resolution) - tile_size + stride - 1) // stride


def untiled_reserve_bytes(resolution: int) -> int:
    pixels = int(resolution) * int(resolution)
    return (VAE_UNTILED_BASE_RESERVE_BYTES
            + pixels * VAE_UNTILED_BYTES_PER_PIXEL
            + VAE_UNTILED_SAFETY_BYTES)


def select_vae_runtime(vae: Any, device: torch.device | None, resolution: int,
                       stage: str, *, force_mode: str | None = None) -> VAERuntimeDecision:
    """Select tiled/untiled mode from one stage's canvas and live VRAM.

    ``force_mode`` exists solely for the benchmark harness; normal callers
    leave it unset and get the runtime policy below.
    """
    if force_mode not in {None, "tiled", "untiled"}:
        raise ValueError("force_mode must be 'tiled', 'untiled', or None")
    reserve = untiled_reserve_bytes(resolution)
    can_tile = callable(getattr(vae, "enable_tiling", None))
    can_untile = callable(getattr(vae, "disable_tiling", None))
    if not (can_tile and can_untile):
        return VAERuntimeDecision(
            stage, int(resolution), "unsupported", "VAE has no reversible tiling API",
            free_vram_bytes(device) if device is not None else None, reserve,
            None, None,
        )

    tile_size, overlap = _tile_properties(vae)
    free = free_vram_bytes(device) if device is not None else None
    tiles = estimated_tiles_per_axis(resolution, tile_size, overlap)
    if force_mode is not None:
        return VAERuntimeDecision(
            stage, int(resolution), force_mode, "benchmark override", free,
            reserve, tile_size, tiles,
        )
    if free is None:
        return VAERuntimeDecision(
            stage, int(resolution), "untiled", "no independent VRAM counter",
            free, reserve, tile_size, tiles,
        )
    if free >= reserve:
        return VAERuntimeDecision(
            stage, int(resolution), "untiled", "available VRAM meets untiled reserve",
            free, reserve, tile_size, tiles,
        )
    return VAERuntimeDecision(
        stage, int(resolution), "tiled", "available VRAM below untiled reserve",
        free, reserve, tile_size, tiles,
    )


def apply_vae_runtime(vae: Any, decision: VAERuntimeDecision) -> None:
    if decision.mode == "tiled":
        vae.enable_tiling()
    elif decision.mode == "untiled":
        vae.disable_tiling()


def is_cuda_oom(error: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", ())
    if oom_type and isinstance(error, oom_type):
        return True
    text = str(error).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text)


def _reset_peak(device: torch.device | None) -> int | None:
    if device is None or device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return int(torch.cuda.memory_allocated(device))


def _peak_bytes(device: torch.device | None, baseline: int | None) -> int | None:
    if device is None or device.type != "cuda" or baseline is None:
        return None
    torch.cuda.synchronize(device)
    return max(0, int(torch.cuda.max_memory_allocated(device)) - baseline)


def run_with_vae_runtime(
    pipeline: Any,
    device: torch.device | None,
    resolution: int,
    stage: str,
    invoke: Callable[[], _Result],
    *,
    force_mode: str | None = None,
    telemetry: list[dict[str, Any]] | None = None,
    log: Callable[[str], None] = lambda _message: None,
) -> _Result:
    """Invoke a body/head diffusion call under the current VAE policy.

    A CUDA OOM from an untiled attempt is retried exactly once with tiling.  A
    tiled call never retries, and non-OOM errors always propagate unchanged.
    """
    vae = getattr(pipeline, "vae", None)
    decision = select_vae_runtime(vae, device, resolution, stage, force_mode=force_mode)
    attempts = 0
    while True:
        apply_vae_runtime(vae, decision)
        log(
            f"VAE {stage} policy: {decision.mode} at {decision.resolution}px "
            f"({decision.reason}; free="
            f"{decision.free_vram_bytes / 2**30:.2f} GiB"
            if decision.free_vram_bytes is not None
            else f"VAE {stage} policy: {decision.mode} at {decision.resolution}px "
                 f"({decision.reason})"
        )
        baseline = _reset_peak(device)
        started = perf_counter()
        try:
            output = invoke()
        except BaseException as error:
            if (attempts == 0 and decision.mode == "untiled"
                    and is_cuda_oom(error)
                    and callable(getattr(vae, "enable_tiling", None))):
                attempts += 1
                empty_cache(device) if device is not None else None
                decision = VAERuntimeDecision(
                    stage, int(resolution), "tiled", "untiled CUDA OOM fallback",
                    free_vram_bytes(device) if device is not None else None,
                    decision.untiled_reserve_bytes, decision.tile_sample_size,
                    decision.estimated_tiles_per_axis,
                )
                log(f"VAE {stage}: untiled CUDA OOM; retrying once with tiling")
                continue
            raise
        record = decision.as_dict()
        record.update({
            "attempt": attempts + 1,
            "runtime_seconds": round(perf_counter() - started, 4),
            "peak_vram_bytes": _peak_bytes(device, baseline),
        })
        if telemetry is not None:
            telemetry.append(record)
        return output
