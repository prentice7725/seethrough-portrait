"""Device selection shared between the ComfyUI nodes and the standalone webui.

`nodes.py` normally asks ComfyUI's `comfy.model_management` for the active
device so it cooperates with ComfyUI's own VRAM scheduling. Outside ComfyUI
there is no such scheduler, so we fall back to plain torch device selection.
"""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Best available compute device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_offload_device() -> torch.device:
    """Where models rest between diffusion calls to save VRAM."""
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
