"""Device selection shared between the ComfyUI nodes and the standalone webui.

`nodes.py` normally asks ComfyUI's `comfy.model_management` for the active
device so it cooperates with ComfyUI's own VRAM scheduling. Outside ComfyUI
there is no such scheduler, so we fall back to plain torch device selection.
"""

from __future__ import annotations

import itertools

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


def module_bytes(module: torch.nn.Module) -> int:
    """How much memory `module`'s weights occupy wherever they currently live."""
    return sum(t.numel() * t.element_size()
               for t in itertools.chain(module.parameters(), module.buffers()))


def free_vram_bytes(device: torch.device) -> int | None:
    """Memory still free on `device`, or None where the question has no answer
    (CPU, and MPS which shares system RAM). Reports what the driver sees, so
    call `empty_cache` first if torch has just released something -- memory the
    caching allocator still holds counts as used here."""
    if device.type != "cuda":
        return None
    return torch.cuda.mem_get_info(device)[0]


def is_group_offloaded(module: torch.nn.Module) -> bool:
    """Whether `module` is already streaming its weights in a block at a time."""
    try:
        from diffusers.hooks.group_offloading import _is_group_offload_enabled
    except ImportError:
        return False
    return _is_group_offload_enabled(module)


def group_offload(module: torch.nn.Module, device: torch.device,
                  offload_device: torch.device, *, offload_type: str = "leaf_level",
                  use_stream: bool | None = None, non_blocking: bool = False) -> None:
    """Leave `module`'s weights on `offload_device` and stream them to `device`
    one block at a time, for a model whose weights do not fit whole.

    Unlike `DiffusionPipeline.enable_model_cpu_offload`, this hooks the blocks
    *inside* the module rather than the module itself, which is what makes it
    usable here: the hooks fire on the block forwards the UNet runs anyway.
    (The pipeline-level API is not an option for this pipeline at all -- its
    `trans_vae` is reached as `trans_vae.decoder(...)`, and `TransparentVAE`
    has no `forward` of its own, so a hook on it would never fire.)

    `diffusers.models.modeling_utils.get_parameter_device` asks the group
    offload hooks for the onload device before falling back to the first
    parameter, so `module.device` keeps reporting `device` even while the
    weights sit on `offload_device` -- which is what lets the vendored
    pipeline's `device = self.unet.device` go on working untouched.

    `offload_type` defaults to "leaf_level" rather than the coarser (and, per
    group's-worth-of-syncs, cheaper) "block_level" because this UNet's blocks
    are wildly uneven -- `up_blocks.0` alone is 3.70 GiB and `down_blocks.2`
    2.37 GiB. Prefetching means an adjacent pair is resident at once, and that
    pair plus the always-resident remainder (`mid_block` is 1.25 GiB on its
    own) comes to more than an 8GB card has. Leaf level keeps the resident set
    small enough to be insensitive to that shape.
    """
    from diffusers.hooks import apply_group_offloading

    if use_stream is None:
        # Prefetching the next group while the current one computes is what
        # keeps the transfers from simply adding to the wall time -- every step
        # has to pull the whole 7.58 GiB across PCIe either way. It costs having
        # two groups resident at once, which is affordable at leaf level but not
        # at block level here (see above).
        use_stream = device.type == "cuda" and offload_type == "leaf_level"

    apply_group_offloading(
        module,
        onload_device=device,
        offload_device=offload_device,
        offload_type=offload_type,
        num_blocks_per_group=1 if offload_type == "block_level" else None,
        use_stream=use_stream,
        # Only meaningful with a stream: that path stages through pinned host
        # memory, which is what makes an async copy safe. Without it the source
        # is pageable and the copy is synchronous regardless.
        non_blocking=non_blocking and use_stream,
    )
