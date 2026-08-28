"""Per-layer depth estimation with Marigold, shared by the ComfyUI
`SeeThrough_GenerateDepth` node and the standalone webui's Spine export.

Marigold is run once over a batch, not once per layer. The batch is indexed by
`VALID_BODY_PARTS_V2` because that is the vocabulary the depth model was
trained against, so a v3 run has to be folded into those slots first: the four
v3 eye layers are alpha-blended into the single v2 `eyes` slot and the two hair
layers into `hair`, then the resulting depth map is redistributed back to the
v3 tags it came from. A v3 tag with no v2 slot and no fold-in rule -- `head` is
the only one -- gets no depth, which is why callers must tolerate a depth dict
that does not cover every layer.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from . import vendor
from .device import empty_cache
from .layers import VALID_BODY_PARTS_V2

__all__ = ["COMPOSE_INTO_V2", "estimate_layer_depths"]

_NOOP_LOG: Callable[[str], None] = lambda msg: None

# v3 layers that have to be merged to fill a v2 batch slot, in back-to-front
# order (the un-composing below walks it in reverse).
COMPOSE_INTO_V2 = {
    "eyes": ["eyewhite", "irides", "eyelash", "eyebrow"],
    "hair": ["back hair", "front hair"],
}

_ALPHA_FLOOR = 15


def estimate_layer_depths(marigold, layer_dict: dict[str, np.ndarray], fullpage: np.ndarray,
                          resolution: int, *, device: torch.device, offload_device: torch.device,
                          seed: int = 42,
                          seed_everything: Callable[[int], None] = lambda seed: None,
                          log: Callable[[str], None] = _NOOP_LOG) -> dict[str, np.ndarray]:
    """Return `{tag: float32 depth map in [0, 1]}`, larger meaning further back.

    Moves `marigold` to `device` for the single batched call and back to
    `offload_device` afterwards, so it does not sit in VRAM alongside the
    diffusion pipeline.
    """
    vendor.ensure_seethrough_importable()

    empty_array = np.zeros((resolution, resolution, 4), dtype=np.uint8)

    def _floored(tag: str) -> np.ndarray:
        arr = layer_dict[tag].copy()
        arr[..., -1][arr[..., -1] < _ALPHA_FLOOR] = 0
        return arr

    img_list = [_floored(tag) if tag in layer_dict else empty_array.copy()
                for tag in VALID_BODY_PARTS_V2]

    composed: dict[str, dict] = {}
    for slot, members in COMPOSE_INTO_V2.items():
        present = [t for t in members if t in layer_dict]
        if not present:
            continue
        imlist = [_floored(t) for t in present]
        img_list[VALID_BODY_PARTS_V2.index(slot)] = vendor.img_alpha_blending(
            imlist, premultiplied=False)
        composed[slot] = {"taglist": present, "imlist": imlist}

    blended_alpha = np.zeros((resolution, resolution), dtype=np.float32)
    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255
    blended_alpha = (np.clip(blended_alpha, 0, 1) * 255).astype(np.uint8)

    fullpage_for_depth = fullpage.copy()
    fullpage_for_depth[..., -1] = blended_alpha
    img_list.append(fullpage_for_depth)

    marigold.to(device=device)
    empty_cache(device)
    log(f"Marigold on GPU, estimating depth for {len(img_list)} images")

    seed_everything(seed)
    pipe_out = marigold(color_map=None, show_progress_bar=False, img_list=img_list)
    depth_pred = pipe_out.depth_tensor.to(device="cpu", dtype=torch.float32).numpy()

    marigold.to(device=offload_device)
    empty_cache(device)

    depth_dict: dict[str, np.ndarray] = {}
    for i, slot in enumerate(VALID_BODY_PARTS_V2):
        depth = depth_pred[i]
        if slot not in composed:
            depth_dict[slot] = np.clip(depth, 0, 1).astype(np.float32)
            continue

        # Split one composite depth map back over the layers that produced it.
        # Walking front-to-back, whatever a nearer layer already covered is
        # hidden for the ones behind it, so those pixels get the layer's own
        # visible median rather than the nearer layer's depth.
        covered = np.zeros((resolution, resolution), dtype=bool)
        for tag, im in zip(composed[slot]["taglist"][::-1], composed[slot]["imlist"][::-1]):
            here = im[..., -1] > _ALPHA_FLOOR
            hidden = np.bitwise_and(covered, here)
            local = np.full((resolution, resolution), fill_value=1.0, dtype=np.float32)
            local[here] = depth[here]
            if np.any(hidden):
                visible = np.bitwise_and(here, np.bitwise_not(hidden))
                if np.any(visible):
                    local[hidden] = np.median(depth[visible])
            covered = np.bitwise_or(covered, here)
            depth_dict[tag] = local

    log(f"Depth complete: {len(depth_dict)} maps, Marigold offloaded to CPU")
    return depth_dict
