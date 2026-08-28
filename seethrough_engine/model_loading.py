"""Model resolution and loading, shared by the ComfyUI nodes and the
standalone webui.

Extracted verbatim (modulo parametrizing the model-cache directory and
dropping ComfyUI-only assumptions) from `SeeThrough_LoadLayerDiffModel` and
`SeeThrough_LoadDepthModel` in nodes.py, which now delegate here. Keeping one
copy means a fix (e.g. to the issue #6 text-encoder placeholder guard) only
has to happen once.
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file

from . import vendor
from .paths import (
    DEFAULT_DEPTH_REPO,
    DEFAULT_LAYERDIFF_REPO,
    LAYERDIFF_MARKER_SUBFOLDER,
    default_models_dir,
    resolve_model_path,
    scan_model_dirs,
)

__all__ = [
    "DEFAULT_LAYERDIFF_REPO",
    "DEFAULT_DEPTH_REPO",
    "LAYERDIFF_MARKER_SUBFOLDER",
    "default_models_dir",
    "scan_model_dirs",
    "resolve_model_path",
    "assert_text_encoder_loaded",
    "load_layerdiff_model",
    "load_depth_model",
]


def assert_text_encoder_loaded(text_encoder, name: str, pretrained: str) -> None:
    """Fail fast when diffusers silently substitutes an empty placeholder
    (e.g. nn.Identity) for a missing text_encoder. See issue #6."""
    if text_encoder is None or next(text_encoder.parameters(), None) is None:
        raise RuntimeError(
            f"{name} failed to load (got empty placeholder with no parameters).\n"
            f"Model path: {pretrained}\n"
            f"Likely causes:\n"
            f"  1. Model checkpoint missing text_encoder/text_encoder_2 subfolder or model_index.json entries.\n"
            f"  2. Incompatible diffusers version silently substituted nn.Identity placeholder.\n"
            f"Fix: re-download the model, or downgrade diffusers to a version compatible with your ComfyUI build.\n"
            f"See: https://github.com/tackcrypto1031/tk_seethrough/issues/6"
        )


def load_layerdiff_model(pretrained: str, vae_ckpt: str = "", unet_ckpt: str = "",
                          dtype: torch.dtype = torch.bfloat16, vae_tiling: bool = True):
    """Load the LayerDiff SDXL pipeline from a resolved local path. `pretrained`
    should already be the output of `resolve_model_path`.

    Every `from_pretrained` below passes `torch_dtype` explicitly. Without it
    diffusers builds the model at the default dtype (fp32) and upcasts the
    checkpoint on the way in, so loading this UNet -- 4.07B parameters, already
    stored as bf16 -- materialized ~16.3 GB of fp32 in system RAM only to be
    cast straight back down. (`torch_dtype` is deprecated in favour of `dtype`
    as of diffusers 1.0.0, but it is the spelling every version this node has
    to run under accepts -- ComfyUI ships its own diffusers, see issue #6.)
    Loading at the target dtype makes the blanket `.to(dtype=...)` calls this
    used to end with redundant, so only the custom-`vae_ckpt` path still
    normalizes.

    `vae_tiling` runs the SDXL VAE's encode/decode in overlapping 512px tiles
    rather than across the whole canvas, which is where its activation peak
    otherwise lands (the VAE is called once to encode `fullpage` and once per
    layer to decode, both at full resolution). It is not bit-identical to the
    untiled path -- tiles are blended over a 25% overlap -- so it is the one
    flag to turn off if a run has to match an untiled baseline exactly.
    """
    vendor.ensure_seethrough_importable()

    print(f"[SeeThrough] Loading LayerDiff model from: {pretrained}", flush=True)
    if os.path.isdir(pretrained) and not os.path.isdir(
            os.path.join(pretrained, LAYERDIFF_MARKER_SUBFOLDER)):
        raise RuntimeError(
            f"{pretrained} is not a LayerDiff checkpoint: no "
            f"'{LAYERDIFF_MARKER_SUBFOLDER}' subfolder.\n"
            f"The Marigold depth model ({DEFAULT_DEPTH_REPO}) lives in the same "
            f"models directory but cannot be used here -- pick the LayerDiff "
            f"model ({DEFAULT_LAYERDIFF_REPO}) instead.\n"
            f"If this should be a LayerDiff checkpoint, the download is likely "
            f"incomplete; delete the folder and let it re-download."
        )
    trans_vae = vendor.TransparentVAE.from_pretrained(pretrained, subfolder="trans_vae", torch_dtype=dtype)

    if unet_ckpt:
        print(f"[SeeThrough] Loading custom UNet from: {unet_ckpt}", flush=True)
        unet = vendor.UNetFrameConditionModel.from_pretrained(unet_ckpt, torch_dtype=dtype)
    else:
        unet = vendor.UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet", torch_dtype=dtype)

    pipeline = vendor.KDiffusionStableDiffusionXLPipeline.from_pretrained(
        pretrained, trans_vae=trans_vae, unet=unet, scheduler=None, torch_dtype=dtype)

    assert_text_encoder_loaded(pipeline.text_encoder, "text_encoder", pretrained)
    assert_text_encoder_loaded(pipeline.text_encoder_2, "text_encoder_2", pretrained)

    if vae_ckpt:
        print(f"[SeeThrough] Loading custom VAE from: {vae_ckpt}", flush=True)
        td_sd, vae_sd = {}, {}
        sd = load_file(vae_ckpt)
        for k, v in sd.items():
            if k.startswith("trans_decoder."):
                td_sd[k[len("trans_decoder."):]] = v
            elif k.startswith("vae."):
                vae_sd[k.replace("vae.", "")] = v
        if vae_sd:
            pipeline.vae.load_state_dict(vae_sd)
        if td_sd:
            pipeline.trans_vae.decoder.load_state_dict(td_sd)
        # `load_state_dict` already copies into the existing bf16 parameters,
        # but normalize in case a ckpt brings buffers those copies miss.
        pipeline.vae.to(dtype=dtype)
        pipeline.trans_vae.to(dtype=dtype)

    if vae_tiling:
        pipeline.vae.enable_tiling()

    print("[SeeThrough] LayerDiff model loaded to CPU (will move to GPU on demand)", flush=True)
    return pipeline


def load_depth_model(pretrained: str, dtype: torch.dtype = torch.bfloat16):
    """Load the Marigold depth pipeline from a resolved local path. `torch_dtype`
    is passed for the same reason as in `load_layerdiff_model`: without it the
    bf16 checkpoint is upcast to fp32 on load only to be cast straight back."""
    vendor.ensure_seethrough_importable()

    print(f"[SeeThrough] Loading Marigold depth model from: {pretrained}", flush=True)
    unet = vendor.UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet", torch_dtype=dtype)
    pipeline = vendor.MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet, torch_dtype=dtype)

    assert_text_encoder_loaded(pipeline.text_encoder, "text_encoder", pretrained)

    print("[SeeThrough] Depth model loaded to CPU (will move to GPU on demand)", flush=True)
    return pipeline
