"""Model resolution and loading, shared by the ComfyUI nodes and the
standalone webui.

Extracted verbatim (modulo parametrizing the model-cache directory and
dropping ComfyUI-only assumptions) from `SeeThrough_LoadLayerDiffModel` and
`SeeThrough_LoadDepthModel` in nodes.py, which now delegate here. Keeping one
copy means a fix (e.g. to the issue #6 text-encoder placeholder guard) only
has to happen once.
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from . import vendor
from .paths import (
    DEFAULT_DEPTH_REPO,
    DEFAULT_LAYERDIFF_REPO,
    default_models_dir,
    resolve_model_path,
    scan_model_dirs,
)

__all__ = [
    "DEFAULT_LAYERDIFF_REPO",
    "DEFAULT_DEPTH_REPO",
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
                          dtype: torch.dtype = torch.bfloat16):
    """Load the LayerDiff SDXL pipeline from a resolved local path. `pretrained`
    should already be the output of `resolve_model_path`."""
    vendor.ensure_seethrough_importable()

    print(f"[SeeThrough] Loading LayerDiff model from: {pretrained}", flush=True)
    trans_vae = vendor.TransparentVAE.from_pretrained(pretrained, subfolder="trans_vae")

    if unet_ckpt:
        print(f"[SeeThrough] Loading custom UNet from: {unet_ckpt}", flush=True)
        unet = vendor.UNetFrameConditionModel.from_pretrained(unet_ckpt)
    else:
        unet = vendor.UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet")

    pipeline = vendor.KDiffusionStableDiffusionXLPipeline.from_pretrained(
        pretrained, trans_vae=trans_vae, unet=unet, scheduler=None)

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

    pipeline.vae.to(dtype=dtype)
    pipeline.trans_vae.to(dtype=dtype)
    pipeline.unet.to(dtype=dtype)
    pipeline.text_encoder.to(dtype=dtype)
    pipeline.text_encoder_2.to(dtype=dtype)

    print("[SeeThrough] LayerDiff model loaded to CPU (will move to GPU on demand)", flush=True)
    return pipeline


def load_depth_model(pretrained: str, dtype: torch.dtype = torch.bfloat16):
    """Load the Marigold depth pipeline from a resolved local path."""
    vendor.ensure_seethrough_importable()

    print(f"[SeeThrough] Loading Marigold depth model from: {pretrained}", flush=True)
    unet = vendor.UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet")
    pipeline = vendor.MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet)

    assert_text_encoder_loaded(pipeline.text_encoder, "text_encoder", pretrained)
    pipeline.to(dtype=dtype)

    print("[SeeThrough] Depth model loaded to CPU (will move to GPU on demand)", flush=True)
    return pipeline
