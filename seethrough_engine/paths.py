"""Model path resolution. No torch/diffusers import, so it can be unit
tested without the heavy inference stack installed (mirrors `portrait_core`'s
"no GPU deps" rule for its own pure logic).
"""

from __future__ import annotations

import os

from . import vendor

DEFAULT_LAYERDIFF_REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
DEFAULT_DEPTH_REPO = "layerdifforg/seethroughv0.0.1_marigold"


def default_models_dir() -> str:
    """Model cache directory used when the caller (e.g. the standalone webui)
    has no ComfyUI `models/SeeThrough` directory to defer to."""
    return os.path.join(str(vendor.REPO_ROOT_DIR), "models", "SeeThrough")


def scan_model_dirs(models_dir: str) -> list[str]:
    if not os.path.isdir(models_dir):
        return []
    return sorted(
        name for name in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, name))
    )


def resolve_model_path(model_name: str, models_dir: str) -> str:
    """Local folder under `models_dir` if present, else download from the
    HuggingFace repo id, else pass `model_name` through (e.g. an existing
    absolute path or something the HF cache already resolves)."""
    model_basename = model_name.split("/")[-1]
    local = os.path.join(models_dir, model_basename)
    if os.path.isdir(local):
        return local
    if "/" in model_name:
        try:
            from huggingface_hub import snapshot_download
            print(f"[SeeThrough] Downloading {model_name} -> {local}", flush=True)
            snapshot_download(repo_id=model_name, local_dir=local)
            return local
        except Exception as e:
            print(f"[SeeThrough] snapshot_download failed ({e}); falling back to HF cache", flush=True)
    return model_name
