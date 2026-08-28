"""Model path resolution. No torch/diffusers import, so it can be unit
tested without the heavy inference stack installed (mirrors `portrait_core`'s
"no GPU deps" rule for its own pure logic).
"""

from __future__ import annotations

import os

from . import vendor

DEFAULT_LAYERDIFF_REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
DEFAULT_DEPTH_REPO = "layerdifforg/seethroughv0.0.1_marigold"

# A LayerDiff checkpoint carries a `trans_vae` subfolder; the Marigold depth
# checkpoint, which lives in the same models directory, does not. That is the
# cheapest on-disk way to tell the two apart without importing diffusers.
LAYERDIFF_MARKER_SUBFOLDER = "trans_vae"


def default_models_dir() -> str:
    """Model cache directory used when the caller (e.g. the standalone webui)
    has no ComfyUI `models/SeeThrough` directory to defer to."""
    return os.path.join(str(vendor.REPO_ROOT_DIR), "models", "SeeThrough")


def scan_model_dirs(models_dir: str, require_subfolder: str | None = None) -> list[str]:
    """Names of the checkpoint folders under `models_dir`.

    `require_subfolder` keeps only those that contain it, so a caller that can
    load exactly one kind of checkpoint doesn't offer the other one. Both the
    LayerDiff model and the Marigold depth model land in this same directory,
    and offering Marigold where a LayerDiff pipeline is expected only produces
    a confusing "no file named config.json" deep inside diffusers.
    """
    if not os.path.isdir(models_dir):
        return []
    names = sorted(
        name for name in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, name))
    )
    if require_subfolder:
        names = [
            name for name in names
            if os.path.isdir(os.path.join(models_dir, name, require_subfolder))
        ]
    return names


def resolve_model_path(model_name: str, models_dir: str) -> str:
    """Local folder under `models_dir`, downloading/completing it from the
    HuggingFace repo id first when `model_name` looks like one (`org/repo`),
    else pass `model_name` through (e.g. an existing absolute path or
    something the HF cache already resolves).

    For a repo id, `snapshot_download` is called even when `local` already
    exists: it is a cheap, resumable no-op when every file is already there
    (an ETag check per file, no re-download), but it is what actually
    finishes an interrupted first download -- a plain "does the folder
    exist" check would otherwise treat a partial download as done forever
    and never retry, which is exactly the failure mode this replaced.
    """
    model_basename = model_name.split("/")[-1]
    local = os.path.join(models_dir, model_basename)
    if "/" in model_name:
        try:
            from huggingface_hub import snapshot_download
            print(f"[SeeThrough] Verifying/downloading {model_name} -> {local}", flush=True)
            snapshot_download(repo_id=model_name, local_dir=local)
            return local
        except Exception as e:
            if os.path.isdir(local):
                print(
                    f"[SeeThrough] snapshot_download failed ({e}); "
                    "using existing local copy as-is (may be incomplete)",
                    flush=True,
                )
                return local
            print(f"[SeeThrough] snapshot_download failed ({e}); falling back to HF cache", flush=True)
            return model_name
    if os.path.isdir(local):
        return local
    return model_name
