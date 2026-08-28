"""Bootstrap access to the vendored `see-through` research package and
re-export the pieces the engine needs, without requiring ComfyUI.

This mirrors the sys.path / sys.modules dance at the top of `nodes.py`
exactly, so both the ComfyUI node graph and the standalone webui import the
same on-disk `see-through/common` code the same way. It is written to be
safe to call more than once (idempotent) and safe to call whether or not
ComfyUI has already done its own version of this setup for `nodes.py`.

Exports are split into two lazily-loaded groups so a caller who only needs
`utils.cv` (torch-free: cv2/numpy/PIL/pycocotools only) doesn't pull in the
torch+diffusers-heavy `modules.layerdiffuse`/`modules.marigold` pipeline
classes just by touching this module -- that keeps e.g.
`seethrough_engine.layers` unit-testable without the full inference stack
installed.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent
REPO_ROOT_DIR = _ENGINE_DIR.parent
SEETHROUGH_ROOT_DIR = str(REPO_ROOT_DIR / "see-through")
SEETHROUGH_COMMON_DIR = str(REPO_ROOT_DIR / "see-through" / "common")

_bootstrapped = False

_CV_UTIL_NAMES = frozenset({"center_square_pad_resize", "img_alpha_blending", "smart_resize"})


def ensure_seethrough_importable() -> None:
    """Make `import modules...` / `import utils...` (the vendored see-through
    package's own internal import style) resolve, and mock `pycocotools` if
    it is not installed (only used for mask RLE, not needed here). Does not
    itself import any vendored submodule -- see `_load_cv_utils` /
    `_load_pipelines` below for that."""
    global _bootstrapped
    if _bootstrapped:
        return

    if not os.path.isdir(SEETHROUGH_COMMON_DIR):
        raise RuntimeError(
            f"Vendored see-through package not found at {SEETHROUGH_COMMON_DIR!r}. "
            "This repo expects a see-through/ checkout next to nodes.py."
        )

    try:
        import pycocotools  # noqa: F401
    except ImportError:
        mock_pycocotools = types.ModuleType("pycocotools")
        mock_mask = types.ModuleType("pycocotools.mask")
        mock_pycocotools.mask = mock_mask
        sys.modules["pycocotools"] = mock_pycocotools
        sys.modules["pycocotools.mask"] = mock_mask

    if SEETHROUGH_COMMON_DIR not in sys.path:
        sys.path.insert(0, SEETHROUGH_COMMON_DIR)
    if SEETHROUGH_ROOT_DIR not in sys.path:
        sys.path.insert(1, SEETHROUGH_ROOT_DIR)

    _bootstrapped = True


def _with_module_namespace_guard(import_fn):
    """Run `import_fn()` with any pre-existing `utils`/`modules` sys.modules
    entries (e.g. from another custom node -- these are common top-level
    package names) stashed away and restored afterwards, so the vendored
    code's bare `import utils...` / `import modules...` can't corrupt an
    unrelated package of the same name."""
    conflict_backup = {}
    for prefix in ("utils", "modules"):
        for key in list(sys.modules.keys()):
            if key == prefix or key.startswith(prefix + "."):
                conflict_backup[key] = sys.modules.pop(key)
    try:
        return import_fn()
    finally:
        for key, mod in conflict_backup.items():
            if key not in sys.modules:
                sys.modules[key] = mod


def _load_cv_utils() -> dict:
    ensure_seethrough_importable()

    def _do():
        from utils.cv import center_square_pad_resize, img_alpha_blending, smart_resize
        return {
            "center_square_pad_resize": center_square_pad_resize,
            "img_alpha_blending": img_alpha_blending,
            "smart_resize": smart_resize,
        }

    return _with_module_namespace_guard(_do)


def _load_pipelines() -> dict:
    ensure_seethrough_importable()

    def _do():
        from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
        from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
        from modules.layerdiffuse.vae import TransparentVAE
        from modules.marigold import MarigoldDepthPipeline
        from utils.torchcv import cluster_inpaint_part
        return {
            "KDiffusionStableDiffusionXLPipeline": KDiffusionStableDiffusionXLPipeline,
            "UNetFrameConditionModel": UNetFrameConditionModel,
            "TransparentVAE": TransparentVAE,
            "MarigoldDepthPipeline": MarigoldDepthPipeline,
            "cluster_inpaint_part": cluster_inpaint_part,
        }

    return _with_module_namespace_guard(_do)


def __getattr__(name: str):
    """Lazy import: `modules.layerdiffuse`/`modules.marigold` need
    torch/diffusers installed, which is not true in lightweight test
    environments. Only pay that cost when a caller actually touches one of
    those names; `utils.cv`'s pure helpers load on their own, more cheaply."""
    exports = _load_cv_utils() if name in _CV_UTIL_NAMES else _load_pipelines()
    if name not in exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals().update(exports)
    return exports[name]
