"""ComfyUI-independent core: model loading, layer generation, and Portrait
Mode orchestration shared by `nodes.py` (the ComfyUI node graph) and
`webui/app.py` (the standalone single-image webui, M2).

Importing this package's *submodules* directly (`seethrough_engine.paths`,
`seethrough_engine.layers`) does not require torch/diffusers/etc. Actually
running generation does, via `seethrough_engine.generation` /
`seethrough_engine.model_loading`. The top-level names below are resolved
lazily for the same reason `portrait_core` stays dependency-light: so tests
can import the pure pieces without the heavy inference stack installed.
"""

_LAZY = {
    "PortraitPipelineResult": ("generation", "PortraitPipelineResult"),
    "run_portrait_pipeline": ("generation", "run_portrait_pipeline"),
    "resolve_device": ("device", "resolve_device"),
    "resolve_offload_device": ("device", "resolve_offload_device"),
    "empty_cache": ("device", "empty_cache"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    try:
        submodule_name, attr_name = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib
    submodule = importlib.import_module(f".{submodule_name}", __name__)
    value = getattr(submodule, attr_name)
    globals()[name] = value
    return value
