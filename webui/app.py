"""Standalone single-image Portrait Mode webui (M2).

No ComfyUI required. Launches a Gradio app that runs Portrait Mode end to
end on one uploaded image -- diffusion decomposition, the Silhouette Guard,
and a verdict/diagnostics report -- and lets you download the layers and
report. This is the M2 exit-condition tool: "A-001 can be run and exported
from the UI" (see docs/M2_IMPLEMENTATION_SPEC.md).

Usage:
    pip install -r webui/requirements.txt
    python webui/app.py
    # then open http://127.0.0.1:7860

Model loading, diffusion, and export all go through `seethrough_engine`, the
same ComfyUI-independent core `nodes.py` delegates to -- so a result from
this webui and a result from the ComfyUI node graph, given the same seed and
settings, come from one implementation, not two.
"""

from __future__ import annotations

import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path

import gradio as gr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
# Opened straight from disk rather than served: the preview reads a run folder
# through a directory picker, so it needs no server and works on an unzipped
# run anywhere. See docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md.
PREVIEW_PAGE = Path(__file__).resolve().parent / "rig_preview" / "index.html"

VERDICT_COLORS = {
    "PASS": "#16a34a",
    "SOFT_PASS": "#65a30d",
    "SOFT_PASS_LOW_CONFIDENCE": "#ca8a04",
    "REWORK": "#ea580c",
    "FAIL": "#dc2626",
}

# Kept warm across runs so re-generating doesn't reload the model each time.
# Cleared whenever a different model is selected, since we only keep one
# checkpoint's worth of VRAM/RAM resident at a time.
_pipeline_cache: dict[str, object] = {}
# Marigold, loaded only if someone asks for depth-ordered Spine output. Kept
# separately from `_pipeline_cache` because it is not what the model dropdown
# selects, and because both rest on the CPU between runs anyway.
_depth_cache: dict[str, object] = {}


def _get_pipeline(model_name: str):
    from seethrough_engine import model_loading

    if model_name not in _pipeline_cache:
        models_dir = model_loading.default_models_dir()
        pretrained = model_loading.resolve_model_path(model_name, models_dir)
        _pipeline_cache.clear()
        _pipeline_cache[model_name] = model_loading.load_layerdiff_model(pretrained)
    return _pipeline_cache[model_name]


def _get_depth_pipeline():
    """Marigold, downloading it on first use the same way the layer model is."""
    from seethrough_engine import model_loading
    from seethrough_engine import paths as st_paths

    repo = st_paths.DEFAULT_DEPTH_REPO
    if repo not in _depth_cache:
        models_dir = model_loading.default_models_dir()
        _depth_cache[repo] = model_loading.load_depth_model(
            model_loading.resolve_model_path(repo, models_dir))
    return _depth_cache[repo]


def _seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_rgba(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr.astype(np.uint8), alpha], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr.astype(np.uint8)
    else:
        raise gr.Error(f"Unsupported image shape {arr.shape}; expected HxWx3 or HxWx4.")
    return arr


def _verdict_badge(verdict: str) -> str:
    color = VERDICT_COLORS.get(verdict, "#6b7280")
    return (
        '<div style="display:inline-block;padding:8px 18px;border-radius:8px;'
        f'background:{color};color:white;font-weight:700;font-size:1.15em;'
        f'letter-spacing:0.02em;">{verdict}</div>'
    )


def _coverage_table(coverage: dict) -> str:
    rows = "\n".join(
        f"| {key} | {value} |" for key, value in coverage.items()
    )
    return "| metric | value |\n| --- | --- |\n" + rows


def _zip_run(out_dir: Path) -> str:
    archive_base = str(out_dir)  # shutil appends .zip
    archive_path = shutil.make_archive(archive_base, "zip", root_dir=out_dir)
    return archive_path


def run_a001(
    image,
    model_name,
    seed,
    resolution,
    num_inference_steps,
    enable_head_detail,
    silhouette_guard,
    auto_fill,
    subject_mask,
    export_spine,
    spine_depth_order,
    export_rig,
    rig_hair_gradient,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Upload an image first.")

    from portrait_core import PortraitConfig
    from seethrough_engine.export import save_portrait_run
    from seethrough_engine.generation import run_portrait_pipeline

    log_lines: list[str] = []
    # run_portrait_pipeline only calls log() at a handful of checkpoints
    # (per diffusion stage, per auto-fill run) -- not per denoising step --
    # so nudge the bar forward on each one instead of leaving it pinned at
    # whatever the last progress(...) call set. This is a coarse heartbeat,
    # not a real fraction-complete estimate.
    progress_state = {"value": 0.15}

    def _log(msg: str) -> None:
        log_lines.append(msg)
        print(f"[webui] {msg}", flush=True)
        progress_state["value"] = min(progress_state["value"] + 0.03, 0.85)
        progress(progress_state["value"], desc=msg)

    try:
        progress(0.05, desc="Loading model...")
        pipeline = _get_pipeline(model_name)

        image_rgba = _ensure_rgba(image)
        provided_mask = None
        if subject_mask is not None:
            mask_arr = np.asarray(subject_mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            provided_mask = mask_arr.astype(np.float32) / (255.0 if mask_arr.max() > 1.0 else 1.0)

        progress(0.15, desc="Running diffusion + Silhouette Guard (this is the slow part)...")
        result = run_portrait_pipeline(
            pipeline,
            image_rgba,
            seed=int(seed),
            resolution=int(resolution),
            num_inference_steps=int(num_inference_steps),
            enable_head_detail=bool(enable_head_detail),
            auto_fill=bool(auto_fill),
            silhouette_guard=bool(silhouette_guard),
            provided_subject_mask=provided_mask,
            portrait_config=PortraitConfig.load(),
            seed_everything=_seed_everything,
            log=_log,
        )

        depth_dict = None
        if export_spine and spine_depth_order:
            progress(0.88, desc="Estimating depth for Spine draw order...")
            from seethrough_engine.depth import estimate_layer_depths
            from seethrough_engine.device import resolve_device, resolve_offload_device

            depth_dict = estimate_layer_depths(
                _get_depth_pipeline(), result.layer_dict, result.fullpage, result.resolution,
                device=resolve_device(), offload_device=resolve_offload_device(),
                seed=int(seed), seed_everything=_seed_everything, log=_log,
            )

        progress(0.9, desc="Saving outputs...")
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = OUTPUT_ROOT / run_id
        manifest = save_portrait_run(str(out_dir), "a001", result, source_filename="upload.png",
                                     export_spine=bool(export_spine), depth_dict=depth_dict,
                                     export_rig=bool(export_rig),
                                     rig_gradient_tags=("back hair",) if rig_hair_gradient else ())
        if manifest.get("spine"):
            spine_info = manifest["spine"]
            _log(f"Spine project ({spine_info['order']} order): "
                 f"{len(spine_info['slots'])} slots -> {spine_info['json']}")
        if manifest.get("rig"):
            rig_info = manifest["rig"]
            _log(f"Rig manifest ({rig_info['depth']} depth): {len(rig_info['parts'])} parts, "
                 f"anchors {', '.join(rig_info['anchors'])} -> {rig_info['manifest']}")
            _log(f"Preview it: open {PREVIEW_PAGE} and pick {out_dir}")

        layer_gallery = [
            (str(out_dir / filename), tag) for tag, filename in sorted(manifest["layers"].items())
        ]
        diagnostics_gallery = [
            (str(out_dir / filename), name) for name, filename in manifest["diagnostics"].items()
        ]

        zip_path = _zip_run(out_dir)

        report = result.report
        reasons = report.get("reasons") or []
        reasons_md = "\n".join(f"- {r}" for r in reasons) or "- (none)"

        progress(1.0, desc="Done")
        return (
            _verdict_badge(manifest["verdict"]),
            _coverage_table(report["coverage"]),
            reasons_md,
            layer_gallery,
            diagnostics_gallery,
            zip_path,
            "\n".join(log_lines),
        )
    except gr.Error:
        raise
    except Exception as e:  # noqa: BLE001 -- surface any failure to the UI, don't just 500
        traceback.print_exc()
        raise gr.Error(f"Run failed: {e}") from e


def build_app() -> gr.Blocks:
    # Torch-free on purpose: populating the model dropdown shouldn't require
    # loading the inference stack (see seethrough_engine.paths).
    from seethrough_engine import paths as st_paths

    # Only LayerDiff checkpoints: the Marigold depth model shares this models
    # directory but is fetched on demand by `_get_depth_pipeline`, never picked
    # here -- and picking it used to be the default, since it sorts first.
    model_choices = st_paths.scan_model_dirs(
        st_paths.default_models_dir(),
        require_subfolder=st_paths.LAYERDIFF_MARKER_SUBFOLDER,
    ) + [st_paths.DEFAULT_LAYERDIFF_REPO]

    with gr.Blocks(title="SeeThrough Portrait -- A-001") as demo:
        gr.Markdown(
            "# SeeThrough Portrait Mode\n"
            "Single-image Portrait Mode: decompose an upper-body portrait into "
            "layers, run the Silhouette Guard, and see the PASS / SOFT_PASS / "
            "REWORK / FAIL verdict -- no ComfyUI required."
        )
        gr.Markdown(
            f"**2.5D rig preview:** open `{PREVIEW_PAGE}` in a browser and pick a "
            "run folder from `webui/outputs/` to animate it -- head turn, tilt, "
            "breathing, and blink, with no Spine and no Live2D."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Portrait (transparent-background PNG recommended)",
                    type="numpy",
                    image_mode="RGBA",
                )
                subject_mask_in = gr.Image(
                    label="Subject mask (optional, white = subject) -- for opaque backgrounds",
                    type="numpy",
                    image_mode="L",
                )
                model_in = gr.Dropdown(
                    label="Model",
                    choices=model_choices,
                    # First local checkpoint if there is one (no network on
                    # launch), else the repo id, which downloads on first run.
                    value=model_choices[0],
                )
                with gr.Row():
                    seed_in = gr.Number(label="Seed", value=42, precision=0)
                    resolution_in = gr.Slider(
                        label="Resolution", minimum=512, maximum=2048, step=64, value=1280
                    )
                steps_in = gr.Slider(
                    label="Inference steps", minimum=1, maximum=100, step=1, value=30
                )
                with gr.Row():
                    head_detail_in = gr.Checkbox(label="Enable head detail", value=True)
                    guard_in = gr.Checkbox(label="Silhouette Guard", value=True)
                    autofill_in = gr.Checkbox(label="Auto-fill (up to 5 runs)", value=False)
                spine_in = gr.Checkbox(
                    label="Export Spine project",
                    value=False,
                    info=(
                        "Adds a Spine 2D skeleton (JSON + cropped PNGs) to the zip. "
                        "Draw order is the fixed Portrait Mode tag order unless you "
                        "also tick depth ordering below."
                    ),
                )
                spine_depth_in = gr.Checkbox(
                    label="Spine: depth-based draw order",
                    value=False,
                    info=(
                        "Runs Marigold over the layers to sort them by estimated depth, "
                        "matching the ComfyUI Export Spine node. Downloads a 3 GB model "
                        "on first use and adds a pass to every run. Ignored unless "
                        "Export Spine is on."
                    ),
                )
                rig_in = gr.Checkbox(
                    label="Export 2.5D rig manifest",
                    value=True,
                    info=(
                        "Writes rig_manifest.json plus cropped part PNGs for the "
                        "browser preview: remainder split into head/neck/body, "
                        "eyes split left/right, anchors, and head-follow weights. "
                        "No model and no GPU pass -- see "
                        "docs/PORTRAIT_AUTO_RIG_FEASIBILITY_v0.1.md."
                    ),
                )
                rig_hair_in = gr.Checkbox(
                    label="Rig: soften back hair",
                    value=False,
                    info=(
                        "Gives 'back hair' a top-to-bottom falloff instead of "
                        "letting it follow the head rigidly. Turn on if hair "
                        "reaching past the shoulders tears in the preview."
                    ),
                )
                run_btn = gr.Button("Run A-001", variant="primary")

            with gr.Column(scale=1):
                verdict_out = gr.HTML(label="Verdict")
                coverage_out = gr.Markdown(label="Coverage")
                reasons_out = gr.Markdown(label="Reasons")
                report_zip_out = gr.File(label="Download layers + report (.zip)")

        with gr.Row():
            layer_gallery_out = gr.Gallery(label="Layers", columns=6, height=300)
        with gr.Row():
            diagnostics_gallery_out = gr.Gallery(
                label="Diagnostics (coverage / missing / spill / reconstruction / body_remainder)",
                columns=5,
                height=220,
            )

        with gr.Accordion("Run log", open=False):
            log_out = gr.Textbox(label="", lines=10, max_lines=30)

        run_btn.click(
            fn=run_a001,
            inputs=[
                image_in, model_in, seed_in, resolution_in, steps_in,
                head_detail_in, guard_in, autofill_in, subject_mask_in,
                spine_in, spine_depth_in, rig_in, rig_hair_in,
            ],
            outputs=[
                verdict_out, coverage_out, reasons_out,
                layer_gallery_out, diagnostics_gallery_out, report_zip_out, log_out,
            ],
        )

    return demo


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
