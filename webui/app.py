"""Standalone SeeThrough Portrait producer WebUI.

No ComfyUI required. Launches a Gradio app that runs Portrait Mode end to
end on one uploaded image -- diffusion decomposition, canonical repair,
static validation, and diagnostics -- and lets you download a Portrait Bundle.

Usage:
    python -m pip install -r webui/requirements.txt
    python webui/app.py
    # then open http://127.0.0.1:7860

Model loading, diffusion, and export all go through `seethrough_engine`, the
same ComfyUI-independent core `nodes.py` delegates to -- so a result from
this webui and a result from the ComfyUI node graph come from one producer
implementation, not two.
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

HEAD_RES_MATCH = "본문과 동일"

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

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


def _dependency_error_message(error: ModuleNotFoundError) -> str | None:
    """Return an actionable message for a missing runtime dependency."""
    if error.name == "cv2":
        return (
            "OpenCV가 현재 실행 중인 Python 환경에 없습니다.\n"
            f"현재 Python: {sys.executable}\n"
            f"설치 명령: \"{sys.executable}\" -m pip install opencv-python"
        )
    return None


def _get_pipeline(model_name: str):
    from seethrough_engine import model_loading

    if model_name not in _pipeline_cache:
        models_dir = model_loading.default_models_dir()
        pretrained = model_loading.resolve_model_path(model_name, models_dir)
        _pipeline_cache.clear()
        _pipeline_cache[model_name] = model_loading.load_layerdiff_model(pretrained)
    return _pipeline_cache[model_name]


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


def _prepare_subject_image(image_rgba: np.ndarray, has_provided_mask: bool,
                           key_background: bool, log) -> np.ndarray:
    """Make sure the upload carries a real subject matte, keying a flat
    background into alpha when it does not.

    The image models in this workflow cannot emit RGBA, so a portrait arrives
    opaque on a solid background. `resolve_subject_mask` would catch that, but
    only after a full diffusion pass, and the failure is visible in the upload
    alone: an opaque region filling its own bounding box is padding, not a
    matte. Left to run, the Silhouette Guard takes the whole rectangle as the
    subject and recovers the *background* into body_remainder, which surfaces
    as a REWORK verdict blaming layer quality for an input problem.
    """
    from portrait_core import PortraitConfig
    from portrait_core.masks import bbox_fill_ratio
    from seethrough_engine import matting
    from seethrough_engine.scale import scale_length

    alpha_cfg = PortraitConfig.load().section("alpha")
    threshold = float(alpha_cfg["binary_threshold"]) * 255.0
    fill = bbox_fill_ratio(image_rgba[..., 3] > threshold)
    if fill < float(alpha_cfg["informative_bbox_fill_max"]):
        # A usable alpha can still carry the original light background in its
        # soft edge pixels (most visible around dark hair and arm/body gaps).
        # Repair that local RGB fringe without re-keying the whole image.
        repaired, info = matting.repair_existing_alpha_edge(image_rgba)
        if info.get("changed_px", 0):
            log(f"Repaired {info['changed_px']} transparent-edge pixels near "
                f"background rgb{tuple(info['background'])}")
        return repaired

    if has_provided_mask:
        return image_rgba  # the mask supplies what the alpha does not

    if not key_background:
        raise gr.Error(
            f"This image has no subject matte: its opaque area fills {fill:.1%} of its "
            "bounding box, so the alpha channel is just padding. Tick "
            "\"Flat background to alpha\", upload a transparent-background PNG, or "
            "supply a subject mask (white = subject)."
        )

    border = scale_length(matting.BORDER_RING_PX, image_rgba.shape)
    edge_band = scale_length(matting.EDGE_BAND_PX, image_rgba.shape)
    found = matting.detect_flat_background(image_rgba, border=border)
    if not found["flat"]:
        raise gr.Error(
            f"Cannot key this background: {found['reason']}. Flat-background keying "
            "needs one solid colour behind the subject. Use a transparent PNG or "
            "supply a subject mask instead."
        )

    keyed, info = matting.key_flat_background(
        image_rgba, border=border, edge_band=edge_band)
    log(f"Keyed a flat background rgb{tuple(info['color'])}: "
        f"{info['border_share']:.0%} of the border, spread {info['border_std']}, "
        f"{info['foreground_ratio']:.1%} of the canvas kept, "
        f"{info['soft_edge_px']} anti-aliased edge pixels")
    for warning in info["warnings"]:
        log(f"  keying note: {warning}")
    return keyed


def _verdict_badge(verdict: str) -> str:
    color = VERDICT_COLORS.get(verdict, "#6b7280")
    return (
        '<div style="display:inline-block;padding:8px 18px;border-radius:8px;'
        f'background:{color};color:white;font-weight:700;font-size:1.15em;'
        f'letter-spacing:0.02em;">{verdict}</div>'
    )


def _report_verdict_badge(report: dict) -> str:
    """Render the Portrait Mode verdict, which belongs to the diagnostic report.

    A Portrait Bundle manifest only records static bundle validation. The
    pipeline's PASS / REWORK / FAIL verdict remains in portrait_report.json
    (and is already available as ``result.report`` in this call path).
    """
    return _verdict_badge(str(report.get("verdict", "UNKNOWN")))


def _coverage_table(coverage: dict) -> str:
    rows = "\n".join(
        f"| {key} | {value} |" for key, value in coverage.items()
    )
    return "| metric | value |\n| --- | --- |\n" + rows


def _zip_run(out_dir: Path) -> str:
    archive_base = str(out_dir)  # shutil appends .zip
    archive_path = shutil.make_archive(archive_base, "zip", root_dir=out_dir)
    return archive_path


def _profile_settings(profile: str) -> tuple[int, bool]:
    """Map the production-facing profile to deterministic attempt policy."""
    settings = {"NORMAL": (1, False), "QUALITY": (3, True), "HARVEST": (5, True)}
    try:
        return settings[str(profile).upper()]
    except KeyError as error:
        raise gr.Error("Production Profile must be NORMAL, QUALITY, or HARVEST.") from error


def _validation_summary(manifest: dict, report: dict) -> str:
    validation = manifest.get("validation") or {}
    labels = {
        "static_reconstruction": "Static Reconstruction",
        "seams": "Seams",
        "local_fidelity": "Local Fidelity",
    }
    rows = "\n".join(
        f"<tr><td>{labels[key]}</td><td><strong>{str(validation.get(key, 'UNKNOWN')).upper()}</strong></td></tr>"
        for key in labels
    )
    warnings = manifest.get("semantics", {}).get("warnings") or []
    diagnostic = str(report.get("verdict", "UNKNOWN"))
    return (
        '<div style="padding:12px;border:1px solid #d1d5db;border-radius:8px">'
        "<h3 style=\"margin-top:0\">PORTRAIT BUNDLE</h3>"
        f"<table><tbody>{rows}</tbody></table>"
        f"<p><strong>Semantic Warnings</strong> {len(warnings)}"
        f" &nbsp; <strong>Diagnostic Summary</strong> {diagnostic}</p></div>"
    )


def _detail_reports(out_dir: Path, manifest: dict, report: dict) -> str:
    validation = manifest.get("validation") or {}
    warnings = manifest.get("semantics", {}).get("warnings") or []
    generation = manifest.get("generation") or {}
    lines = [
        "### Detail Reports",
        f"- Static Fidelity: **{str(validation.get('static_reconstruction', 'UNKNOWN')).upper()}**",
        f"- Local Fidelity: **{str(validation.get('local_fidelity', 'UNKNOWN')).upper()}**",
        f"- Seams: **{str(validation.get('seams', 'UNKNOWN')).upper()}**",
        f"- Semantic Warnings: `{', '.join(warnings) if warnings else 'none'}`",
        f"- Seed: `{generation.get('seed', 'n/a')}` · mode `{generation.get('seed_mode', 'n/a')}` · attempt `{generation.get('attempt_index', 'n/a')}`",
    ]
    return "\n".join(lines)


def run_portrait(
    image,
    model_name,
    profile,
    subject_mask,
    key_background,
    resolution,
    num_inference_steps,
    head_resolution,
    seed_mode,
    manual_seed,
    enable_head_detail,
    disable_guard,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Upload an image first.")

    try:
        from portrait_core import PortraitConfig
        from seethrough_engine.export import save_portrait_bundle
        from seethrough_engine.generation import run_portrait_pipeline
    except ModuleNotFoundError as error:
        message = _dependency_error_message(error)
        if message is not None:
            raise gr.Error(message) from error
        raise

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
        image_rgba = _ensure_rgba(image)
        provided_mask = None
        if subject_mask is not None:
            mask_arr = np.asarray(subject_mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            provided_mask = mask_arr.astype(np.float32) / (255.0 if mask_arr.max() > 1.0 else 1.0)
        # Before the model load, not after the diffusion pass.
        image_rgba = _prepare_subject_image(
            image_rgba, provided_mask is not None, bool(key_background), _log)

        progress(0.05, desc="Loading model...")
        pipeline = _get_pipeline(model_name)

        head_px = int(resolution) if head_resolution == HEAD_RES_MATCH else int(head_resolution)
        _log(f"Diffusing: body {int(resolution)}px, head {head_px}px")
        progress(0.15, desc="Running diffusion + Silhouette Guard (this is the slow part)...")
        attempts, auto_fill = _profile_settings(profile)
        resolved_seed_mode = str(seed_mode)
        if resolved_seed_mode not in {"deterministic_auto", "regression"}:
            raise gr.Error("Reproducibility must be Production or Regression.")
        result = run_portrait_pipeline(
            pipeline,
            image_rgba,
            seed=int(manual_seed),
            seed_mode=resolved_seed_mode,
            resolution=int(resolution),
            head_resolution=None if head_resolution == HEAD_RES_MATCH else int(head_resolution),
            num_inference_steps=int(num_inference_steps),
            enable_head_detail=bool(enable_head_detail),
            auto_fill=bool(auto_fill),
            max_runs=int(attempts),
            silhouette_guard=not bool(disable_guard),
            provided_subject_mask=provided_mask,
            portrait_config=PortraitConfig.load(),
            seed_everything=_seed_everything,
            log=_log,
            # Keep the 8GB-safe baseline until a fresh-pipeline A/B proves an
            # async offload setting is both faster and memory-stable.
            offload_non_blocking=True,
            offload_record_stream=False,
        )

        progress(0.9, desc="Saving outputs...")
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = OUTPUT_ROOT / f"{run_id}.portrait"
        manifest = save_portrait_bundle(str(out_dir), result, source_filename="upload.png")
        fid_path = out_dir / manifest["diagnostics"]["fidelity"]
        import json
        with open(fid_path, encoding="utf-8") as handle:
            fid = json.load(handle)
        if fid:
            sem = fid.get("semantic_only") or {}
            _log(f"Layer composite vs original: mae {fid['mae']:.2f}, "
                 f"{fid['bad_ratio']:.2%} of subject pixels off by >30 "
                 f"(semantic layers alone: mae {sem.get('mae', float('nan')):.2f}, "
                 f"{sem.get('bad_ratio', 0):.2%})")
            if fid["bad_ratio"] > 0.03:
                _log("WARNING: the rendered stack does not reproduce the original. "
                     "A feature layer is probably missing or miscoloured -- check "
                     "the composite_error diagnostic. Coverage metrics are "
                     "alpha-only and cannot see this.")

        layer_gallery = [
            (str(out_dir / info["path"]), f"CANONICAL · {tag}")
            for tag, info in sorted(manifest["layers"].items())
        ]
        diagnostics_gallery = [
            (str(out_dir / filename), f"DIAGNOSTIC · {name}")
            for name, filename in manifest["diagnostics"].items()
            if filename.lower().endswith(".png")
        ]

        zip_path = _zip_run(out_dir)

        report = result.report
        reasons = report.get("reasons") or []
        reasons_md = "\n".join(f"- {r}" for r in reasons) or "- (none)"

        progress(1.0, desc="Done")
        return (
            _validation_summary(manifest, report),
            _coverage_table(report["coverage"]),
            "### Diagnostic Summary\n" + reasons_md,
            layer_gallery,
            diagnostics_gallery,
            _detail_reports(out_dir, manifest, report),
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

    with gr.Blocks(title="SeeThrough Portrait") as demo:
        gr.Markdown(
            "# SeeThrough Portrait\n"
            "Source Portrait → validated **Portrait Bundle v1**\n\n"
            "이 화면은 정적 semantic portrait asset을 만드는 producer입니다. "
            "Composer가 donor·variant·draw order를 조립하고, 별도 `portrait-autorig`가 "
            "mesh·weight·deformation을 담당합니다."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Source Portrait (투명 배경 PNG 권장)",
                    type="numpy", image_mode="RGBA",
                )
                subject_mask_in = gr.Image(
                    label="Subject Mask (선택 · 흰색 = 피사체)",
                    type="numpy", image_mode="L",
                )
                key_bg_in = gr.Checkbox(
                    label="단색 배경을 투명 처리",
                    value=True,
                    info="불투명한 단색 배경을 피사체 alpha로 바꿉니다. 실제 RGBA나 마스크가 있으면 필요 없습니다.",
                )
                profile_in = gr.Radio(
                    label="Production Profile",
                    choices=["NORMAL", "QUALITY", "HARVEST"],
                    value="NORMAL",
                    info=(
                        "NORMAL=1회 표준 생성 · QUALITY=3회 deterministic 후보 비교 · "
                        "HARVEST=5회 후보 생성(Composer harvest가 아님)"
                    ),
                )
                run_btn = gr.Button("Generate Portrait Bundle", variant="primary")

            with gr.Column(scale=1):
                validation_out = gr.HTML(label="Bundle validation")
                coverage_out = gr.Markdown(label="Coverage")
                report_zip_out = gr.File(label="Portrait Bundle 다운로드 (.zip)")

        with gr.Accordion("Advanced", open=False):
            model_in = gr.Dropdown(
                label="Model checkpoint", choices=model_choices, value=model_choices[0],
            )
            with gr.Row():
                resolution_in = gr.Dropdown(
                    label="해상도",
                    choices=[
                        ("512 · Fast", "512"), ("640 · Faster", "640"),
                        ("768 · Standard", "768"), ("896 · High", "896"),
                        ("1024 · Very High", "1024"),
                    ], value="768",
                )
                head_res_in = gr.Dropdown(
                    label="얼굴 디테일 해상도",
                    choices=[
                        (HEAD_RES_MATCH, HEAD_RES_MATCH), ("640 · Fast", "640"),
                        ("768 · Standard", "768"), ("896 · High", "896"),
                        ("1024 · Very High", "1024"), ("1280", "1280"),
                    ], value=HEAD_RES_MATCH,
                    info="작은 눈·코·입 semantic을 위한 head canvas입니다. 최종 품질은 원본 ROI 검증으로 판정합니다.",
                )
            steps_in = gr.Slider(label="추론 단계", minimum=1, maximum=100, step=1, value=30)

        with gr.Accordion("Reproducibility", open=False):
            seed_mode_in = gr.Radio(
                label="Seed mode",
                choices=[
                    ("Production · deterministic_auto", "deterministic_auto"),
                    ("Regression · seed 42", "regression"),
                ], value="deterministic_auto",
                info="Production은 source identity 기반 seed를 사용합니다. Regression은 고정 seed 42로 비교합니다.",
            )
            manual_seed_in = gr.Number(label="Regression seed", value=42, precision=0)

        with gr.Accordion("Research / Debug", open=False):
            head_detail_in = gr.Checkbox(label="Face detail generation", value=True)
            disable_guard_in = gr.Checkbox(
                label="Disable Silhouette Guard", value=False,
                info="Production에서는 켜 둔 상태를 권장합니다. 연구/디버그 용도로만 끄세요.",
            )

        summary_out = gr.Markdown(label="Diagnostic Summary")
        with gr.Row():
            layer_gallery_out = gr.Gallery(
                label="CANONICAL · production assets", columns=6, height=300,
            )
            diagnostics_gallery_out = gr.Gallery(
                label="DIAGNOSTICS · static evidence", columns=5, height=260,
            )
        detail_out = gr.Markdown(label="Detail Reports")
        with gr.Accordion("실행 로그", open=False):
            log_out = gr.Textbox(label="", lines=10, max_lines=30)

        run_btn.click(
            fn=run_portrait,
            inputs=[
                image_in, model_in, profile_in, subject_mask_in, key_bg_in,
                resolution_in, steps_in, head_res_in, seed_mode_in, manual_seed_in,
                head_detail_in, disable_guard_in,
            ],
            outputs=[
                validation_out, coverage_out, summary_out, layer_gallery_out,
                diagnostics_gallery_out, detail_out, report_zip_out, log_out,
            ],
        )

    return demo


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
