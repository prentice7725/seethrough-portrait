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
        return image_rgba  # already has a usable matte

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
    key_background,
    head_resolution,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Upload an image first.")

    from portrait_core import PortraitConfig
    from seethrough_engine.export import save_portrait_bundle
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
        result = run_portrait_pipeline(
            pipeline,
            image_rgba,
            seed=int(seed),
            resolution=int(resolution),
            head_resolution=None if head_resolution == HEAD_RES_MATCH else int(head_resolution),
            num_inference_steps=int(num_inference_steps),
            enable_head_detail=bool(enable_head_detail),
            auto_fill=bool(auto_fill),
            silhouette_guard=bool(silhouette_guard),
            provided_subject_mask=provided_mask,
            portrait_config=PortraitConfig.load(),
            seed_everything=_seed_everything,
            log=_log,
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
            (str(out_dir / info["path"]), tag)
            for tag, info in sorted(manifest["layers"].items())
        ]
        diagnostics_gallery = [
            (str(out_dir / filename), name)
            for name, filename in manifest["diagnostics"].items()
            if filename.lower().endswith(".png")
        ]

        zip_path = _zip_run(out_dir)

        report = result.report
        reasons = report.get("reasons") or []
        reasons_md = "\n".join(f"- {r}" for r in reasons) or "- (none)"

        progress(1.0, desc="Done")
        return (
            _verdict_badge(report["verdict"]),
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

    with gr.Blocks(title="SeeThrough Portrait") as demo:
        gr.Markdown(
            "# SeeThrough Portrait\n"
            "인물 이미지 한 장을 레이어로 분해하고 Silhouette Guard를 적용해 "
            "PASS / SOFT_PASS / REWORK / FAIL 결과를 확인합니다. ComfyUI는 필요하지 않습니다."
        )
        gr.Markdown(
            "다운로드 파일은 검증과 fidelity repair가 끝난 canonical 레이어를 담은 "
            "**Portrait Bundle v1**입니다. 애니메이션 제작은 별도 `portrait-autorig` "
            "프로젝트에서 이 Bundle을 사용합니다."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="인물 이미지 (투명 배경 PNG 권장)",
                    type="numpy",
                    image_mode="RGBA",
                )
                key_bg_in = gr.Checkbox(
                    label="단색 배경을 투명 처리",
                    value=True,
                    info=(
                        "RGBA를 만들 수 없는 모델의 이미지용 옵션입니다. 단색 배경을 "
                        "찾아 제거하고, 머리카락 가장자리는 반투명도로 추정해 색 테두리가 "
                        "남지 않게 합니다. 업로드 이미지에 실제 알파 마스크가 있으면 적용하지 않습니다."
                    ),
                )
                subject_mask_in = gr.Image(
                    label="피사체 마스크 (선택, 흰색 = 피사체 · 불투명 배경용)",
                    type="numpy",
                    image_mode="L",
                )
                model_in = gr.Dropdown(
                    label="모델",
                    choices=model_choices,
                    # First local checkpoint if there is one (no network on
                    # launch), else the repo id, which downloads on first run.
                    value=model_choices[0],
                )
                with gr.Row():
                    seed_in = gr.Number(label="시드", value=42, precision=0)
                    resolution_in = gr.Slider(
                        label="Resolution", minimum=512, maximum=2048, step=64, value=768
                    )
                head_res_in = gr.Dropdown(
                    label="얼굴 디테일 해상도",
                    choices=[HEAD_RES_MATCH, "640", "768", "1024", "1280"],
                    value="768",
                    info=(
                        "The v3 head pass re-diffuses a crop of the head on its own "
                        "square canvas, and its size is what decides whether the fine "
                        "facial layers resolve: on A-001, 512 returns no eyewhite and "
                        "the layers composite to mae 15.1, while 768 returns it at "
                        "11.9. Set this to 768 with a 512 body to keep the detail "
                        "without paying for a full-resolution body pass. Peak VRAM "
                        "follows the larger of the two. A002 currently regresses "
                        "at a 1024 head pass, so 768 is the validated safe profile; "
                        "local eye fidelity will flag a visible loss."
                    ),
                )
                steps_in = gr.Slider(
                    label="추론 단계", minimum=1, maximum=100, step=1, value=30
                )
                with gr.Row():
                    head_detail_in = gr.Checkbox(label="얼굴 디테일 생성", value=True)
                    guard_in = gr.Checkbox(label="실루엣 보호 (Silhouette Guard)", value=True)
                    autofill_in = gr.Checkbox(label="자동 보완 (최대 5회 실행)", value=False)
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=1):
                verdict_out = gr.HTML(label="판정")
                coverage_out = gr.Markdown(label="커버리지")
                reasons_out = gr.Markdown(label="판정 사유")
                report_zip_out = gr.File(label="Portrait Bundle 다운로드 (.zip)")

        with gr.Row():
            layer_gallery_out = gr.Gallery(label="레이어", columns=6, height=300)
        with gr.Row():
            diagnostics_gallery_out = gr.Gallery(
                label="진단 이미지 (커버리지 / 누락 / 유출 / 복원 / body_remainder)",
                columns=5,
                height=220,
            )

        with gr.Accordion("실행 로그", open=False):
            log_out = gr.Textbox(label="", lines=10, max_lines=30)

        run_btn.click(
            fn=run_a001,
            inputs=[
                image_in, model_in, seed_in, resolution_in, steps_in,
                head_detail_in, guard_in, autofill_in, subject_mask_in,
                key_bg_in, head_res_in,
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
