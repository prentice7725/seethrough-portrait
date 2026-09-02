"""Layer-generation primitives shared by the ComfyUI nodes and the standalone
webui: the raw diffusion call and the Portrait Mode orchestration loop.

`nodes.py`'s `_run_diffusion` / `_layer_similarity` static methods on
`SeeThrough_GenerateLayers_Custom` delegate to `run_diffusion_stage` /
`layers.layer_similarity` so there is one implementation of the GPU-facing
call. The node keeps its own (unchanged) control flow for the full
auto-fill/portrait orchestration, since that loop is interleaved with
ComfyUI-specific VRAM offload bookkeeping; `run_portrait_pipeline` below is
the equivalent orchestration for callers with no ComfyUI underneath (i.e.
the standalone webui), built from the same shared primitives. Tag lists,
head-region cropping, similarity scoring, and preview compositing live in
`.layers` (no torch import, so they stay unit-testable without the heavy
inference stack).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
import torch

try:
    # ComfyUI loads this package's parent as a relative package, where
    # portrait_core is a sibling -- same reason nodes.py imports it this way.
    from ..portrait_core import (
        PortraitConfig,
        apply_silhouette_guard,
        evaluate_portrait_layers,
        resolve_subject_mask,
        select_best_layer_set,
    )
    from ..portrait_core.report import build_portrait_report
except ImportError:
    # Standalone (webui): portrait_core is a plain top-level package on
    # sys.path, not a relative sibling.
    from portrait_core import (
        PortraitConfig,
        apply_silhouette_guard,
        evaluate_portrait_layers,
        resolve_subject_mask,
        select_best_layer_set,
    )
    from portrait_core.report import build_portrait_report

from . import vendor
from .device import (
    empty_cache,
    free_vram_bytes,
    group_offload,
    is_group_offloaded,
    module_bytes,
    resolve_device,
    resolve_offload_device,
)
from .layers import (
    ALL_TAGS,
    VALID_BODY_PARTS_V2,
    VALID_BODY_PARTS_V3_BODY,
    VALID_BODY_PARTS_V3_HEAD,
    align_subject_mask_to_canvas,
    crop_head,
    layer_similarity,
    make_preview,
)
from .eyewhite_derivation import derive_missing_eyewhite
from .repair import repair_portrait_layers
from .semantic import semantic_warnings
from .image import composite_layers
from .local_fidelity import local_fidelity_report
from .ownership import recover_missing_ownership
from .vae_runtime import run_with_vae_runtime

__all__ = [
    "ALL_TAGS",
    "VALID_BODY_PARTS_V2",
    "VALID_BODY_PARTS_V3_BODY",
    "VALID_BODY_PARTS_V3_HEAD",
    "align_subject_mask_to_canvas",
    "crop_head",
    "layer_similarity",
    "make_preview",
    "fit_unet_on",
    "run_diffusion_stage",
    "run_portrait_pipeline",
    "PortraitPipelineResult",
]

_NOOP_LOG: Callable[[str], None] = lambda msg: None

# Room to leave free for activations after the weights are placed. The diffusion
# loop runs one frame per body-part tag as a batch, so this is generous on
# purpose -- guessing low here trades a working (if slower) run for an OOM.
ACTIVATION_HEADROOM_BYTES = 1_500_000_000


def fit_unet_on(unet, device, offload_device, *,
                headroom_bytes: int = ACTIVATION_HEADROOM_BYTES,
                log: Callable[[str], None] = _NOOP_LOG) -> bool:
    """Put the UNet where the diffusion loop can reach it, and say whether it
    had to be streamed rather than moved.

    Moving it wholesale is much faster, so that stays the default. But this
    UNet is 4.07B parameters -- 7.58 GiB even in bf16 -- which is more than an
    8GB card can hold at all, and `.to(device)` on a card that cannot take it
    does not degrade, it raises OutOfMemoryError partway through the move
    (before any of the run's actual settings -- resolution, steps, head detail
    -- get a chance to matter). So when the weights plus activation headroom
    will not fit, stream them a block at a time instead.

    Callers must not `.to(offload_device)` a UNet this returns True for: the
    weights already live there, and the group offload hooks own placement from
    that point on.
    """
    if is_group_offloaded(unet):
        # A pipeline kept warm across runs is already set up.
        return True

    free = free_vram_bytes(device)
    needed = module_bytes(unet)
    if free is None or needed + headroom_bytes <= free:
        unet.to(device)
        return False

    log(f"UNet weights are {needed / 2**30:.2f} GiB and only {free / 2**30:.2f} GiB is free "
        f"on {device} -- streaming them one block at a time instead")
    group_offload(unet, device, offload_device)
    return True


def diffuse_head_stage(pipeline, device, rng, run_layer_dict, input_img, scale, pad_pos,
                       resolution, head_resolution, head_embeds, head_pooled,
                       num_inference_steps, *, vae_mode_override: str | None = None,
                       vae_runtime_events: list[dict[str, Any]] | None = None,
                       log: Callable[[str], None] = _NOOP_LOG) -> dict[str, np.ndarray]:
    """Re-diffuse only the v3 head crop on its own `head_resolution` square
    canvas, and paste the result back onto the body `resolution` canvas.

    Extracted from `run_diffusion_stage`'s v3 head branch so a caller that
    already has a body pass's `run_layer_dict` (for the `head` layer's bbox)
    can re-run just this stage without repeating the body pass -- e.g. the
    head-only semantic rescue ladder in `run_portrait_pipeline`, which never
    re-diffuses the body just to try the head at a different resolution.
    Returns `{}` if the body pass's `head` layer has no visible pixels to
    crop around (mirrors the original inline guard exactly).
    """
    head_resolution = int(head_resolution)
    head_img = run_layer_dict.get("head")
    if head_img is None:
        return {}
    nz = cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8))
    if nz is None:
        return {}
    hx0, hy0, hw, hh = cv2.boundingRect(nz)
    hx = int(hx0 * scale) - pad_pos[0]
    hy = int(hy0 * scale) - pad_pos[1]
    input_head, (hx1, hy1, hx2, hy2) = crop_head(input_img, [hx, hy, int(hw * scale), int(hh * scale)])
    hx1 = int(hx1 / scale + pad_pos[0] / scale)
    hy1 = int(hy1 / scale + pad_pos[1] / scale)
    ih, iw = input_head.shape[:2]
    input_head, head_pad_size, head_pad_pos = vendor.center_square_pad_resize(
        input_head, head_resolution, return_pad_info=True)

    out = run_with_vae_runtime(
        pipeline, device, head_resolution, "head",
        lambda: pipeline(
            strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
            generator=rng, guidance_scale=1.0,
            prompt_embeds=head_embeds, pooled_prompt_embeds=head_pooled,
            fullpage=input_head, group_index=1,
        ),
        force_mode=vae_mode_override, telemetry=vae_runtime_events, log=log,
    )
    log(f"v3 head diffusion complete ({head_resolution}px head canvas)")

    canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    coords = np.array([head_pad_pos[1], head_pad_pos[1] + ih, head_pad_pos[0], head_pad_pos[0] + iw])
    py1, py2, px1, px2 = (coords / scale).astype(np.int64)
    scale_size = (int(head_pad_size[0] / scale), int(head_pad_size[1] / scale))

    head_layer_dict: dict[str, np.ndarray] = {}
    for rst, tag in zip(out.images, VALID_BODY_PARTS_V3_HEAD):
        rst = vendor.smart_resize(rst, scale_size)[py1:py2, px1:px2]
        full = canvas.copy()
        full[hy1:hy1 + rst.shape[0], hx1:hx1 + rst.shape[1]] = rst
        head_layer_dict[tag] = full
    return head_layer_dict


def run_diffusion_stage(pipeline, device, rng, tag_version, num_inference_steps, fullpage,
                         *, prompt_embeds=None, pooled_prompt_embeds=None,
                         body_embeds=None, body_pooled=None, head_embeds=None, head_pooled=None,
                         enable_head_detail=True, input_img=None, scale=1.0, pad_pos=None,
                         resolution=1280, head_resolution=None,
                         vae_mode_override: str | None = None,
                         vae_runtime_events: list[dict[str, Any]] | None = None,
                         log: Callable[[str], None] = _NOOP_LOG) -> dict[str, np.ndarray]:
    """Run a single diffusion pass (body stage, plus the head stage for v3
    when enabled) and return {tag: RGBA ndarray} in canvas (`fullpage`) space.

    `head_resolution` sizes the v3 head pass independently of the body pass,
    defaulting to `resolution` so the two move together unless asked otherwise.
    The head stage crops the head and re-diffuses it on its own square canvas,
    so the pixels it gets are `head_resolution`, not whatever the head happens
    to occupy in the body canvas -- and that is what decides whether the fine
    facial layers resolve at all. Measured on A-001: at 512 the model returns
    no `eyewhite` layer and the layer stack reproduces the original to
    mae 15.1; at 768 `eyewhite` appears and mae drops to 11.9.

    Splitting them is safe because `center_square_pad_resize` computes
    `pad_size`/`pad_pos` from the source *before* resizing, so they are in
    crop pixels and carry no dependence on the target size; the head output is
    mapped back through `scale`, which belongs to the body pass. Only
    `canvas` below must stay at the body `resolution`.
    """
    head_resolution = int(head_resolution or resolution)
    run_layer_dict: dict[str, np.ndarray] = {}

    if tag_version == "v2":
        out = run_with_vae_runtime(
            pipeline, device, resolution, "body",
            lambda: pipeline(
                strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                generator=rng, guidance_scale=1.0,
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                fullpage=fullpage,
            ),
            force_mode=vae_mode_override, telemetry=vae_runtime_events, log=log,
        )
        log("v2 diffusion complete")
        for rst, tag in zip(out.images, VALID_BODY_PARTS_V2):
            run_layer_dict[tag] = rst

    elif tag_version == "v3":
        out = run_with_vae_runtime(
            pipeline, device, resolution, "body",
            lambda: pipeline(
                strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                generator=rng, guidance_scale=1.0,
                prompt_embeds=body_embeds, pooled_prompt_embeds=body_pooled,
                fullpage=fullpage, group_index=0,
            ),
            force_mode=vae_mode_override, telemetry=vae_runtime_events, log=log,
        )
        log("v3 body diffusion complete")
        for rst, tag in zip(out.images, VALID_BODY_PARTS_V3_BODY):
            run_layer_dict[tag] = rst

        if enable_head_detail:
            run_layer_dict.update(diffuse_head_stage(
                pipeline, device, rng, run_layer_dict, input_img, scale, pad_pos,
                resolution, head_resolution, head_embeds, head_pooled, num_inference_steps,
                vae_mode_override=vae_mode_override, vae_runtime_events=vae_runtime_events, log=log,
            ))

    return run_layer_dict


@dataclass
class PortraitPipelineResult:
    layer_dict: dict[str, np.ndarray]
    raw_layer_dict: dict[str, np.ndarray]
    fullpage: np.ndarray
    input_img: np.ndarray
    resolution: int
    pad_size: tuple
    pad_pos: tuple
    portrait_mask: Any
    guard: Any
    evaluation: Any
    report: dict[str, Any]
    repair_report: dict[str, Any]
    ownership_report: dict[str, Any] = field(default_factory=dict)
    all_runs_layers: list[dict[str, Any]] = field(default_factory=list)
    selection_trace: tuple = ()


# Default ladder for the head-only semantic rescue below. Overridable via
# `portrait_defaults.json`'s `head_rescue.ladder` -- if a future corpus shows
# 896 buys nothing, drop it there without touching code.
HEAD_RESCUE_DEFAULT_LADDER: tuple[int, ...] = (896, 1024)
HEAD_RESCUE_DEFAULT_MAX_ESCALATIONS = 2
HEAD_RESCUE_DEFAULT_SCLERA_WARNING = "missing_visible_eyewhite"


def _head_rescue_ladder(requested_head_resolution: int, ladder: list[int],
                        max_escalations: int) -> list[int]:
    """Profiles from `ladder` strictly larger than what was already tried,
    ascending, capped at `max_escalations`. Pure and GPU-free on purpose --
    this is the one piece of the rescue policy worth unit-testing directly."""
    candidates = sorted(r for r in ladder if r > requested_head_resolution)
    return candidates[:max_escalations]


def _mean_eye_bad_ratio(local_report: dict[str, Any]) -> float:
    eyes = local_report.get("eyes") or []
    if not eyes:
        return 1.0
    return sum(float(eye["bad_ratio"]) for eye in eyes) / len(eyes)


def _better_head_local_fidelity(current: dict[str, Any], candidate: dict[str, Any], *,
                                sclera_warning: str) -> bool:
    """Compare two `local_fidelity_report()` results for the rescue ladder.

    A result where the sclera-loss warning is gone always beats one where it
    persists, regardless of eye MAE; between two results that agree on that,
    the lower mean eye `bad_ratio` wins. Mouth/neckline never enter this
    decision -- the ladder only ever re-diffuses head/face tags, so a
    mouth-only fluctuation between attempts is not this decision's business.
    """
    current_bad = sclera_warning in (current.get("warnings") or [])
    candidate_bad = sclera_warning in (candidate.get("warnings") or [])
    if current_bad != candidate_bad:
        return not candidate_bad
    return _mean_eye_bad_ratio(candidate) < _mean_eye_bad_ratio(current)


@dataclass(frozen=True)
class HeadRescueOutcome:
    layers: dict[str, np.ndarray]
    report: dict[str, Any]


def _rescue_head_semantic(
    layer_dict: dict[str, np.ndarray],
    fullpage: np.ndarray,
    *,
    pipeline, device, input_img: np.ndarray, scale: float, pad_pos,
    resolution: int, requested_head_resolution: int,
    head_embeds, head_pooled, num_inference_steps: int, seed: int,
    vae_mode_override: str | None, vae_runtime_events: list[dict[str, Any]] | None,
    config: PortraitConfig, log: Callable[[str], None],
) -> HeadRescueOutcome:
    """Escalate head-only re-diffusion only when the original visibly shows a
    sclera the model failed to draw, and keep whichever head result actually
    reconstructs the eyes best.

    No head resolution is "safe" in general: measured on real portraits, the
    model can skip `eyewhite` at 768 on one character and draw it fine at the
    same resolution on another, or need 1024 where 768 failed. Resolution
    alone buys nothing; what decides is whether *this* head result, judged
    against the original, still shows the loss `local_fidelity`'s sclera
    check already knows how to see (`_sclera_observation` in
    `local_fidelity.py`: visible in the original around the iris, absent from
    the composite -- a closed or stylised-shut eye never triggers this, so
    those portraits never pay for a retry). Only the head crop re-diffuses;
    the body pass, already ~half the run's total time, never repeats.
    """
    cfg = config.raw.get("head_rescue")
    cfg = cfg if isinstance(cfg, dict) else {}
    if not cfg.get("enabled", True):
        return HeadRescueOutcome(layer_dict, {"enabled": False})

    sclera_warning = str(cfg.get("sclera_warning", HEAD_RESCUE_DEFAULT_SCLERA_WARNING))
    shape = fullpage.shape[:2]
    best_layers = layer_dict
    best_report = local_fidelity_report(fullpage, composite_layers(best_layers, shape), best_layers)
    attempts: list[dict[str, Any]] = []

    derived_info: dict[str, Any] = {"attempted": False, "used": False, "derived_px": 0, "reason": "not_needed"}
    if sclera_warning not in (best_report.get("warnings") or []):
        return HeadRescueOutcome(best_layers, {
            "enabled": True, "requested_head_resolution": requested_head_resolution,
            "derived_eyewhite": derived_info, "attempts": attempts, "resolved": True,
            "final_head_resolution": requested_head_resolution,
        })

    # Attempt 0: cheap (no GPU call) and, when it succeeds, exact rather than
    # another generative guess -- copy the sclera straight from the original
    # wherever the evidence around the existing `irides` layer is decisive.
    # Tried before any resolution escalation, never instead of the fidelity
    # gate below: a derived patch only survives if it actually helps.
    derive_result = derive_missing_eyewhite(best_layers, fullpage)
    derived_info = {
        "attempted": True, "used": False,
        "derived_px": derive_result.derived_px, "reason": derive_result.reason,
    }
    if derive_result.layer is not None:
        candidate_layers = {**best_layers, "eyewhite": derive_result.layer}
        candidate_report = local_fidelity_report(
            fullpage, composite_layers(candidate_layers, shape), candidate_layers)
        if _better_head_local_fidelity(best_report, candidate_report, sclera_warning=sclera_warning):
            best_layers, best_report = candidate_layers, candidate_report
            derived_info["used"] = True
        resolved = sclera_warning not in (best_report.get("warnings") or [])
        log(f"Head rescue: derived eyewhite from ground truth ({derive_result.derived_px}px) -- "
            f"{'resolved' if resolved else 'used but not enough' if derived_info['used'] else 'rejected, no improvement'}")
        if resolved:
            return HeadRescueOutcome(best_layers, {
                "enabled": True, "requested_head_resolution": requested_head_resolution,
                "derived_eyewhite": derived_info, "attempts": attempts, "resolved": True,
                "final_head_resolution": requested_head_resolution,
            })

    ladder = _head_rescue_ladder(
        requested_head_resolution,
        [int(r) for r in cfg.get("ladder", HEAD_RESCUE_DEFAULT_LADDER)],
        int(cfg.get("max_escalations", HEAD_RESCUE_DEFAULT_MAX_ESCALATIONS)),
    )
    final_resolution = requested_head_resolution
    for candidate_resolution in ladder:
        log(f"Head rescue: {sclera_warning} at {final_resolution}px, "
            f"retrying head only at {candidate_resolution}px")
        # Same seed at every rung: resolution is the only variable this
        # ladder is testing, not luck from a fresh draw.
        rng = torch.Generator(device=device).manual_seed(seed)
        head_layers = diffuse_head_stage(
            pipeline, device, rng, layer_dict, input_img, scale, pad_pos,
            resolution, candidate_resolution, head_embeds, head_pooled, num_inference_steps,
            vae_mode_override=vae_mode_override, vae_runtime_events=vae_runtime_events, log=log,
        )
        candidate_layers = {**best_layers, **head_layers}
        candidate_report = local_fidelity_report(
            fullpage, composite_layers(candidate_layers, shape), candidate_layers)
        candidate_bad = sclera_warning in (candidate_report.get("warnings") or [])
        attempts.append({
            "head_resolution": candidate_resolution,
            "missing_visible_eyewhite": candidate_bad,
            "eyes_bad_ratio_mean": round(_mean_eye_bad_ratio(candidate_report), 6),
        })
        if _better_head_local_fidelity(best_report, candidate_report, sclera_warning=sclera_warning):
            best_layers, best_report, final_resolution = candidate_layers, candidate_report, candidate_resolution
        resolved = sclera_warning not in (best_report.get("warnings") or [])
        log(f"Head rescue at {candidate_resolution}px: "
            f"{'resolved' if not candidate_bad else 'still missing'}, "
            f"adopted={final_resolution == candidate_resolution}")
        if resolved:
            break

    return HeadRescueOutcome(best_layers, {
        "enabled": True,
        "requested_head_resolution": requested_head_resolution,
        "derived_eyewhite": derived_info,
        "attempts": attempts,
        "resolved": sclera_warning not in (best_report.get("warnings") or []),
        "final_head_resolution": final_resolution,
    })


def run_portrait_pipeline(
    pipeline,
    input_img_rgba: np.ndarray,
    *,
    seed: int = 42,
    resolution: int = 1280,
    head_resolution: int | None = None,
    num_inference_steps: int = 30,
    enable_head_detail: bool = True,
    auto_fill: bool = False,
    max_runs: int = 5,
    silhouette_guard: bool = True,
    provided_subject_mask: np.ndarray | None = None,
    vae_mode_override: str | None = None,
    portrait_config: PortraitConfig | None = None,
    device: torch.device | None = None,
    offload_device: torch.device | None = None,
    seed_everything: Callable[[int], None] = lambda seed: None,
    log: Callable[[str], None] = _NOOP_LOG,
) -> PortraitPipelineResult:
    """Portrait Mode, end to end, for a caller with no ComfyUI graph
    underneath: load an image, run the diffusion stage(s), apply the
    Silhouette Guard, and (optionally) auto-fill low-coverage runs. Mirrors
    the `portrait_mode=True` branch of `SeeThrough_GenerateLayers_Custom.generate`
    in nodes.py, built from the same `run_diffusion_stage` primitive.

    `input_img_rgba` may carry a real alpha channel (e.g. a transparent-PNG
    portrait cutout); unlike nodes.py's ComfyUI branch it is not discarded,
    so `resolve_subject_mask`'s own informative-alpha detection can pick it
    up directly. `provided_subject_mask`, if given, must be the same H,W as
    `input_img_rgba` (before padding) -- it is aligned to the model's square
    canvas the same way the source image is.
    """
    vendor.ensure_seethrough_importable()
    device = device or resolve_device()
    offload_device = offload_device or resolve_offload_device()
    portrait_config = portrait_config or PortraitConfig.load()

    seed_everything(seed)
    input_img = np.asarray(input_img_rgba)
    if input_img.ndim != 3 or input_img.shape[-1] != 4:
        raise ValueError(f"input_img_rgba must be HxWx4, got {input_img.shape}")

    fullpage, pad_size, pad_pos = vendor.center_square_pad_resize(input_img, resolution, return_pad_info=True)
    scale = pad_size[0] / resolution

    aligned_subject_mask = None
    if provided_subject_mask is not None:
        # `provided_subject_mask` is expected at the same H,W as
        # `input_img_rgba` (before padding) -- align it through the same
        # center-square transform so it lines up with `fullpage`.
        aligned_subject_mask = align_subject_mask_to_canvas(provided_subject_mask, resolution)

    tag_version = pipeline.unet.get_tag_version()

    pipeline.text_encoder.to(device)
    pipeline.text_encoder_2.to(device)

    prompt_embeds, pooled_prompt_embeds = None, None
    body_embeds, body_pooled = None, None
    head_embeds, head_pooled = None, None

    if tag_version == "v2":
        prompt_embeds, pooled_prompt_embeds = pipeline.encode_cropped_prompt_77tokens(VALID_BODY_PARTS_V2)
    elif tag_version == "v3":
        body_embeds, body_pooled = pipeline.encode_cropped_prompt_77tokens(VALID_BODY_PARTS_V3_BODY)
        if enable_head_detail:
            head_embeds, head_pooled = pipeline.encode_cropped_prompt_77tokens(VALID_BODY_PARTS_V3_HEAD)
    else:
        raise ValueError(f"Unknown tag version: {tag_version}")

    pipeline.text_encoder.to(offload_device)
    pipeline.text_encoder_2.to(offload_device)
    # Before measuring what is free for the UNet, hand the encoders' VRAM back
    # to the driver -- until then the caching allocator still counts it as used.
    empty_cache(device)

    # The VAE pair is small (under 0.4 GiB together) and is needed at both ends
    # of every pipeline call, so it goes over outright; the UNet is the one that
    # may not fit.
    pipeline.vae.to(device)
    pipeline.trans_vae.to(device)
    unet_streamed = fit_unet_on(pipeline.unet, device, offload_device, log=log)
    empty_cache(device)

    vae_runtime_events: list[dict[str, Any]] = []

    def _diffuse(run_seed: int) -> dict[str, np.ndarray]:
        rng = torch.Generator(device=device).manual_seed(run_seed)
        return run_diffusion_stage(
            pipeline, device, rng, tag_version, num_inference_steps, fullpage,
            prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
            body_embeds=body_embeds, body_pooled=body_pooled,
            head_embeds=head_embeds, head_pooled=head_pooled,
            enable_head_detail=enable_head_detail, input_img=input_img,
            scale=scale, pad_pos=pad_pos, resolution=resolution,
            head_resolution=head_resolution, log=log,
            vae_mode_override=vae_mode_override,
            vae_runtime_events=vae_runtime_events,
        )

    layer_dict = _diffuse(seed)
    all_runs_layers = [{"run": 1, "seed": seed, "layer_dict": dict(layer_dict)}] if auto_fill else []

    # No head resolution is "safe" on its own -- see `_rescue_head_semantic`.
    # Runs before mask/coverage resolution so everything downstream (guard,
    # repair, auto-fill) sees whichever head result actually reconstructed
    # the eyes best.
    head_rescue_outcome: HeadRescueOutcome | None = None
    if tag_version == "v3" and enable_head_detail:
        head_rescue_outcome = _rescue_head_semantic(
            layer_dict, fullpage,
            pipeline=pipeline, device=device, input_img=input_img, scale=scale, pad_pos=pad_pos,
            resolution=resolution, requested_head_resolution=int(head_resolution or resolution),
            head_embeds=head_embeds, head_pooled=head_pooled,
            num_inference_steps=num_inference_steps, seed=seed,
            vae_mode_override=vae_mode_override, vae_runtime_events=vae_runtime_events,
            config=portrait_config, log=log,
        )
        layer_dict = head_rescue_outcome.layers
        if all_runs_layers:
            all_runs_layers[0]["layer_dict"] = dict(layer_dict)

    # Unlike nodes.py's portrait branch, `fullpage` here can carry a real
    # source alpha channel: ComfyUI's IMAGE type is RGB-only (which is why
    # that branch synthesizes opaque alpha and instead leans on a separately
    # supplied MASK input), but a caller with no ComfyUI graph underneath can
    # hand us a true RGBA image directly. Passing it through as-is lets
    # `resolve_subject_mask` do its own informative-alpha detection, which is
    # what the A-001 protocol's "subject-mask source: source alpha, HIGH
    # confidence" baseline expects for a transparent-background portrait.
    portrait_mask = resolve_subject_mask(
        fullpage,
        provided_mask=aligned_subject_mask,
        generated_layers=layer_dict,
        config=portrait_config,
    )
    portrait_eval = evaluate_portrait_layers(
        layer_dict, portrait_mask, enable_head_detail=enable_head_detail, config=portrait_config,
    )
    initial_guard = apply_silhouette_guard(fullpage, layer_dict, portrait_mask, portrait_config)
    log(
        f"Run 1: coverage={initial_guard.metrics.silhouette_coverage:.4f}, "
        f"remainder={initial_guard.metrics.recovered_ratio:.4f}, "
        f"critical_missing={list(portrait_eval.missing_critical_groups)}"
    )

    selection_trace: tuple = ()
    if auto_fill:
        pass_coverage = float(portrait_config.section("verdict")["pass_pre_coverage_min"])
        needs_improvement = (
            initial_guard.metrics.silhouette_coverage < pass_coverage
            or bool(portrait_eval.missing_critical_groups)
        )
        if needs_improvement:
            raw_runs = [dict(layer_dict)]
            for run_idx in range(2, max_runs + 1):
                run_seed = seed + run_idx - 1
                log(f"Auto-fill run {run_idx}/{max_runs} (seed={run_seed})")
                seed_everything(run_seed)
                run_layer_dict = _diffuse(run_seed)
                raw_runs.append(dict(run_layer_dict))
                all_runs_layers.append({"run": run_idx, "seed": run_seed, "layer_dict": dict(run_layer_dict)})

                selected = select_best_layer_set(
                    raw_runs, fullpage, portrait_mask, config=portrait_config,
                    enable_head_detail=enable_head_detail,
                )
                layer_dict = selected.layers
                selection_trace = selected.trace
                current_eval = evaluate_portrait_layers(
                    layer_dict, portrait_mask, enable_head_detail=enable_head_detail, config=portrait_config,
                )
                current_guard = apply_silhouette_guard(fullpage, layer_dict, portrait_mask, portrait_config)
                log(
                    f"After run {run_idx}: coverage={current_guard.metrics.silhouette_coverage:.4f}, "
                    f"critical_missing={list(current_eval.missing_critical_groups)}"
                )
                if (
                    current_guard.metrics.silhouette_coverage >= pass_coverage
                    and not current_eval.missing_critical_groups
                ):
                    break

    if not unet_streamed:
        pipeline.unet.to(offload_device)
    pipeline.vae.to(offload_device)
    pipeline.trans_vae.to(offload_device)
    empty_cache(device)

    # Preserve the selected model output before the guard or repair changes
    # it.  This is optional forensic output in Portrait Bundle v1 and must
    # never be consumed as canonical layers.
    raw_layer_dict = dict(layer_dict)

    if silhouette_guard:
        pre_recovery_guard = apply_silhouette_guard(
            fullpage, layer_dict, portrait_mask, portrait_config)
        repair_result = repair_portrait_layers(
            pre_recovery_guard.guarded_layers, fullpage)
        ownership_result = recover_missing_ownership(
            repair_result.layers,
            fullpage,
            pre_recovery_guard.subject_mask,
        )
        layer_dict = ownership_result.layers
        final_guard = apply_silhouette_guard(
            fullpage, layer_dict, portrait_mask, portrait_config)
        layer_dict = final_guard.guarded_layers
    else:
        raw_config = PortraitConfig(raw={
            **portrait_config.raw,
            "guard": {**portrait_config.section("guard"), "enabled": False, "clip_layers_to_subject": False},
        })
        final_guard = apply_silhouette_guard(fullpage, layer_dict, portrait_mask, raw_config)
        ownership_result = None
        repair_result = repair_portrait_layers(layer_dict, fullpage)
        layer_dict = repair_result.layers

    final_evaluation = evaluate_portrait_layers(
        layer_dict, portrait_mask, enable_head_detail=enable_head_detail, config=portrait_config,
    )

    report = build_portrait_report(
        source={
            "filename": "", "width": int(fullpage.shape[1]),
            "height": int(fullpage.shape[0]), "tag_version": str(tag_version),
        },
        run={
            "seed": int(seed),
            "resolution": int(resolution),
        "head_resolution": int(head_resolution or resolution),
            "steps": int(num_inference_steps),
            "auto_fill": bool(auto_fill),
            "run_count": int(len(all_runs_layers) if all_runs_layers else 1),
            "enable_head_detail": bool(enable_head_detail),
            "silhouette_guard": bool(silhouette_guard),
        },
        mask=portrait_mask,
        guard=final_guard,
        evaluation=final_evaluation,
        config=portrait_config,
        selection_trace=selection_trace,
    )
    report["semantic"]["warnings"] = semantic_warnings(layer_dict, fullpage)
    report["run"]["vae_runtime"] = vae_runtime_events
    report["run"]["head_rescue"] = (
        head_rescue_outcome.report if head_rescue_outcome is not None else {"enabled": False}
    )
    ownership_report = ownership_result.report if ownership_result is not None else {
        "version": "disabled",
        "initial_missing_px": int(final_guard.metrics.missing_area_px),
        "semantic_recovered_px": 0,
        "recovered_by_tag": {},
        "unresolved_remainder_px": int(final_guard.metrics.missing_area_px),
        "unresolved_remainder_ratio": round(float(final_guard.metrics.missing_ratio), 6),
        "candidates": [],
    }
    report["semantic_ownership"] = ownership_report
    canonical_for_validation = dict(layer_dict)
    if np.any(final_guard.body_remainder[..., 3] > 10):
        canonical_for_validation["body_remainder"] = final_guard.body_remainder
    local_report = local_fidelity_report(
        fullpage,
        composite_layers(canonical_for_validation, fullpage.shape[:2]),
        layer_dict,
    )
    report["local_fidelity"] = local_report
    report["semantic"]["warnings"] = list(dict.fromkeys(
        [*report["semantic"]["warnings"], *local_report["warnings"]]
    ))
    if local_report["status"] == "review" and report["verdict"] in {
        "PASS", "SOFT_PASS", "SOFT_PASS_LOW_CONFIDENCE",
    }:
        report["verdict"] = "REWORK"
        report["reasons"].append("Face-critical local fidelity requires review.")

    return PortraitPipelineResult(
        layer_dict=layer_dict,
        raw_layer_dict=raw_layer_dict,
        fullpage=fullpage,
        input_img=input_img,
        resolution=resolution,
        pad_size=pad_size,
        pad_pos=pad_pos,
        portrait_mask=portrait_mask,
        guard=final_guard,
        evaluation=final_evaluation,
        report=report,
        repair_report=repair_result.report,
        ownership_report=ownership_report,
        all_runs_layers=all_runs_layers,
        selection_trace=selection_trace,
    )
