"""Measure tiled versus untiled LayerDiff VAE stages on one portrait input.

Example (A002, same seed/steps for each mode):
  python tools/benchmark_vae_tiling.py \
    webui/outputs/20260902_061821_4fdd3ec8.portrait/original.png \
    --output webui/outputs/vae_tiling_ab.json

The normal WebUI does *not* use ``--mode``: it selects the runtime policy.
This tool deliberately forces each mode so its report can compare stage wall
time, CUDA peak, static reconstruction, cross-mode output delta, and A002 eye
semantic/local-fidelity observations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seethrough_engine.device import empty_cache, resolve_device
from seethrough_engine.generation import run_portrait_pipeline
from seethrough_engine.image import composite_fidelity, composite_layers
from seethrough_engine.model_loading import (
    default_models_dir,
    load_layerdiff_model,
    resolve_model_path,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_composite(result) -> np.ndarray:
    layers = dict(result.layer_dict)
    if np.any(result.guard.body_remainder[..., 3] > 10):
        layers["body_remainder"] = result.guard.body_remainder
    return composite_layers(layers, result.fullpage.shape[:2])


def _single_run(pipeline, image: np.ndarray, *, resolution: int, mode: str,
                seed: int, steps: int) -> tuple[dict, np.ndarray]:
    started = perf_counter()
    result = run_portrait_pipeline(
        pipeline, image, seed=seed, resolution=resolution,
        head_resolution=resolution, num_inference_steps=steps,
        enable_head_detail=True, auto_fill=False, vae_mode_override=mode,
        seed_everything=_seed_everything,
        log=lambda message: print(f"[{resolution}/{mode}] {message}", flush=True),
    )
    events = result.report["run"].get("vae_runtime", [])
    local = result.report.get("local_fidelity", {})
    composite = _canonical_composite(result)
    record = {
        "mode": mode,
        "wall_seconds": round(perf_counter() - started, 4),
        "stage_runtime_seconds": round(sum(event["runtime_seconds"] for event in events), 4),
        "peak_vram_bytes": max(
            (event["peak_vram_bytes"] or 0 for event in events), default=0),
        "stage_events": events,
        "source_composite_fidelity": composite_fidelity(
            result.fullpage, composite, result.portrait_mask),
        "eyewhite_present": any(tag in result.layer_dict for tag in (
            "eyewhite", "eyewhitel", "eyewhiter")),
        "local_fidelity": local,
        "semantic_warnings": result.report.get("semantic", {}).get("warnings", []),
    }
    return record, composite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="RGBA source portrait (A002 original.png)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="seethroughv0.0.2_layerdiff3d")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[768, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--modes", choices=("tiled", "untiled"), nargs="+",
                        default=["tiled", "untiled"])
    args = parser.parse_args()

    image = np.array(Image.open(args.input).convert("RGBA"))
    pretrained = resolve_model_path(args.model, default_models_dir())
    pipeline = load_layerdiff_model(pretrained)
    device = resolve_device()
    rows: dict[str, dict[str, dict]] = {}
    composites: dict[tuple[int, str], np.ndarray] = {}
    try:
        for resolution in args.resolutions:
            rows[str(resolution)] = {}
            for mode in args.modes:
                empty_cache(device)
                record, composite = _single_run(
                    pipeline, image, resolution=resolution, mode=mode,
                    seed=args.seed, steps=args.steps,
                )
                rows[str(resolution)][mode] = record
                composites[(resolution, mode)] = composite
    finally:
        empty_cache(device)

    for resolution in args.resolutions:
        tiled = composites.get((resolution, "tiled"))
        untiled = composites.get((resolution, "untiled"))
        if tiled is None or untiled is None:
            continue
        error = np.abs(tiled[..., :3].astype(np.int16)
                       - untiled[..., :3].astype(np.int16))
        rows[str(resolution)]["mode_output_delta"] = {
            "rgb_mae": round(float(error.mean()), 4),
            "pixels_over_30_ratio": round(float((error.sum(axis=2) > 30).mean()), 6),
        }

    payload = {
        "benchmark": "LayerDiff VAE tiled-vs-untiled",
        "input": str(args.input),
        "seed": args.seed,
        "steps": args.steps,
        "device": str(device),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
