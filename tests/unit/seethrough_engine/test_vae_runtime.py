import inspect

from seethrough_engine import model_loading
import seethrough_engine.vae_runtime as vae_runtime
from seethrough_engine.vae_runtime import (
    estimated_tiles_per_axis,
    run_with_vae_runtime,
    select_vae_runtime,
)


class FakeVAE:
    tile_sample_min_size = 512
    tile_overlap_factor = 0.25

    def __init__(self):
        self.calls = []

    def enable_tiling(self):
        self.calls.append("enable")

    def disable_tiling(self):
        self.calls.append("disable")


class FakePipeline:
    def __init__(self):
        self.vae = FakeVAE()


def test_the_known_512px_vae_geometry_is_reported_as_2x2_and_3x3():
    assert estimated_tiles_per_axis(768, 512, 0.25) == 2
    assert estimated_tiles_per_axis(1024, 512, 0.25) == 3


def test_cpu_or_unknown_memory_prefers_untiled_and_can_be_overridden_for_ab():
    vae = FakeVAE()
    decision = select_vae_runtime(vae, None, 768, "body")
    assert decision.mode == "untiled"
    assert decision.estimated_tiles_per_axis == 2
    forced = select_vae_runtime(vae, None, 1024, "head", force_mode="tiled")
    assert forced.mode == "tiled"
    assert forced.estimated_tiles_per_axis == 3


def test_1024_reserve_keeps_measured_1p8_gib_head_stage_untiled(monkeypatch):
    vae = FakeVAE()
    monkeypatch.setattr(vae_runtime, "free_vram_bytes", lambda _device: int(1.8 * 2**30))
    assert select_vae_runtime(vae, object(), 1024, "head").mode == "untiled"
    monkeypatch.setattr(vae_runtime, "free_vram_bytes", lambda _device: int(1.7 * 2**30))
    assert select_vae_runtime(vae, object(), 1024, "head").mode == "tiled"


def test_untiled_cuda_oom_retries_exactly_once_with_tiling():
    pipeline = FakePipeline()
    events = []
    attempts = 0

    def invoke():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    assert run_with_vae_runtime(
        pipeline, None, 768, "body", invoke, telemetry=events) == "ok"
    assert attempts == 2
    assert pipeline.vae.calls == ["disable", "enable"]
    assert events[0]["mode"] == "tiled"
    assert events[0]["reason"] == "untiled CUDA OOM fallback"
    assert events[0]["attempt"] == 2


def test_runtime_timing_is_attached_to_telemetry():
    pipeline = FakePipeline()
    timing = {"input_encode_seconds": 0.1, "unet_denoise_seconds": 0.2}
    events = []
    assert run_with_vae_runtime(
        pipeline, None, 512, "body", lambda: "ok",
        timing=timing, telemetry=events,
    ) == "ok"
    assert events[0]["pipeline_timing"] == timing


def test_load_time_vae_tiling_flag_is_removed():
    assert "vae_tiling" not in inspect.signature(
        model_loading.load_layerdiff_model).parameters
