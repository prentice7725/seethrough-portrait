import torch
from unittest.mock import patch

from seethrough_engine.generation import _encode_prompt_cached, fit_unet_on


class FakePipeline:
    def __init__(self):
        self.calls = 0

    def encode_cropped_prompt_77tokens(self, prompt):
        self.calls += 1
        value = float(self.calls)
        return torch.full((1, 2, 3), value), torch.full((1, 4), value)


def test_fixed_prompt_embedding_cache_encodes_once_and_stays_on_cpu():
    pipeline = FakePipeline()
    key = ("face", "v3", "torch.bfloat16")
    first = _encode_prompt_cached(pipeline, ["face"], cache_key=key)
    second = _encode_prompt_cached(pipeline, ["face"], cache_key=key)

    assert pipeline.calls == 1
    assert first[2] is False
    assert second[2] is True
    assert first[0].device.type == "cpu"
    assert second[0].device.type == "cpu"
    assert torch.equal(first[0], second[0])


def test_async_record_stream_pair_is_disabled_as_8gb_safety_guard():
    class FakeUNet:
        def to(self, _device):
            raise AssertionError("the guarded path should stream")

    calls = {}

    def fake_group(*args, **kwargs):
        calls.update(kwargs)

    with patch("seethrough_engine.generation.is_group_offloaded", return_value=False), \
         patch("seethrough_engine.generation.free_vram_bytes", return_value=0), \
         patch("seethrough_engine.generation.module_bytes", return_value=10), \
         patch("seethrough_engine.generation.group_offload", side_effect=fake_group):
        assert fit_unet_on(
            FakeUNet(), torch.device("cuda"), torch.device("cpu"),
            offload_non_blocking=True, offload_record_stream=True,
        ) is True

    assert calls["non_blocking"] is False
    assert calls["record_stream"] is False
