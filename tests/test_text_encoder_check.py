"""Tests for assert_text_encoder_loaded — regression for issue #6.

See https://github.com/tackcrypto1031/tk_seethrough/issues/6

Diffusers may silently substitute nn.Identity for a missing text_encoder
(e.g. when model_index.json does not list it). The helper must fail fast
on any empty-placeholder module, not just nn.Identity.
"""
import pytest
import torch

from seethrough_engine.model_loading import assert_text_encoder_loaded


class _RealTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


def test_passes_on_module_with_parameters():
    assert_text_encoder_loaded(_RealTextEncoder(), "text_encoder", "/fake/path")


def test_raises_on_identity_placeholder():
    with pytest.raises(RuntimeError) as exc:
        assert_text_encoder_loaded(torch.nn.Identity(), "text_encoder", "/fake/path")
    msg = str(exc.value)
    assert "empty placeholder" in msg
    assert "/fake/path" in msg
    assert "issues/6" in msg


def test_raises_on_any_empty_placeholder():
    class CustomEmpty(torch.nn.Module):
        pass

    with pytest.raises(RuntimeError):
        assert_text_encoder_loaded(CustomEmpty(), "text_encoder_2", "/fake/path")


def test_raises_on_none():
    with pytest.raises(RuntimeError):
        assert_text_encoder_loaded(None, "text_encoder", "/fake/path")


def test_error_includes_component_name():
    with pytest.raises(RuntimeError) as exc:
        assert_text_encoder_loaded(torch.nn.Identity(), "text_encoder_2", "/model")
    assert "text_encoder_2" in str(exc.value)
