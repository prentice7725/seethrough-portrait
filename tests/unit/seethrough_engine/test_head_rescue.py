"""No head resolution is "safe" in general -- the v3 head pass can skip a
semantic (most visibly `eyewhite`) on one character and not another, at any
resolution. `_rescue_head_semantic` retries only the head crop, at a higher
resolution, only when the original visibly shows a sclera the composite
lost, and keeps whichever attempt actually reconstructs the eyes best.

`_head_rescue_ladder` and `_better_head_local_fidelity` are pure and tested
directly. The orchestration is tested with `diffuse_head_stage` replaced by a
stand-in (its own geometry is `test_head_resolution.py`'s job) so this file
exercises only what's new: the escalate/compare/stop policy.
"""

from unittest.mock import patch

import numpy as np

from portrait_core import PortraitConfig
from seethrough_engine.generation import (
    deterministic_seed,
    _better_head_local_fidelity,
    _head_rescue_ladder,
    _rescue_head_semantic,
)


def test_deterministic_seed_schedule_is_stable_and_attempt_specific():
    first = deterministic_seed("a002-neutral", 0)
    assert first == deterministic_seed("a002-neutral", 0)
    assert first != deterministic_seed("a002-neutral", 1)
    assert first != deterministic_seed("other-source", 0)
    assert 0 <= first < 2**31 - 1

CANVAS = 32
RING_RGB = (235, 233, 230)   # bright, low-chroma -- a visible sclera
IRIS_RGB = (40, 30, 25)
SKIN_RGB = (60, 45, 40)


def _config(**head_rescue_overrides):
    head_rescue = {
        "enabled": True,
        "ladder": [896, 1024],
        "max_escalations": 2,
        "sclera_warning": "missing_visible_eyewhite",
        **head_rescue_overrides,
    }
    return PortraitConfig(raw={"head_rescue": head_rescue})


def _layer(fill_rects, size=CANVAS):
    img = np.zeros((size, size, 4), np.uint8)
    for (y0, y1, x0, x1), rgb in fill_rects:
        img[y0:y1, x0:x1, :3] = rgb
        img[y0:y1, x0:x1, 3] = 255
    return img


def _scene_with_visible_but_undrawn_sclera(*, with_support: bool = True):
    """Original shows a bright ring around the iris; the generated head has
    no `eyewhite`, so the composite there falls back to skin -- exactly the
    "visible in original, lost in composite" case the rescue should catch.

    `with_support=False` omits `head`/`face` (deterministic derivation's
    required plausibility anchor, see `eyewhite_derivation.py`) while leaving
    `local_fidelity`'s sclera check -- which never looks at those tags --
    fully able to detect the loss. That isolates the ladder path: attempt 0
    (deterministic derivation) is guaranteed to decline, so escalation tests
    exercise `diffuse_head_stage` instead of being resolved for free."""
    c = CANVAS // 2
    original = _layer([((0, CANVAS, 0, CANVAS), SKIN_RGB),
                       ((c - 3, c + 3, c - 5, c + 5), RING_RGB),
                       ((c - 2, c + 2, c - 2, c + 2), IRIS_RGB)])
    irides = _layer([((c - 2, c + 2, c - 2, c + 2), IRIS_RGB)])
    layer_dict = {"irides": irides}
    if with_support:
        layer_dict["head"] = _layer([((c - 10, c + 10, c - 10, c + 10), SKIN_RGB)])
        layer_dict["face"] = _layer([((c - 6, c + 6, c - 8, c + 8), SKIN_RGB)])
    return original, layer_dict


def _eyewhite_head_result(present: bool) -> dict[str, np.ndarray]:
    c = CANVAS // 2
    if not present:
        return {"eyewhite": np.zeros((CANVAS, CANVAS, 4), np.uint8)}
    return {"eyewhite": _layer([((c - 3, c + 3, c - 5, c + 5), RING_RGB)])}


def _rescue_kwargs(layer_dict, config):
    return dict(
        layer_dict=layer_dict, fullpage=None,  # fullpage set per-call below
        pipeline=None, device="cpu", input_img=None, scale=1.0, pad_pos=(0, 0),
        resolution=CANVAS, requested_head_resolution=768,
        head_embeds=None, head_pooled=None, num_inference_steps=1, seed=42,
        vae_mode_override=None, vae_runtime_events=None,
        config=config, log=lambda msg: None,
    )


class _FakeDiffuseHeadStage:
    """Returns canned head layers keyed by the requested `head_resolution`,
    standing in for real (GPU) re-diffusion. `calls` records what was asked."""

    def __init__(self, by_resolution: dict[int, dict[str, np.ndarray]]):
        self.by_resolution = by_resolution
        self.calls: list[int] = []

    def __call__(self, pipeline, device, rng, run_layer_dict, input_img, scale, pad_pos,
                resolution, head_resolution, head_embeds, head_pooled, num_inference_steps,
                **kwargs):
        self.calls.append(head_resolution)
        return self.by_resolution.get(head_resolution, {"eyewhite": np.zeros((CANVAS, CANVAS, 4), np.uint8)})


def test_ladder_only_escalates_to_strictly_larger_profiles_capped_at_max():
    assert _head_rescue_ladder(768, [640, 896, 1024, 1280], max_escalations=2) == [896, 1024]
    assert _head_rescue_ladder(1024, [640, 896, 1024, 1280], max_escalations=2) == [1280]
    assert _head_rescue_ladder(1280, [640, 896, 1024, 1280], max_escalations=2) == []
    assert _head_rescue_ladder(768, [896, 1024], max_escalations=1) == [896]


def test_better_head_local_fidelity_prefers_resolved_over_lower_error():
    resolved = {"warnings": [], "eyes": [{"bad_ratio": 0.5}]}
    unresolved_but_lower_error = {"warnings": ["missing_visible_eyewhite"], "eyes": [{"bad_ratio": 0.01}]}
    assert _better_head_local_fidelity(
        unresolved_but_lower_error, resolved, sclera_warning="missing_visible_eyewhite")
    assert not _better_head_local_fidelity(
        resolved, unresolved_but_lower_error, sclera_warning="missing_visible_eyewhite")


def test_better_head_local_fidelity_breaks_ties_by_eye_bad_ratio():
    worse = {"warnings": [], "eyes": [{"bad_ratio": 0.3}]}
    better = {"warnings": [], "eyes": [{"bad_ratio": 0.1}]}
    assert _better_head_local_fidelity(worse, better, sclera_warning="missing_visible_eyewhite")
    assert not _better_head_local_fidelity(better, worse, sclera_warning="missing_visible_eyewhite")


def test_rescue_is_skipped_when_nothing_is_visibly_missing():
    c = CANVAS // 2
    original = _layer([((0, CANVAS, 0, CANVAS), SKIN_RGB)])
    layer_dict = {
        "head": _layer([((c - 10, c + 10, c - 10, c + 10), SKIN_RGB)]),
        "face": _layer([((c - 6, c + 6, c - 8, c + 8), SKIN_RGB)]),
    }
    fake = _FakeDiffuseHeadStage({})
    with patch("seethrough_engine.generation.diffuse_head_stage", fake):
        outcome = _rescue_head_semantic(**{
            **_rescue_kwargs(layer_dict, _config()), "fullpage": original,
        })
    assert fake.calls == []
    assert outcome.report["resolved"] is True
    assert outcome.report["attempts"] == []
    assert outcome.layers is layer_dict


def test_rescue_resolves_via_deterministic_derivation_without_any_diffusion():
    """Attempt 0: when `head`/`face` support the evidence, the sclera is
    copied straight from the original and the ladder never has to run."""
    original, layer_dict = _scene_with_visible_but_undrawn_sclera(with_support=True)
    fake = _FakeDiffuseHeadStage({1024: _eyewhite_head_result(present=True)})
    with patch("seethrough_engine.generation.diffuse_head_stage", fake):
        outcome = _rescue_head_semantic(**{
            **_rescue_kwargs(layer_dict, _config()), "fullpage": original,
        })

    assert fake.calls == []  # no GPU call was needed
    assert outcome.report["resolved"] is True
    assert outcome.report["derived_eyewhite"]["used"] is True
    assert outcome.report["derived_eyewhite"]["derived_px"] > 0
    assert outcome.report["attempts"] == []
    assert outcome.report["final_head_resolution"] == 768  # body/head resolution never changed
    c = CANVAS // 2
    np.testing.assert_array_equal(
        outcome.layers["eyewhite"][c, c - 3, :3], original[c, c - 3, :3])


def test_rescue_escalates_and_adopts_the_resolution_that_actually_fixes_it():
    original, layer_dict = _scene_with_visible_but_undrawn_sclera(with_support=False)
    fake = _FakeDiffuseHeadStage({
        896: _eyewhite_head_result(present=False),
        1024: _eyewhite_head_result(present=True),
    })
    with patch("seethrough_engine.generation.diffuse_head_stage", fake):
        outcome = _rescue_head_semantic(**{
            **_rescue_kwargs(layer_dict, _config()), "fullpage": original,
        })

    assert outcome.report["derived_eyewhite"]["reason"] == "no_head_or_face_support"
    assert fake.calls == [896, 1024]  # stopped once 1024 resolved it
    assert outcome.report["resolved"] is True
    assert outcome.report["final_head_resolution"] == 1024
    assert [a["head_resolution"] for a in outcome.report["attempts"]] == [896, 1024]
    assert outcome.report["attempts"][0]["missing_visible_eyewhite"] is True
    assert outcome.report["attempts"][1]["missing_visible_eyewhite"] is False
    assert outcome.layers["eyewhite"][..., 3].max() > 0


def test_rescue_gives_up_after_the_ladder_and_reports_unresolved():
    original, layer_dict = _scene_with_visible_but_undrawn_sclera(with_support=False)
    fake = _FakeDiffuseHeadStage({
        896: _eyewhite_head_result(present=False),
        1024: _eyewhite_head_result(present=False),
    })
    with patch("seethrough_engine.generation.diffuse_head_stage", fake):
        outcome = _rescue_head_semantic(**{
            **_rescue_kwargs(layer_dict, _config()), "fullpage": original,
        })

    assert fake.calls == [896, 1024]
    assert outcome.report["resolved"] is False
    assert outcome.report["final_head_resolution"] == 768


def test_rescue_respects_the_disabled_config_flag():
    original, layer_dict = _scene_with_visible_but_undrawn_sclera()
    fake = _FakeDiffuseHeadStage({1024: _eyewhite_head_result(present=True)})
    with patch("seethrough_engine.generation.diffuse_head_stage", fake):
        outcome = _rescue_head_semantic(**{
            **_rescue_kwargs(layer_dict, _config(enabled=False)), "fullpage": original,
        })
    assert fake.calls == []
    assert outcome.report == {"enabled": False}
    assert outcome.layers is layer_dict


def test_rescue_ladder_already_at_the_top_makes_no_attempt():
    original, layer_dict = _scene_with_visible_but_undrawn_sclera(with_support=False)
    fake = _FakeDiffuseHeadStage({})
    kwargs = _rescue_kwargs(layer_dict, _config())
    kwargs["requested_head_resolution"] = 1280  # already above the whole ladder
    kwargs["fullpage"] = original
    with patch("seethrough_engine.generation.diffuse_head_stage", fake):
        outcome = _rescue_head_semantic(**kwargs)
    assert fake.calls == []
    assert outcome.report["resolved"] is False
    assert outcome.report["attempts"] == []
