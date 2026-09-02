import numpy as np

from portrait_core import PortraitConfig, resolve_subject_mask
from seethrough_engine.backends import (
    BackendProposal,
    CurrentDeterministicMatteBackend,
    CurrentEdgeAlphaBackend,
    DecompositionBackend,
    EdgeMatteBackend,
    SeeThroughDecompositionBackend,
    SubjectMatteBackend,
)
from seethrough_engine.repair import fit_edge_alpha
from tests.unit.helpers import portrait_subject, rgba

CONFIG = PortraitConfig.load()


def test_backend_proposal_metadata_matches_the_plan_contract():
    proposal = BackendProposal(
        backend="vitmatte", backend_version="0.1", target="front_hair",
        confidence=0.91, value=np.zeros((2, 2)), roi=(0, 0, 2, 2), reason="edge_refinement",
    )
    assert proposal.metadata() == {
        "backend": "vitmatte",
        "backend_version": "0.1",
        "target": "front_hair",
        "confidence": 0.91,
        "roi": [0, 0, 2, 2],
        "reason": "edge_refinement",
        "source": "proposal",
        "accepted": False,
    }


def test_backend_proposal_roi_is_none_when_not_given():
    proposal = BackendProposal(backend="x", backend_version="1", target="t", confidence=0.5, value=None)
    assert proposal.metadata()["roi"] is None


def test_current_matte_backend_satisfies_the_protocol():
    backend = CurrentDeterministicMatteBackend(config=CONFIG)
    assert isinstance(backend, SubjectMatteBackend)


def test_current_matte_backend_wraps_resolve_subject_mask_unchanged():
    image = rgba(portrait_subject())
    direct = resolve_subject_mask(image, config=CONFIG)
    backend = CurrentDeterministicMatteBackend(config=CONFIG)
    proposal = backend.matte(image)

    assert proposal.backend == "current_deterministic"
    assert proposal.target == "subject_mask"
    assert proposal.source == "proposal"
    assert proposal.accepted is False
    assert proposal.value.source == direct.source
    np.testing.assert_array_equal(proposal.value.binary, direct.binary)


def test_current_matte_backend_confidence_reflects_evidence_confidence():
    image = rgba(portrait_subject())
    backend = CurrentDeterministicMatteBackend(config=CONFIG)
    proposal = backend.matte(image)
    assert proposal.confidence == 0.95  # HIGH, source_alpha


def test_see_through_decomposition_backend_satisfies_the_protocol():
    backend = SeeThroughDecompositionBackend(diffuse=lambda: {})
    assert isinstance(backend, DecompositionBackend)


def test_see_through_decomposition_backend_calls_the_bound_diffuse_once():
    calls = []

    def diffuse():
        calls.append(1)
        return {"topwear": rgba(portrait_subject())}

    backend = SeeThroughDecompositionBackend(diffuse=diffuse)
    proposal = backend.decompose(image=None, subject_mask=None)

    assert len(calls) == 1
    assert set(proposal.value) == {"topwear"}
    assert proposal.backend == "see_through"
    assert proposal.confidence == 1.0


def test_current_edge_alpha_backend_satisfies_the_protocol():
    backend = CurrentEdgeAlphaBackend()
    assert isinstance(backend, EdgeMatteBackend)


def test_current_edge_alpha_backend_wraps_fit_edge_alpha_unchanged():
    original = rgba(np.ones((24, 24)), rgb=(200, 200, 200))
    layer_dict = {
        "neck": rgba(np.ones((24, 24)), rgb=(200, 200, 200)),
        "topwear": rgba(portrait_subject(size=24), rgb=(50, 50, 50)),
    }
    expected_layers, expected_moved = fit_edge_alpha(dict(layer_dict), original)

    backend = CurrentEdgeAlphaBackend()
    proposal = backend.refine(original, dict(layer_dict))

    assert proposal.target == "edge_alpha"
    assert proposal.roi is None
    for tag in expected_layers:
        np.testing.assert_array_equal(proposal.value[tag], expected_layers[tag])
    assert str(len(expected_moved)) in proposal.reason
