import json
from pathlib import Path

from seethrough_engine.resolution_regression import compare_resolution_snapshots


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "resolution_regression" / "a002_neutral_768_1024.json"
)


def test_a002_captured_768_good_1024_bad_is_a_formal_regression_fixture():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = compare_resolution_snapshots(
        payload["baseline"], payload["comparison"])
    assert result == {
        "status": "regression",
        "baseline_resolution": 768,
        "comparison_resolution": 1024,
        "regressions": [
            "eyewhite_tag_lost",
            "eye_local_fidelity_regressed",
            "semantic_tag_count_decreased",
            "semantic_warnings_added",
        ],
        "new_semantic_warnings": ["missing_eyewhite"],
        "safe_profile": 768,
    }


def test_global_mae_cannot_override_eye_local_regression():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    baseline, comparison = payload["baseline"], payload["comparison"]
    # The 1024 bad-pixel ratio is globally better, yet both sclerae are gone.
    assert comparison["global_composite_fidelity"]["bad_ratio"] < \
        baseline["global_composite_fidelity"]["bad_ratio"]
    assert compare_resolution_snapshots(baseline, comparison)["status"] == "regression"
