import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from seethrough_engine.export import save_portrait_bundle
from seethrough_engine.repair import REPAIR_ORDER, REPAIR_VERSION, repair_portrait_layers


def rgba(side=32, value=180):
    image = np.zeros((side, side, 4), np.uint8)
    image[4:-4, 4:-4, :3] = value
    image[4:-4, 4:-4, 3] = 255
    return image


def result_fixture():
    canonical = rgba()
    raw = canonical.copy()
    raw[8, 8, 0] = 99
    mask = canonical[..., 3] > 0
    guard = SimpleNamespace(
        body_remainder=np.zeros_like(canonical),
        generated_union_post_guard=mask.astype(np.float32),
        missing_mask=np.zeros(mask.shape, np.float32),
        spill_mask=np.zeros(mask.shape, np.float32),
        reconstruction_rgba=canonical.copy(),
        subject_mask=mask,
    )
    return SimpleNamespace(
        layer_dict={"face": canonical},
        raw_layer_dict={"face": raw},
        fullpage=canonical.copy(),
        guard=guard,
        repair_report={"version": REPAIR_VERSION, "order": list(REPAIR_ORDER)},
        ownership_report={
            "version": "1.0",
            "initial_missing_px": 0,
            "semantic_recovered_px": 0,
            "recovered_by_tag": {},
            "unresolved_remainder_px": 0,
            "unresolved_remainder_ratio": 0.0,
            "candidates": [],
        },
        report={
            "verdict": "PASS",
            "recovery_verdict": "PASS",
            "reasons": [],
            "source": {"tag_version": "v3"},
            "run": {"silhouette_guard": True},
            "local_fidelity": {"version": "1.0", "status": "pass", "warnings": []},
        },
    )


def test_bundle_publishes_repaired_layers_and_keeps_raw_forensics(tmp_path):
    manifest = save_portrait_bundle(str(tmp_path), result_fixture(), source_filename="input.png")

    assert manifest["format"] == "portrait-bundle"
    assert manifest["version"] == "1.0"
    assert manifest["layer_contract"]["canonical_stage"] == "production_repaired"
    assert manifest["layer_contract"]["fidelity_repair"]["version"] == REPAIR_VERSION
    assert manifest["layer_contract"]["fidelity_repair"]["order"] == list(REPAIR_ORDER)
    assert manifest["layer_contract"]["semantic_ownership"]["stage"] == "post_repair_pre_remainder"
    assert manifest["validation"]["local_fidelity"] == "pass"
    assert manifest["semantics"]["warnings"] == []
    assert "rig" not in manifest
    assert "spine" not in manifest

    canonical = np.array(Image.open(tmp_path / manifest["layers"]["face"]["path"]))
    raw = np.array(Image.open(tmp_path / manifest["raw_layers"]["face"]))
    assert canonical[8, 8, 0] != raw[8, 8, 0]
    assert (tmp_path / "diagnostics" / "fidelity.json").is_file()
    assert (tmp_path / "diagnostics" / "seams.json").is_file()
    assert (tmp_path / "diagnostics" / "semantic_ownership.json").is_file()
    assert (tmp_path / "diagnostics" / "local_fidelity.json").is_file()
    assert (tmp_path / "diagnostics" / "neckline_contact.json").is_file()

    with open(tmp_path / "manifest.json", encoding="utf-8") as handle:
        assert json.load(handle) == manifest


def test_bundle_can_omit_raw_layers_without_changing_the_canonical_contract(tmp_path):
    manifest = save_portrait_bundle(
        str(tmp_path), result_fixture(), preserve_raw_layers=False
    )
    assert manifest["raw_layers"] == {}
    assert manifest["layer_contract"]["raw_layers_preserved"] is False
    assert not (tmp_path / "raw_layers").exists()


def test_deep_repair_interface_runs_the_declared_order():
    original = rgba(64, 180)
    result = repair_portrait_layers({"face": original.copy()}, original)
    assert result.report["order"] == list(REPAIR_ORDER)
    assert set(result.layers) == {"face"}


def test_bundle_warns_when_canonical_eyewhite_is_missing_even_if_raw_has_it(tmp_path):
    result = result_fixture()
    original = rgba(64, 180)
    original[24:32, 14:26, :3] = (245, 242, 238)
    original[24:32, 38:50, :3] = (245, 242, 238)
    head = original.copy()
    irides = np.zeros_like(original)
    irides[26:30, 18:22] = (55, 45, 40, 255)
    irides[26:30, 42:46] = (55, 45, 40, 255)
    eyewhite = np.zeros_like(original)
    eyewhite[24:32, 14:26] = original[24:32, 14:26]
    eyewhite[24:32, 38:50] = original[24:32, 38:50]
    mask = original[..., 3] > 0
    result.fullpage = original
    result.layer_dict = {"head": head, "irides": irides}
    result.raw_layer_dict = {"head": head.copy(), "irides": irides.copy(), "eyewhite": eyewhite}
    result.guard.subject_mask = mask
    result.guard.generated_union_post_guard = mask.astype(np.float32)

    manifest = save_portrait_bundle(str(tmp_path), result)

    assert manifest["semantics"]["warnings"] == ["missing_eyewhite"]
    assert "eyewhite" not in manifest["layers"]
    assert manifest["raw_layers"]["eyewhite"] == "raw_layers/eyewhite.png"
    with open(tmp_path / manifest["diagnostics"]["portrait_report"], encoding="utf-8") as handle:
        assert json.load(handle)["semantic"]["warnings"] == ["missing_eyewhite"]
