import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from seethrough_engine.export import save_portrait_bundle
from seethrough_engine.repair import REPAIR_ORDER, repair_portrait_layers


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
        repair_report={"version": "1.0", "order": list(REPAIR_ORDER)},
        report={
            "verdict": "PASS",
            "recovery_verdict": "PASS",
            "reasons": [],
            "source": {"tag_version": "v3"},
            "run": {"silhouette_guard": True},
        },
    )


def test_bundle_publishes_repaired_layers_and_keeps_raw_forensics(tmp_path):
    manifest = save_portrait_bundle(str(tmp_path), result_fixture(), source_filename="input.png")

    assert manifest["format"] == "portrait-bundle"
    assert manifest["version"] == "1.0"
    assert manifest["layer_contract"]["canonical_stage"] == "production_repaired"
    assert manifest["layer_contract"]["fidelity_repair"]["order"] == list(REPAIR_ORDER)
    assert "rig" not in manifest
    assert "spine" not in manifest

    canonical = np.array(Image.open(tmp_path / manifest["layers"]["face"]["path"]))
    raw = np.array(Image.open(tmp_path / manifest["raw_layers"]["face"]))
    assert canonical[8, 8, 0] != raw[8, 8, 0]
    assert (tmp_path / "diagnostics" / "fidelity.json").is_file()
    assert (tmp_path / "diagnostics" / "seams.json").is_file()

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
