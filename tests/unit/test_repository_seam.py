from pathlib import Path


def test_upstream_python_does_not_import_autorig_modules():
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in [root / "nodes.py", *sorted((root / "seethrough_engine").glob("*.py"))]:
        text = path.read_text(encoding="utf-8")
        if "portrait_autorig" in text:
            offenders.append(path.name)
    assert offenders == []


def test_rig_implementation_is_not_present_upstream():
    engine = Path(__file__).resolve().parents[2] / "seethrough_engine"
    assert not (engine / "rig.py").exists()
    assert not (engine / "spine.py").exists()
    assert not (engine / "expression.py").exists()
