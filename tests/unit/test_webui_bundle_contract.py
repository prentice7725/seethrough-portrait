"""Regression checks for the standalone WebUI's Portrait Bundle handoff."""

from webui.app import _report_verdict_badge


def test_webui_reads_pipeline_verdict_from_the_portrait_report():
    badge = _report_verdict_badge({"verdict": "REWORK"})

    assert "REWORK" in badge
    assert "#ea580c" in badge


def test_webui_tolerates_a_report_without_a_verdict():
    assert "UNKNOWN" in _report_verdict_badge({})
