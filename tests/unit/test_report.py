import json
import unittest

from portrait_core import PortraitConfig, apply_silhouette_guard, evaluate_portrait_layers, resolve_subject_mask
from portrait_core.report import build_portrait_report
from tests.unit.helpers import portrait_subject, rgba


class ReportTests(unittest.TestCase):
    def test_report_separates_recovery_and_semantic_verdict(self):
        config = PortraitConfig.load()
        subject = portrait_subject()
        original = rgba(subject)
        layers = {"topwear": rgba(subject)}
        mask = resolve_subject_mask(original, config=config)
        guard = apply_silhouette_guard(original, layers, mask, config)
        evaluation = evaluate_portrait_layers(layers, mask, config=config)
        report = build_portrait_report(
            {"filename": "synthetic.png", "width": 32, "height": 32},
            {"seed": 42}, mask, guard, evaluation, config,
        )
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["recovery_verdict"], "PASS")
        self.assertEqual(report["verdict"], "REWORK")
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
