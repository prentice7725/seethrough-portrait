import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from portrait_core import PortraitConfig, apply_silhouette_guard, resolve_subject_mask
from seethrough_engine.image import composite_fidelity, composite_layers
from seethrough_engine.repair import repair_portrait_layers


root = Path("webui/outputs/20260901_233148_2abcb0c2.portrait")
original = np.array(Image.open(root / "original.png").convert("RGBA"))
raw = {
    path.stem: np.array(Image.open(path).convert("RGBA"))
    for path in (root / "raw_layers").glob("*.png")
}
config = PortraitConfig()
evidence = resolve_subject_mask(original, generated_layers=raw, config=config)
guarded = apply_silhouette_guard(original, raw, evidence, config).guarded_layers
cv2.setRNGSeed(0)
result = repair_portrait_layers(guarded, original)
subject = original[..., 3] > 10
before = composite_fidelity(
    original, composite_layers(guarded, original.shape[:2]), subject
)
after = composite_fidelity(
    original, composite_layers(result.layers, original.shape[:2]), subject
)
diagnostics = root / "diagnostics"
Image.fromarray(result.layers["topwear"]).save(
    diagnostics / "orphan_cleanup_v12_topwear.png"
)
removed = np.zeros(original.shape[:2], np.uint8)
removed[
    (guarded["topwear"][..., 3] > 10)
    & (result.layers["topwear"][..., 3] <= 10)
] = 255
Image.fromarray(removed).save(diagnostics / "orphan_cleanup_v12_removed_mask.png")
payload = {
    "before": before,
    "after": after,
    "repair": result.report,
    "topwear_changed_px": int(np.any(
        guarded["topwear"] != result.layers["topwear"], axis=2
    ).sum()),
}
(diagnostics / "orphan_cleanup_v12.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
