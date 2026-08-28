"""Pure layer-generation helpers: tag lists, head-region cropping, layer
similarity scoring, and preview compositing. No torch import, so this module
can be unit tested without the heavy inference stack installed (mirrors
`portrait_core`'s "no GPU deps" rule for its own pure logic). The GPU-facing
diffusion call lives in `generation.py`, which imports these.
"""

from __future__ import annotations

import numpy as np

from . import vendor

VALID_BODY_PARTS_V2 = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear",
    "nose", "mouth", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]

VALID_BODY_PARTS_V3_BODY = [
    "front hair", "back hair", "head", "neck", "neckwear",
    "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "tail", "wings", "objects",
]

VALID_BODY_PARTS_V3_HEAD = [
    "headwear", "face", "irides", "eyebrow", "eyewhite",
    "eyelash", "eyewear", "ears", "earwear", "nose", "mouth",
]

ALL_TAGS = list(dict.fromkeys(VALID_BODY_PARTS_V2 + VALID_BODY_PARTS_V3_BODY + VALID_BODY_PARTS_V3_HEAD))


def crop_head(img: np.ndarray, xywh):
    x, y, w, h = xywh
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = x, y, x + w, y + h
    if w < iw // 2:
        px = min(iw - x - w, x, w // 5)
        x1 = min(max(x - px, 0), iw)
        x2 = min(max(x + w + px, 0), iw)
    if h < ih // 2:
        py = min(ih - y - h, y, h // 5)
        y2 = min(max(y + h + py, 0), ih)
        y1 = min(max(y - py, 0), ih)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def layer_similarity(layer_img: np.ndarray, original_img: np.ndarray) -> float:
    """[0,1] similarity between a generated layer and the original image,
    comparing RGB only where the layer has alpha > 10. 1.0 = perfect match."""
    mask = layer_img[..., -1] > 10
    if not np.any(mask):
        return 0.0
    h = min(layer_img.shape[0], original_img.shape[0])
    w = min(layer_img.shape[1], original_img.shape[1])
    mask = mask[:h, :w]
    layer_rgb = layer_img[:h, :w, :3][mask].astype(np.float32)
    orig_rgb = original_img[:h, :w, :3][mask].astype(np.float32)
    if layer_rgb.size == 0:
        return 0.0
    mae = np.mean(np.abs(layer_rgb - orig_rgb)) / 255.0
    return float(1.0 - mae)


def align_subject_mask_to_canvas(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Pad/resize a foreground-positive mask (same H,W as the original,
    pre-padding image) through the same center-square transform used on the
    main image, so per-pixel positions still line up after `fullpage` is
    built. Returns float32 [0,1] HxW."""
    mask_np = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_rgb = np.repeat((mask_np[..., None] * 255.0).astype(np.uint8), 3, axis=-1)
    padded, _, _ = vendor.center_square_pad_resize(mask_rgb, resolution, return_pad_info=True)
    if padded.ndim == 3:
        padded = padded[..., 0]
    return padded.astype(np.float32) / 255.0


def make_preview(layer_dict: dict[str, np.ndarray], resolution: int) -> np.ndarray:
    """RGB float32 [0,1] HxWx3 alpha-blended preview of all layers."""
    drawables = []
    for tag, img in layer_dict.items():
        mask = img[..., -1] > 10
        if np.any(mask):
            drawables.append({"img": img, "xyxy": [0, 0, resolution, resolution]})
    if drawables:
        blended = vendor.img_alpha_blending(drawables, premultiplied=False, final_size=(resolution, resolution))
    else:
        blended = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    return blended[..., :3].astype(np.float32) / 255.0
