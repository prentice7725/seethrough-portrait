import numpy as np


def rgba(alpha, rgb=(120, 80, 40)):
    alpha = np.asarray(alpha, dtype=np.float32)
    out = np.empty((*alpha.shape, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return out


def portrait_subject(size=32):
    mask = np.zeros((size, size), dtype=np.float32)
    mask[2:12, 10:22] = 1.0
    mask[10:31, 5:27] = 1.0
    return mask
