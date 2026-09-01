"""Resolution-normalized geometry for portrait post-processing.

All tuned pixel measurements in static repair and validation are expressed at
the 768 px reference canvas.  Lengths scale linearly; areas scale quadratically.
"""

from __future__ import annotations

import math

BASE_RESOLUTION = 768


def canvas_scale(shape: tuple[int, ...], base: int = BASE_RESOLUTION) -> float:
    height, width = int(shape[0]), int(shape[1])
    return math.sqrt(max(height * width, 1) / float(base * base))


def scale_length(value_at_base: float, shape: tuple[int, ...], *, minimum: int = 1) -> int:
    return max(minimum, int(round(float(value_at_base) * canvas_scale(shape))))


def scale_area(value_at_base: float, shape: tuple[int, ...], *, minimum: int = 1) -> int:
    scale = canvas_scale(shape)
    return max(minimum, int(round(float(value_at_base) * scale * scale)))


def odd_kernel(value_at_base: float, shape: tuple[int, ...], *, minimum: int = 1) -> int:
    value = scale_length(value_at_base, shape, minimum=minimum)
    return value if value % 2 else value + 1
