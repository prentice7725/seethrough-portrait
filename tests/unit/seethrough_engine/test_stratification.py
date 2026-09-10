import numpy as np
import pytest

from seethrough_engine.stratification import (
    build_stratification,
    stratify_left_right,
)


def _layer():
    image = np.zeros((32, 32, 4), np.uint8)
    image[8:16, 3:9, :3] = (80, 90, 100)
    image[8:16, 3:9, 3] = 255
    image[8:16, 23:29, :3] = (80, 90, 100)
    image[8:16, 23:29, 3] = 255
    return image


def test_left_right_stratification_is_derived_and_does_not_mutate_source():
    source = _layer()
    snapshot = source.copy()

    derived, report = stratify_left_right({"handwear": source})

    assert set(derived["handwear"]) == {"left", "right"}
    assert report["status"] == "computed"
    assert int((derived["handwear"]["left"][..., 3] > 10).sum()) == 48
    assert int((derived["handwear"]["right"][..., 3] > 10).sum()) == 48
    np.testing.assert_array_equal(source, snapshot)


def test_single_component_is_not_forced_into_two_sides():
    source = np.zeros((32, 32, 4), np.uint8)
    source[8:16, 8:24, 3] = 255

    derived, report = stratify_left_right({"ears": source})

    assert derived == {}
    assert report["status"] == "not_computed"


def test_depth_is_optional_and_shape_checked():
    source = _layer()
    result = build_stratification(
        {"handwear": source},
        depth_maps={"handwear": np.full((32, 32), 0.25, np.float32)},
        left_right=True,
    )

    assert result.report["depth"]["status"] == "computed"
    assert result.report["left_right"]["status"] == "computed"
    assert result.depth["handwear"].dtype == np.float32
    assert result.depth["handwear"].max() == pytest.approx(0.25)

    with pytest.raises(ValueError, match="depth map"):
        build_stratification(
            {"handwear": source},
            depth_maps={"handwear": np.zeros((16, 16), np.float32)},
        )

