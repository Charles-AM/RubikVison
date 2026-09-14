from __future__ import annotations

import numpy as np
import pytest

from src.color_classifier import HSVColorClassifier, bgr_to_hsv, draw_color_predictions
from src.sticker_detector import extract_stickers


@pytest.mark.parametrize(
    ("bgr", "expected"),
    [
        ((235, 235, 235), "white"),
        ((0, 255, 255), "yellow"),
        ((0, 0, 255), "red"),
        ((0, 128, 255), "orange"),
        ((255, 0, 0), "blue"),
        ((0, 200, 0), "green"),
    ],
)
def test_classifier_recognizes_reference_colors(
    bgr: tuple[int, int, int], expected: str
) -> None:
    label, confidence = HSVColorClassifier().classify(bgr)

    assert label == expected
    assert 0 <= confidence <= 1


def test_classifier_recognizes_dim_white() -> None:
    label, _ = HSVColorClassifier().classify((100, 105, 110))
    assert label == "white"


@pytest.mark.parametrize(
    ("bgr", "expected"),
    [
        ((20, 55, 190), "red"),
        ((15, 120, 235), "orange"),
        ((20, 215, 235), "yellow"),
        ((10, 28, 95), "red"),
        ((8, 55, 105), "orange"),
        ((10, 90, 105), "yellow"),
    ],
)
def test_classifier_separates_warm_colors_under_varied_brightness(
    bgr: tuple[int, int, int], expected: str
) -> None:
    label, _ = HSVColorClassifier().classify(bgr)
    assert label == expected


def test_bgr_to_hsv_uses_opencv_hue_range() -> None:
    hue, saturation, value = bgr_to_hsv((0, 0, 255))
    assert hue == 0
    assert saturation == 255
    assert value == 255


def test_classify_regions_preserves_sticker_order() -> None:
    face = np.full((300, 300, 3), (0, 255, 255), dtype=np.uint8)
    regions = extract_stickers(face)

    predictions = HSVColorClassifier().classify_regions(regions)

    assert [prediction.sticker_index for prediction in predictions] == list(range(1, 10))
    assert {prediction.label for prediction in predictions} == {"yellow"}


def test_draw_predictions_requires_matching_lengths() -> None:
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    regions = extract_stickers(face)
    with pytest.raises(ValueError, match="same length"):
        draw_color_predictions(face, regions, [])
