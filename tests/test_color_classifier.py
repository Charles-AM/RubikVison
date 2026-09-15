from __future__ import annotations

import json

import numpy as np
import pytest

from src.color_classifier import HSVColorClassifier, bgr_to_hsv, draw_color_predictions
from src.sticker_detector import extract_stickers


REFERENCE_COLORS = {
    "white": (230, 235, 240),
    "yellow": (20, 220, 240),
    "red": (20, 45, 220),
    "orange": (20, 125, 235),
    "blue": (220, 55, 25),
    "green": (30, 190, 45),
}


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


def test_complete_calibration_is_saved_and_reloaded(tmp_path) -> None:
    calibration_path = tmp_path / "color_calibration.json"
    classifier = HSVColorClassifier(calibration_path=calibration_path)
    for label, color in REFERENCE_COLORS.items():
        classifier.calibrate(label, color)

    reloaded = HSVColorClassifier(calibration_path=calibration_path)

    assert reloaded.is_calibrated
    assert reloaded.prototypes == REFERENCE_COLORS


def test_repeated_calibration_keeps_multiple_angle_samples(tmp_path) -> None:
    calibration_path = tmp_path / "color_calibration.json"
    classifier = HSVColorClassifier(calibration_path=calibration_path)
    for label, color in REFERENCE_COLORS.items():
        classifier.calibrate(label, color)
    angled_red = (45, 95, 195)
    classifier.calibrate("red", angled_red)

    reloaded = HSVColorClassifier(calibration_path=calibration_path)

    assert reloaded.sample_count("red") == 2
    assert reloaded.classify(angled_red)[0] == "red"
    assert json.loads(calibration_path.read_text())["version"] == 2


def test_version_one_calibration_is_migrated_on_load(tmp_path) -> None:
    calibration_path = tmp_path / "color_calibration.json"
    calibration_path.write_text(json.dumps({name: list(color) for name, color in REFERENCE_COLORS.items()}))

    classifier = HSVColorClassifier(calibration_path=calibration_path)

    assert classifier.is_calibrated
    assert classifier.sample_count("yellow") == 1
    assert classifier.prototypes == REFERENCE_COLORS


@pytest.mark.parametrize(("label", "bgr"), REFERENCE_COLORS.items())
def test_calibrated_classifier_handles_brightness_change(tmp_path, label, bgr) -> None:
    classifier = HSVColorClassifier(calibration_path=tmp_path / "calibration.json")
    for reference_label, color in REFERENCE_COLORS.items():
        classifier.calibrate(reference_label, color)
    dimmed = tuple(round(channel * 0.55) for channel in bgr)

    prediction, confidence = classifier.classify(dimmed)

    assert prediction == label
    assert confidence >= 0.5


def test_invalid_calibration_file_falls_back_to_hsv(tmp_path) -> None:
    calibration_path = tmp_path / "color_calibration.json"
    calibration_path.write_text("not json")

    classifier = HSVColorClassifier(calibration_path=calibration_path)

    assert not classifier.is_calibrated
    assert classifier.classify((0, 0, 255))[0] == "red"
