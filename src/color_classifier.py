"""Baseline HSV classifier for Rubik's Cube sticker colors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from src.sticker_detector import StickerRegion


COLOR_NOTATION = {
    "white": "W",
    "yellow": "Y",
    "red": "R",
    "orange": "O",
    "blue": "B",
    "green": "G",
}

REQUIRED_COLORS = frozenset(COLOR_NOTATION)

LABEL_COLORS = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "blue": (255, 80, 30),
    "green": (0, 200, 0),
}


@dataclass(frozen=True)
class ColorPrediction:
    """Predicted color information for one sticker."""

    sticker_index: int
    label: str
    notation: str
    hsv: tuple[int, int, int]
    confidence: float


def bgr_to_hsv(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert one OpenCV BGR color to an OpenCV HSV tuple."""
    pixel = np.array([[color]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return tuple(int(value) for value in hsv)


class HSVColorClassifier:
    """Classify representative sticker colors using interpretable HSV rules."""

    def __init__(
        self,
        white_saturation_max: int = 65,
        dark_value_min: int = 45,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.white_saturation_max = white_saturation_max
        self.dark_value_min = dark_value_min
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.prototypes = self._load_calibration()

    @property
    def is_calibrated(self) -> bool:
        """Return whether all six color prototypes have been captured."""
        return REQUIRED_COLORS.issubset(self.prototypes)

    def calibrate(self, label: str, color_bgr: tuple[int, int, int]) -> None:
        """Store one center-sticker reference and persist it when configured."""
        if label not in REQUIRED_COLORS:
            raise ValueError(f"Unsupported calibration color: {label}")
        self.prototypes[label] = tuple(int(value) for value in color_bgr)
        if self.calibration_path is not None:
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {name: list(color) for name, color in sorted(self.prototypes.items())}
            self.calibration_path.write_text(json.dumps(payload, indent=2) + "\n")

    def classify(self, color_bgr: tuple[int, int, int]) -> tuple[str, float]:
        """Return a color label and heuristic confidence in the range [0, 1]."""
        if self.is_calibrated:
            return self._classify_with_prototypes(color_bgr)

        hue, saturation, value = bgr_to_hsv(color_bgr)

        if saturation <= self.white_saturation_max and value >= self.dark_value_min:
            confidence = 1.0 - saturation / max(self.white_saturation_max, 1)
            return "white", _clamp_confidence(0.55 + 0.45 * confidence)

        # Warm-color hue boundaries shift substantially with camera white
        # balance. The green/red ratio is more stable: low for red, medium for
        # orange, and high for yellow.
        if hue < 45 or hue >= 160:
            red_channel = max(color_bgr[2], 1)
            green_red_ratio = color_bgr[1] / red_channel
            if green_red_ratio < 0.42:
                label, ratio_center, ratio_width = "red", 0.20, 0.30
            elif green_red_ratio < 0.78:
                label, ratio_center, ratio_width = "orange", 0.58, 0.28
            else:
                label, ratio_center, ratio_width = "yellow", 0.95, 0.30

            ratio_score = max(
                0.0, 1.0 - abs(green_red_ratio - ratio_center) / ratio_width
            )
            saturation_score = saturation / 255.0
            value_score = min(value / max(self.dark_value_min * 2, 1), 1.0)
            confidence = (
                0.55 * ratio_score + 0.25 * saturation_score + 0.20 * value_score
            )
            return label, _clamp_confidence(confidence)

        if hue < 90:
            label, center, half_width = "green", 60, 30
        elif hue < 145:
            label, center, half_width = "blue", 115, 30
        else:
            # Reflections can shift blue toward magenta; cubes have no purple.
            label, center, half_width = "blue", 145, 25

        hue_distance = _circular_hue_distance(hue, center)
        hue_score = max(0.0, 1.0 - hue_distance / half_width)
        saturation_score = saturation / 255.0
        value_score = min(value / max(self.dark_value_min * 2, 1), 1.0)
        confidence = 0.5 * hue_score + 0.3 * saturation_score + 0.2 * value_score
        return label, _clamp_confidence(confidence)

    def _classify_with_prototypes(
        self, color_bgr: tuple[int, int, int]
    ) -> tuple[str, float]:
        """Use illumination-resistant channel proportions after calibration."""
        sample = _chromaticity(color_bgr)
        distances = sorted(
            (
                float(np.linalg.norm(sample - _chromaticity(prototype))),
                label,
            )
            for label, prototype in self.prototypes.items()
            if label in REQUIRED_COLORS
        )
        nearest_distance, label = distances[0]
        second_distance = distances[1][0]
        separation = (second_distance - nearest_distance) / max(second_distance, 1e-6)
        confidence = 0.5 + 0.5 * max(0.0, separation)
        return label, _clamp_confidence(confidence)

    def _load_calibration(self) -> dict[str, tuple[int, int, int]]:
        if self.calibration_path is None or not self.calibration_path.is_file():
            return {}
        try:
            payload = json.loads(self.calibration_path.read_text())
            return {
                label: tuple(int(value) for value in payload[label])
                for label in REQUIRED_COLORS
                if label in payload and len(payload[label]) == 3
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def classify_regions(self, regions: list[StickerRegion]) -> list[ColorPrediction]:
        """Classify nine extracted sticker regions in row-major order."""
        predictions: list[ColorPrediction] = []
        for region in regions:
            label, confidence = self.classify(region.median_bgr)
            predictions.append(
                ColorPrediction(
                    sticker_index=region.index,
                    label=label,
                    notation=COLOR_NOTATION[label],
                    hsv=bgr_to_hsv(region.median_bgr),
                    confidence=confidence,
                )
            )
        return predictions


def _circular_hue_distance(first: int, second: int) -> int:
    difference = abs(first - second)
    return min(difference, 180 - difference)


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _chromaticity(color_bgr: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(color_bgr, dtype=np.float32)
    return values / max(float(values.sum()), 1.0)


def draw_color_predictions(
    normalized_face: np.ndarray,
    regions: list[StickerRegion],
    predictions: list[ColorPrediction],
    show_hsv: bool = False,
) -> np.ndarray:
    """Overlay predicted color notation and confidence on every sticker."""
    if len(regions) != len(predictions):
        raise ValueError("regions and predictions must have the same length.")

    for region, prediction in zip(regions, predictions, strict=True):
        x1, y1, x2, y2 = region.bounds
        if show_hsv:
            hue, saturation, _ = prediction.hsv
            text = f"{prediction.notation} H{hue} S{saturation}"
        else:
            text = f"{prediction.notation} {prediction.confidence:.0%}"
        label_color = LABEL_COLORS[prediction.label]
        cv2.rectangle(normalized_face, (x1 + 4, y2 - 28), (x2 - 4, y2 - 5), (0, 0, 0), -1)
        cv2.putText(
            normalized_face,
            text,
            (x1 + 8, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_color,
            1,
            cv2.LINE_AA,
        )
    return normalized_face
