from __future__ import annotations

from app.app import capture_color_reference, parse_terminal_key
from src.color_classifier import HSVColorClassifier
from src.tracker import TemporalColorTracker


def test_parse_terminal_key_accepts_color_command() -> None:
    assert parse_terminal_key(" W\n") == ord("w")


def test_parse_terminal_key_uses_first_character() -> None:
    assert parse_terminal_key("red\n") == ord("r")


def test_parse_terminal_key_ignores_empty_command() -> None:
    assert parse_terminal_key("  \n") is None


def test_capture_color_reference_saves_center_color(tmp_path) -> None:
    classifier = HSVColorClassifier(calibration_path=tmp_path / "calibration.json")
    tracker = TemporalColorTracker()

    count = capture_color_reference(classifier, tracker, "orange", (10, 120, 230))

    assert count == 1
    assert classifier.prototypes["orange"] == (10, 120, 230)
