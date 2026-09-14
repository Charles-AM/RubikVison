from __future__ import annotations

import pytest

from src.color_classifier import COLOR_NOTATION, ColorPrediction
from src.tracker import TemporalColorTracker


def prediction(index: int, label: str, confidence: float = 0.9) -> ColorPrediction:
    return ColorPrediction(
        sticker_index=index,
        label=label,
        notation=COLOR_NOTATION[label],
        hsv=(0, 200, 200),
        confidence=confidence,
    )


def face(label: str, confidence: float = 0.9) -> list[ColorPrediction]:
    return [prediction(index, label, confidence) for index in range(1, 10)]


def test_single_frame_outlier_does_not_change_stable_color() -> None:
    tracker = TemporalColorTracker(window_size=5)
    tracker.update(face("red"))
    tracker.update(face("red"))

    stable = tracker.update(
        [prediction(index, "orange" if index == 1 else "red") for index in range(1, 10)]
    )

    assert stable[0].label == "red"


def test_majority_vote_eventually_changes_color() -> None:
    tracker = TemporalColorTracker(window_size=3, center_switch_frames=3)
    tracker.update(face("red"))
    tracker.update(face("red"))
    tracker.update([prediction(1, "orange"), *face("red")[1:]])
    tracker.update([prediction(1, "orange"), *face("red")[1:]])
    stable = tracker.update([prediction(1, "orange"), *face("red")[1:]])

    assert stable[0].label == "orange"


def test_confirmed_center_change_resets_for_new_face() -> None:
    tracker = TemporalColorTracker(window_size=5, center_switch_frames=2)
    tracker.update(face("red"))
    first_change = tracker.update(face("orange"))
    second_change = tracker.update(face("orange"))

    assert first_change[4].label == "red"
    assert second_change[4].label == "orange"


def test_missing_frames_clear_stale_history() -> None:
    tracker = TemporalColorTracker(window_size=5, missing_reset_frames=2)
    tracker.update(face("red"))
    tracker.mark_missing()
    tracker.mark_missing()

    stable = tracker.update(face("yellow"))

    assert {item.label for item in stable} == {"yellow"}


def test_confidence_reflects_vote_consensus() -> None:
    tracker = TemporalColorTracker(window_size=3, center_switch_frames=3)
    tracker.update(face("green", confidence=0.9))
    tracker.update(face("green", confidence=0.9))
    stable = tracker.update(
        [prediction(index, "blue" if index == 1 else "green", 0.9) for index in range(1, 10)]
    )

    assert stable[0].confidence == pytest.approx(0.6)


def test_tracker_rejects_duplicate_indices() -> None:
    tracker = TemporalColorTracker()
    with pytest.raises(ValueError, match="unique"):
        tracker.update([prediction(1, "red"), prediction(1, "blue")])

