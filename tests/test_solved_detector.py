from __future__ import annotations

import numpy as np

from src.cube_state import FaceState
from src.solved_detector import SolvedStateDetector, draw_solved_status


def solved_face(color: str) -> FaceState:
    return FaceState((color,) * 9)


def test_face_requires_consecutive_stable_frames() -> None:
    detector = SolvedStateDetector(required_stable_frames=3)

    first = detector.update(solved_face("G"))
    second = detector.update(solved_face("G"))
    third = detector.update(solved_face("G"))

    assert not first.visible_face_solved
    assert not second.visible_face_solved
    assert third.visible_face_solved
    assert third.confirmed_faces == {"G"}


def test_missing_frame_breaks_confirmation_streak() -> None:
    detector = SolvedStateDetector(required_stable_frames=2)
    detector.update(solved_face("B"))
    detector.mark_missing()

    status = detector.update(solved_face("B"))

    assert status.stable_frames == 1
    assert not status.visible_face_solved


def test_unsolved_face_clears_previous_completion_evidence() -> None:
    detector = SolvedStateDetector(required_stable_frames=1)
    detector.update(solved_face("R"))

    status = detector.update(FaceState(tuple("RRRRGRRRR")))

    assert not status.confirmed_faces
    assert not status.cube_solved


def test_low_confidence_face_is_not_confirmed() -> None:
    detector = SolvedStateDetector(required_stable_frames=1, minimum_confidence=0.7)

    status = detector.update(solved_face("W"), average_confidence=0.6)

    assert not status.visible_face_solved
    assert not status.confirmed_faces


def test_all_six_confirmed_faces_complete_cube() -> None:
    detector = SolvedStateDetector(required_stable_frames=1)

    for color in "WYROBG":
        status = detector.update(solved_face(color))

    assert status.cube_solved
    assert status.confirmed_faces == set("WYROBG")


def test_draw_solved_status_modifies_frame() -> None:
    detector = SolvedStateDetector(required_stable_frames=1)
    status = detector.update(solved_face("Y"))
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    before = frame.copy()

    result = draw_solved_status(frame, status)

    assert result is frame
    assert np.any(frame != before)

