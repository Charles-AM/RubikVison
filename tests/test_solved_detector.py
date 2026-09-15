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


def test_brief_missing_frame_preserves_confirmation_streak() -> None:
    detector = SolvedStateDetector(required_stable_frames=2, missing_tolerance_frames=1)
    detector.update(solved_face("B"))
    detector.mark_missing()

    status = detector.update(solved_face("B"))

    assert status.stable_frames == 2
    assert status.visible_face_solved


def test_prolonged_missing_detection_breaks_confirmation_streak() -> None:
    detector = SolvedStateDetector(required_stable_frames=2, missing_tolerance_frames=1)
    detector.update(solved_face("B"))
    detector.mark_missing()
    detector.mark_missing()

    status = detector.update(solved_face("B"))

    assert status.stable_frames == 1
    assert not status.visible_face_solved


def test_single_unsolved_frame_does_not_remove_confirmation() -> None:
    detector = SolvedStateDetector(required_stable_frames=1, unsolved_stable_frames=2)
    detector.update(solved_face("R"))

    status = detector.update(FaceState(tuple("RRRRGRRRR")))

    assert status.confirmed_faces == {"R"}


def test_isolated_bad_frames_decay_instead_of_resetting_evidence() -> None:
    detector = SolvedStateDetector(required_stable_frames=4)
    detector.update(solved_face("Y"))
    detector.update(solved_face("Y"))
    detector.update(solved_face("Y"))

    status = detector.update(FaceState(tuple("RYYYYYYYY")))

    assert status.stable_frames == 2
    assert not status.visible_face_solved


def test_intermittent_good_frames_can_still_confirm_face() -> None:
    detector = SolvedStateDetector(required_stable_frames=4)
    unsolved = FaceState(tuple("YRRRRRRRR"))
    observations = [
        solved_face("R"),
        solved_face("R"),
        unsolved,
        solved_face("R"),
        solved_face("R"),
        solved_face("R"),
    ]

    for state in observations:
        status = detector.update(state)

    assert status.visible_face_solved
    assert "R" in status.confirmed_faces


def test_sustained_unsolved_face_removes_only_that_center() -> None:
    detector = SolvedStateDetector(required_stable_frames=1, unsolved_stable_frames=2)
    detector.update(solved_face("R"))
    detector.update(solved_face("G"))
    unsolved_red = FaceState(tuple("GGGGRGGGG"))

    detector.update(unsolved_red)
    status = detector.update(unsolved_red)

    assert status.confirmed_faces == {"G"}
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


def test_transitional_unsolved_faces_do_not_erase_scan_progress() -> None:
    detector = SolvedStateDetector(required_stable_frames=1, unsolved_stable_frames=3)
    detector.update(solved_face("B"))
    detector.update(FaceState(tuple("BRBBGBBBB")))
    detector.update(solved_face("O"))

    assert detector.status().confirmed_faces == {"B", "O"}


def test_draw_solved_status_modifies_frame() -> None:
    detector = SolvedStateDetector(required_stable_frames=1)
    status = detector.update(solved_face("Y"))
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    before = frame.copy()

    result = draw_solved_status(frame, status)

    assert result is frame
    assert np.any(frame != before)
