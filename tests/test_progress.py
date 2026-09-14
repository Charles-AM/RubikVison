from __future__ import annotations

import numpy as np

from src.cube_state import FaceState
from src.progress import calculate_visible_face_progress, draw_face_progress


def test_solved_visible_face_has_full_consistency() -> None:
    progress = calculate_visible_face_progress(FaceState(("G",) * 9))

    assert progress.matching_stickers == 9
    assert progress.score == 1.0


def test_mixed_visible_face_uses_center_as_reference() -> None:
    state = FaceState(tuple("RGWGGYGBG"))

    progress = calculate_visible_face_progress(state)

    assert state.center == "G"
    assert progress.matching_stickers == 5
    assert progress.score == 5 / 9


def test_draw_face_progress_modifies_frame() -> None:
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    state = FaceState(("O",) * 9)
    progress = calculate_visible_face_progress(state)
    before = frame.copy()

    result = draw_face_progress(frame, state, progress)

    assert result is frame
    assert np.any(frame != before)

