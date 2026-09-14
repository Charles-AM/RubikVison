from __future__ import annotations

import pytest

from src.color_classifier import ColorPrediction
from src.cube_state import FaceState


def prediction(index: int, notation: str) -> ColorPrediction:
    names = {"W": "white", "Y": "yellow", "R": "red", "O": "orange", "B": "blue", "G": "green"}
    return ColorPrediction(
        sticker_index=index,
        label=names[notation],
        notation=notation,
        hsv=(0, 0, 0),
        confidence=0.9,
    )


def test_face_state_preserves_row_major_order() -> None:
    notation = "RRWBGGYRB"
    predictions = [prediction(index, value) for index, value in enumerate(notation, 1)]

    state = FaceState.from_predictions(list(reversed(predictions)))

    assert state.compact == notation
    assert state.grid == (("R", "R", "W"), ("B", "G", "G"), ("Y", "R", "B"))
    assert state.center == "G"


def test_face_state_requires_nine_stickers() -> None:
    with pytest.raises(ValueError, match="exactly nine"):
        FaceState(("R",) * 8)


def test_face_state_rejects_invalid_notation() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        FaceState(("R",) * 8 + ("X",))


def test_from_predictions_requires_unique_complete_indices() -> None:
    predictions = [prediction(index, "R") for index in range(1, 9)]
    predictions.append(prediction(8, "R"))
    with pytest.raises(ValueError, match="indices 1 through 9"):
        FaceState.from_predictions(predictions)

