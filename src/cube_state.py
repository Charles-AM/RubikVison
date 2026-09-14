"""Rubik's Cube face-state representation."""

from __future__ import annotations

from dataclasses import dataclass

from src.color_classifier import COLOR_NOTATION, ColorPrediction


VALID_NOTATION = frozenset(COLOR_NOTATION.values())


@dataclass(frozen=True)
class FaceState:
    """Nine sticker colors in row-major order using standard cube notation."""

    stickers: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.stickers) != 9:
            raise ValueError("A face state must contain exactly nine stickers.")
        invalid = set(self.stickers) - VALID_NOTATION
        if invalid:
            raise ValueError(f"Invalid sticker notation: {sorted(invalid)}")

    @classmethod
    def from_predictions(cls, predictions: list[ColorPrediction]) -> "FaceState":
        """Create a state from exactly one prediction for sticker indices 1-9."""
        by_index = {prediction.sticker_index: prediction for prediction in predictions}
        if set(by_index) != set(range(1, 10)) or len(predictions) != 9:
            raise ValueError("Predictions must contain sticker indices 1 through 9 once.")
        return cls(tuple(by_index[index].notation for index in range(1, 10)))

    @property
    def center(self) -> str:
        return self.stickers[4]

    @property
    def compact(self) -> str:
        return "".join(self.stickers)

    @property
    def grid(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            tuple(self.stickers[start : start + 3])
            for start in range(0, 9, 3)
        )

