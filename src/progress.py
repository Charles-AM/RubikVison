"""Clearly scoped visual progress metrics for an observed cube face."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.cube_state import FaceState


@dataclass(frozen=True)
class VisibleFaceProgress:
    """Agreement between visible stickers and the fixed center sticker."""

    matching_stickers: int
    total_stickers: int = 9

    @property
    def score(self) -> float:
        return self.matching_stickers / self.total_stickers


def calculate_visible_face_progress(state: FaceState) -> VisibleFaceProgress:
    """Count visible stickers matching the face's center color."""
    matches = sum(sticker == state.center for sticker in state.stickers)
    return VisibleFaceProgress(matching_stickers=matches)


def draw_face_progress(
    frame: np.ndarray,
    state: FaceState,
    progress: VisibleFaceProgress,
) -> np.ndarray:
    """Overlay the observed state and explicitly scoped consistency metric."""
    lines = (
        f"Face state: {state.compact}",
        (
            "Visible consistency: "
            f"{progress.matching_stickers}/{progress.total_stickers} "
            f"({progress.score:.0%})"
        ),
    )
    for index, text in enumerate(lines):
        origin = (16, 102 + index * 30)
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame

