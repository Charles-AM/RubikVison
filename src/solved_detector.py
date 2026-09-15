"""Stable visible-face and whole-cube solved-state confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.cube_state import FaceState, VALID_NOTATION


@dataclass(frozen=True)
class SolvedStatus:
    visible_face_solved: bool
    stable_frames: int
    required_stable_frames: int
    confirmed_faces: frozenset[str]
    cube_solved: bool


class SolvedStateDetector:
    """Confirm solved faces over time and require all six for completion."""

    def __init__(
        self,
        required_stable_frames: int = 15,
        minimum_confidence: float = 0.55,
        unsolved_stable_frames: int = 5,
    ) -> None:
        if required_stable_frames < 1:
            raise ValueError("required_stable_frames must be at least 1.")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in the range [0, 1].")
        if unsolved_stable_frames < 1:
            raise ValueError("unsolved_stable_frames must be at least 1.")
        self.required_stable_frames = required_stable_frames
        self.minimum_confidence = minimum_confidence
        self.unsolved_stable_frames = unsolved_stable_frames
        self._current_center: str | None = None
        self._stable_frames = 0
        self._confirmed_faces: set[str] = set()
        self._unsolved_center: str | None = None
        self._unsolved_frames = 0

    def update(self, state: FaceState, average_confidence: float = 1.0) -> SolvedStatus:
        """Process one observed face and return current completion evidence."""
        face_uniform = all(sticker == state.center for sticker in state.stickers)
        sufficiently_confident = average_confidence >= self.minimum_confidence

        if not face_uniform or not sufficiently_confident:
            self._current_center = None
            self._stable_frames = 0
            if not face_uniform:
                if state.center == self._unsolved_center:
                    self._unsolved_frames += 1
                else:
                    self._unsolved_center = state.center
                    self._unsolved_frames = 1
                if self._unsolved_frames >= self.unsolved_stable_frames:
                    self._confirmed_faces.discard(state.center)
            return self.status()

        self._unsolved_center = None
        self._unsolved_frames = 0
        if state.center == self._current_center:
            self._stable_frames += 1
        else:
            self._current_center = state.center
            self._stable_frames = 1

        if self._stable_frames >= self.required_stable_frames:
            self._confirmed_faces.add(state.center)
        return self.status()

    def mark_missing(self) -> None:
        """Break the consecutive-frame streak when no face is visible."""
        self._current_center = None
        self._stable_frames = 0
        self._unsolved_center = None
        self._unsolved_frames = 0

    def reset(self) -> None:
        self._current_center = None
        self._stable_frames = 0
        self._confirmed_faces.clear()
        self._unsolved_center = None
        self._unsolved_frames = 0

    def status(self) -> SolvedStatus:
        visible_solved = (
            self._current_center is not None
            and self._stable_frames >= self.required_stable_frames
        )
        return SolvedStatus(
            visible_face_solved=visible_solved,
            stable_frames=min(self._stable_frames, self.required_stable_frames),
            required_stable_frames=self.required_stable_frames,
            confirmed_faces=frozenset(self._confirmed_faces),
            cube_solved=self._confirmed_faces == VALID_NOTATION,
        )


def draw_solved_status(frame: np.ndarray, status: SolvedStatus) -> np.ndarray:
    """Overlay conservative solved-state evidence on the camera frame."""
    if status.cube_solved:
        line_one = "CUBE SOLVED"
        color = (70, 255, 90)
    elif status.visible_face_solved:
        line_one = "Visible face: SOLVED"
        color = (70, 255, 90)
    elif status.stable_frames:
        line_one = (
            "Confirming solved face: "
            f"{status.stable_frames}/{status.required_stable_frames}"
        )
        color = (0, 215, 255)
    else:
        line_one = "Visible face: not confirmed solved"
        color = (180, 180, 180)

    confirmed = ",".join(sorted(status.confirmed_faces)) or "none"
    lines = (
        line_one,
        f"Faces confirmed: {len(status.confirmed_faces)}/6 ({confirmed})",
    )
    for index, text in enumerate(lines):
        origin = (16, 194 + index * 30)
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame
