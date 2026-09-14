"""Solve timing and timer overlay utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class TimerSnapshot:
    state: str
    elapsed_seconds: float

    @property
    def display_time(self) -> str:
        minutes, seconds = divmod(self.elapsed_seconds, 60)
        return f"{int(minutes):02d}:{seconds:05.2f}"


class SolveTimer:
    """Monotonic start/stop/reset timer with an injectable clock for tests."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self._state = "ready"
        self._started_at: float | None = None
        self._elapsed = 0.0

    def start(self) -> None:
        if self._state != "running":
            self._started_at = self.clock()
            self._state = "running"

    def stop(self) -> None:
        if self._state == "running" and self._started_at is not None:
            self._elapsed += self.clock() - self._started_at
            self._started_at = None
            self._state = "stopped"

    def toggle(self) -> None:
        if self._state == "running":
            self.stop()
        else:
            self.start()

    def reset(self) -> None:
        self._state = "ready"
        self._started_at = None
        self._elapsed = 0.0

    def snapshot(self) -> TimerSnapshot:
        elapsed = self._elapsed
        if self._state == "running" and self._started_at is not None:
            elapsed += self.clock() - self._started_at
        return TimerSnapshot(state=self._state, elapsed_seconds=max(elapsed, 0.0))


def draw_timer(frame: np.ndarray, snapshot: TimerSnapshot) -> np.ndarray:
    """Overlay solve time and timer state on the camera frame."""
    state_colors = {
        "ready": (0, 215, 255),
        "running": (70, 255, 90),
        "stopped": (255, 180, 70),
    }
    color = state_colors[snapshot.state]
    text = f"Solve time: {snapshot.display_time} [{snapshot.state.upper()}]"
    origin = (16, 162)
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame

