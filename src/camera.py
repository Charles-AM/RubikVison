"""Video capture and frame-rate utilities for RubikVision."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import cv2
import numpy as np


Source = int | str | Path


def normalize_source(source: Source) -> int | str:
    """Convert camera-like values to an index and paths to strings."""
    if isinstance(source, int):
        return source

    value = str(source).strip()
    if not value:
        raise ValueError("Video source cannot be empty.")
    if value.isdecimal():
        return int(value)
    return value


class VideoCapture:
    """Small context-managed wrapper around ``cv2.VideoCapture``."""

    def __init__(self, source: Source = 0) -> None:
        self.source = normalize_source(source)
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> "VideoCapture":
        if isinstance(self.source, str) and not Path(self.source).is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.source}")

        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Could not open video source: {self.source}")
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None:
            raise RuntimeError("Video source has not been opened.")
        return self._capture.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoCapture":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.release()


class FPSCounter:
    """Calculate a stable FPS value using an exponential moving average."""

    def __init__(
        self,
        smoothing: float = 0.9,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in the range [0, 1).")
        self.smoothing = smoothing
        self.clock = clock
        self._last_time: float | None = None
        self.fps = 0.0

    def update(self) -> float:
        now = self.clock()
        if self._last_time is None:
            self._last_time = now
            return self.fps

        elapsed = now - self._last_time
        self._last_time = now
        if elapsed <= 0:
            return self.fps

        instantaneous = 1.0 / elapsed
        if self.fps == 0:
            self.fps = instantaneous
        else:
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instantaneous
        return self.fps

