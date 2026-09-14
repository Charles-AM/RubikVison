"""Temporal smoothing for per-frame sticker color predictions."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from src.color_classifier import COLOR_NOTATION, ColorPrediction


class TemporalColorTracker:
    """Stabilize sticker colors with a rolling majority vote."""

    def __init__(
        self,
        window_size: int = 5,
        missing_reset_frames: int = 10,
        center_switch_frames: int = 2,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if missing_reset_frames < 1:
            raise ValueError("missing_reset_frames must be at least 1.")
        if center_switch_frames < 1:
            raise ValueError("center_switch_frames must be at least 1.")

        self.window_size = window_size
        self.missing_reset_frames = missing_reset_frames
        self.center_switch_frames = center_switch_frames
        self._history: dict[int, deque[ColorPrediction]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._missing_frames = 0
        self._pending_center: str | None = None
        self._pending_center_frames = 0

    def update(self, predictions: list[ColorPrediction]) -> list[ColorPrediction]:
        """Add one frame and return stable row-major predictions."""
        if not predictions:
            raise ValueError("predictions cannot be empty.")
        indices = [prediction.sticker_index for prediction in predictions]
        if len(indices) != len(set(indices)):
            raise ValueError("sticker indices must be unique.")

        self._missing_frames = 0
        if self._handle_face_switch(predictions):
            return [self._stable_prediction(index) for index in sorted(self._history)]
        for prediction in predictions:
            self._history[prediction.sticker_index].append(prediction)

        return [
            self._stable_prediction(index)
            for index in sorted(self._history)
            if self._history[index]
        ]

    def mark_missing(self) -> None:
        """Record a frame without a face and reset stale history if needed."""
        self._missing_frames += 1
        if self._missing_frames >= self.missing_reset_frames:
            self.reset()

    def reset(self) -> None:
        """Clear all tracked predictions."""
        self._history.clear()
        self._missing_frames = 0
        self._pending_center = None
        self._pending_center_frames = 0

    def _handle_face_switch(self, predictions: list[ColorPrediction]) -> bool:
        """Return True while an unconfirmed face change should be ignored."""
        center = next(
            (prediction for prediction in predictions if prediction.sticker_index == 5),
            None,
        )
        if center is None or 5 not in self._history or not self._history[5]:
            return False

        stable_center = self._stable_prediction(5).label
        if center.label == stable_center:
            self._pending_center = None
            self._pending_center_frames = 0
            return False

        if center.label == self._pending_center:
            self._pending_center_frames += 1
        else:
            self._pending_center = center.label
            self._pending_center_frames = 1

        if self._pending_center_frames >= self.center_switch_frames:
            self.reset()
            return False
        return True

    def _stable_prediction(self, sticker_index: int) -> ColorPrediction:
        history = list(self._history[sticker_index])
        counts = Counter(prediction.label for prediction in history)
        confidence_sums: dict[str, float] = defaultdict(float)
        latest_position: dict[str, int] = {}
        for position, prediction in enumerate(history):
            confidence_sums[prediction.label] += prediction.confidence
            latest_position[prediction.label] = position

        label = max(
            counts,
            key=lambda candidate: (
                counts[candidate],
                confidence_sums[candidate],
                latest_position[candidate],
            ),
        )
        winners = [prediction for prediction in history if prediction.label == label]
        latest = winners[-1]
        consensus = len(winners) / len(history)
        average_confidence = sum(item.confidence for item in winners) / len(winners)
        confidence = average_confidence * consensus
        return ColorPrediction(
            sticker_index=sticker_index,
            label=label,
            notation=COLOR_NOTATION[label],
            hsv=latest.hsv,
            confidence=confidence,
        )
