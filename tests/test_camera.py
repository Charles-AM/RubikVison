from __future__ import annotations

import pytest

from src.camera import FPSCounter, normalize_source


def test_normalize_source_converts_camera_index() -> None:
    assert normalize_source("0") == 0
    assert normalize_source(" 2 ") == 2


def test_normalize_source_preserves_video_path() -> None:
    assert normalize_source("data/samples/solve.mp4") == "data/samples/solve.mp4"


def test_normalize_source_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_source("   ")


def test_fps_counter_uses_elapsed_time() -> None:
    times = iter([10.0, 10.05, 10.10])
    counter = FPSCounter(smoothing=0, clock=lambda: next(times))

    assert counter.update() == 0
    assert counter.update() == pytest.approx(20.0)
    assert counter.update() == pytest.approx(20.0)


def test_fps_counter_rejects_invalid_smoothing() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        FPSCounter(smoothing=1)

