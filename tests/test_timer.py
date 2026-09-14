from __future__ import annotations

import numpy as np

from src.timer import SolveTimer, TimerSnapshot, draw_timer


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_timer_starts_and_reports_elapsed_time() -> None:
    clock = FakeClock()
    timer = SolveTimer(clock=clock)
    timer.start()
    clock.now = 12.34

    snapshot = timer.snapshot()

    assert snapshot.state == "running"
    assert snapshot.elapsed_seconds == 12.34
    assert snapshot.display_time == "00:12.34"


def test_timer_stop_freezes_elapsed_time() -> None:
    clock = FakeClock()
    timer = SolveTimer(clock=clock)
    timer.start()
    clock.now = 65.5
    timer.stop()
    clock.now = 90.0

    snapshot = timer.snapshot()

    assert snapshot.state == "stopped"
    assert snapshot.elapsed_seconds == 65.5
    assert snapshot.display_time == "01:05.50"


def test_timer_can_resume_after_stop() -> None:
    clock = FakeClock()
    timer = SolveTimer(clock=clock)
    timer.start()
    clock.now = 5.0
    timer.stop()
    clock.now = 10.0
    timer.start()
    clock.now = 13.0

    assert timer.snapshot().elapsed_seconds == 8.0


def test_timer_reset_returns_to_ready() -> None:
    clock = FakeClock()
    timer = SolveTimer(clock=clock)
    timer.start()
    clock.now = 5.0
    timer.reset()

    snapshot = timer.snapshot()
    assert snapshot.state == "ready"
    assert snapshot.elapsed_seconds == 0.0


def test_draw_timer_modifies_frame() -> None:
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    before = frame.copy()

    result = draw_timer(frame, TimerSnapshot("running", 42.17))

    assert result is frame
    assert np.any(frame != before)

