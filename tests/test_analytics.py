from __future__ import annotations

import csv
from datetime import datetime, timezone

from src.analytics import SolveHistory, SolveRecord, SolveSession


def test_session_accumulates_only_active_frames() -> None:
    session = SolveSession()
    session.start()
    session.observe(30.0, 0.8)
    session.observe(20.0, 0.6)
    session.pause()
    session.observe(100.0, 1.0)

    record = session.complete(
        42.345,
        set("WYROBG"),
        completed_at=datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert record is not None
    assert record.solve_time_seconds == 42.34
    assert record.frames_processed == 2
    assert record.average_fps == 25.0
    assert record.average_color_confidence == 0.7
    assert record.confirmed_faces == "WYROBG"


def test_session_can_only_complete_once() -> None:
    session = SolveSession()
    session.start()

    assert session.complete(10.0, set("WYROBG")) is not None
    assert session.complete(10.0, set("WYROBG")) is None


def test_history_creates_header_and_appends_records(tmp_path) -> None:
    path = tmp_path / "outputs" / "solve_history.csv"
    history = SolveHistory(path)
    first = SolveRecord("one", "2026-09-19T12:00:00+00:00", 40.0, 100, 25.0, 0.9, "WYROBG")
    second = SolveRecord("two", "2026-09-19T12:05:00+00:00", 35.0, 90, 27.0, 0.92, "WYROBG")

    assert history.append(first)
    assert history.append(second)

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["solve_id"] for row in rows] == ["one", "two"]
    assert rows[1]["solve_time_seconds"] == "35.0"


def test_history_rejects_duplicate_solve_id(tmp_path) -> None:
    history = SolveHistory(tmp_path / "solve_history.csv")
    record = SolveRecord("same", "2026-09-19T12:00:00+00:00", 10.0, 30, 30.0, 0.9, "WYROBG")

    assert history.append(record)
    assert not history.append(record)

    with history.path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1

