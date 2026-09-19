"""Solve-session metrics and durable CSV history."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from uuid import uuid4


CSV_FIELDS = (
    "solve_id",
    "completed_at",
    "solve_time_seconds",
    "frames_processed",
    "average_fps",
    "average_color_confidence",
    "confirmed_faces",
)
FACE_ORDER = "WYROBG"


@dataclass(frozen=True)
class SolveRecord:
    solve_id: str
    completed_at: str
    solve_time_seconds: float
    frames_processed: int
    average_fps: float
    average_color_confidence: float
    confirmed_faces: str


@dataclass(frozen=True)
class SolveSummary:
    total_solves: int
    fastest_seconds: float
    slowest_seconds: float
    average_seconds: float
    median_seconds: float
    average_fps: float
    average_color_confidence: float

    def format(self) -> str:
        """Return a compact terminal-friendly analytics report."""
        return "\n".join(
            (
                "RubikVision Solve History",
                f"Total solves: {self.total_solves}",
                f"Fastest: {_format_duration(self.fastest_seconds)}",
                f"Slowest: {_format_duration(self.slowest_seconds)}",
                f"Average: {_format_duration(self.average_seconds)}",
                f"Median: {_format_duration(self.median_seconds)}",
                f"Average FPS: {self.average_fps:.2f}",
                f"Average color confidence: {self.average_color_confidence:.1%}",
            )
        )


class SolveSession:
    """Accumulate metrics only while a solve timer is running."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.solve_id: str | None = None
        self.active = False
        self.completed = False
        self.frames_processed = 0
        self._fps_total = 0.0
        self._fps_samples = 0
        self._confidence_total = 0.0
        self._confidence_samples = 0

    @property
    def has_started(self) -> bool:
        return self.solve_id is not None

    def start(self) -> None:
        """Start a new session or resume its metrics after a pause."""
        if self.completed:
            self.reset()
        if self.solve_id is None:
            self.solve_id = uuid4().hex
        self.active = True

    def pause(self) -> None:
        self.active = False

    def observe(self, fps: float, color_confidence: float | None = None) -> None:
        if not self.active:
            return
        self.frames_processed += 1
        if fps > 0:
            self._fps_total += fps
            self._fps_samples += 1
        if color_confidence is not None:
            self._confidence_total += color_confidence
            self._confidence_samples += 1

    def complete(
        self,
        solve_time_seconds: float,
        confirmed_faces: set[str] | frozenset[str],
        completed_at: datetime | None = None,
    ) -> SolveRecord | None:
        """Build one immutable record, returning None for invalid duplicates."""
        if not self.has_started or self.completed:
            return None
        self.active = False
        self.completed = True
        timestamp = completed_at or datetime.now().astimezone()
        average_fps = self._fps_total / self._fps_samples if self._fps_samples else 0.0
        average_confidence = (
            self._confidence_total / self._confidence_samples
            if self._confidence_samples
            else 0.0
        )
        faces = "".join(color for color in FACE_ORDER if color in confirmed_faces)
        return SolveRecord(
            solve_id=self.solve_id or "",
            completed_at=timestamp.isoformat(timespec="seconds"),
            solve_time_seconds=round(max(solve_time_seconds, 0.0), 2),
            frames_processed=self.frames_processed,
            average_fps=round(average_fps, 2),
            average_color_confidence=round(average_confidence, 4),
            confirmed_faces=faces,
        )


class SolveHistory:
    """Append solve records to a CSV while preventing duplicate IDs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: SolveRecord) -> bool:
        """Append a record and return False when its ID already exists."""
        if self._contains(record.solve_id):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(asdict(record))
        return True

    def records(self) -> list[SolveRecord]:
        """Load valid history rows while ignoring malformed entries."""
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return []
        records: list[SolveRecord] = []
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    records.append(
                        SolveRecord(
                            solve_id=row["solve_id"],
                            completed_at=row["completed_at"],
                            solve_time_seconds=float(row["solve_time_seconds"]),
                            frames_processed=int(row["frames_processed"]),
                            average_fps=float(row["average_fps"]),
                            average_color_confidence=float(
                                row["average_color_confidence"]
                            ),
                            confirmed_faces=row["confirmed_faces"],
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return records

    def summary(self) -> SolveSummary | None:
        """Calculate aggregate metrics, or None when no valid solves exist."""
        records = self.records()
        if not records:
            return None
        times = [record.solve_time_seconds for record in records]
        return SolveSummary(
            total_solves=len(records),
            fastest_seconds=min(times),
            slowest_seconds=max(times),
            average_seconds=mean(times),
            median_seconds=median(times),
            average_fps=mean(record.average_fps for record in records),
            average_color_confidence=mean(
                record.average_color_confidence for record in records
            ),
        )

    def _contains(self, solve_id: str) -> bool:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return False
        with self.path.open(newline="") as handle:
            return any(
                row.get("solve_id") == solve_id for row in csv.DictReader(handle)
            )


def _format_duration(seconds: float) -> str:
    minutes, remaining = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{remaining:05.2f}"
