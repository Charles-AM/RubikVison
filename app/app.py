"""RubikVision Phase 1 OpenCV application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

# Allow the documented ``python app/app.py`` command from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import FPSCounter, VideoCapture  # noqa: E402


WINDOW_NAME = "RubikVision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a webcam or video file with a live FPS counter."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (default: 0) or path to a video file.",
    )
    return parser.parse_args()


def draw_fps(frame, fps: float):
    """Draw the FPS overlay and return the same frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    return frame


def run(source: str | int = 0) -> int:
    """Run the display loop until the source ends or the user quits."""
    counter = FPSCounter()

    try:
        with VideoCapture(source) as capture:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                draw_fps(frame, counter.update())
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args.source)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"RubikVision error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

