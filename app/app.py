"""RubikVision OpenCV application."""

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
from src.color_classifier import HSVColorClassifier, draw_color_predictions  # noqa: E402
from src.face_detector import CubeFaceDetector, draw_detection  # noqa: E402
from src.perspective import warp_face  # noqa: E402
from src.sticker_detector import draw_sticker_regions, extract_stickers  # noqa: E402
from src.tracker import TemporalColorTracker  # noqa: E402


WINDOW_NAME = "RubikVision"
FACE_WINDOW_NAME = "RubikVision - Normalized Face"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a webcam or video file with a live FPS counter."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (default: 0) or path to a video file.",
    )
    parser.add_argument(
        "--debug-colors",
        action="store_true",
        help="Show HSV values instead of confidence in the normalized face.",
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


def run(source: str | int = 0, debug_colors: bool = False) -> int:
    """Run the display loop until the source ends or the user quits."""
    counter = FPSCounter()
    detector = CubeFaceDetector()
    classifier = HSVColorClassifier()
    tracker = TemporalColorTracker()

    try:
        with VideoCapture(source) as capture:
            print("RubikVision is running. Focus the video window and press Q or Esc to quit.")
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                detection = detector.detect(frame)
                if detection is not None:
                    normalized_face = warp_face(frame, detection.corners)
                    stickers = extract_stickers(normalized_face)
                    draw_sticker_regions(normalized_face, stickers)
                    predictions = classifier.classify_regions(stickers)
                    predictions = tracker.update(predictions)
                    draw_color_predictions(
                        normalized_face,
                        stickers,
                        predictions,
                        show_hsv=debug_colors,
                    )
                    cv2.imshow(FACE_WINDOW_NAME, normalized_face)
                else:
                    tracker.mark_missing()
                draw_detection(frame, detection)
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
        return run(args.source, debug_colors=args.debug_colors)
    except KeyboardInterrupt:
        print("\nRubikVision stopped.")
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"RubikVision error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
