"""RubikVision OpenCV application."""

from __future__ import annotations

import argparse
from pathlib import Path
import select
import sys

import cv2

# Allow the documented ``python app/app.py`` command from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import FPSCounter, VideoCapture  # noqa: E402
from src.color_classifier import (  # noqa: E402
    REQUIRED_COLORS,
    HSVColorClassifier,
    draw_color_predictions,
)
from src.cube_state import FaceState  # noqa: E402
from src.face_detector import CubeFaceDetector, draw_detection  # noqa: E402
from src.perspective import warp_face  # noqa: E402
from src.progress import calculate_visible_face_progress, draw_face_progress  # noqa: E402
from src.solved_detector import SolvedStateDetector, draw_solved_status  # noqa: E402
from src.sticker_detector import draw_sticker_regions, extract_stickers  # noqa: E402
from src.timer import SolveTimer, draw_timer  # noqa: E402
from src.tracker import TemporalColorTracker  # noqa: E402


WINDOW_NAME = "RubikVision"
FACE_WINDOW_NAME = "RubikVision - Normalized Face"
CALIBRATION_PATH = PROJECT_ROOT / "config" / "color_calibration.json"
CALIBRATION_KEYS = {
    ord("w"): "white",
    ord("y"): "yellow",
    ord("r"): "red",
    ord("o"): "orange",
    ord("b"): "blue",
    ord("g"): "green",
}


def parse_terminal_key(command: str) -> int | None:
    """Convert a terminal command into the same key code OpenCV returns."""
    value = command.strip().lower()
    return ord(value[0]) if value else None


def poll_terminal_key() -> int | None:
    """Read a completed terminal line without blocking the video loop."""
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    if not readable:
        return None
    return parse_terminal_key(sys.stdin.readline())


def capture_color_reference(
    classifier: HSVColorClassifier,
    tracker: TemporalColorTracker,
    label: str,
    center_color: tuple[int, int, int],
) -> int:
    """Save one calibration reference and return the completed color count."""
    classifier.calibrate(label, center_color)
    tracker.reset()
    return len(REQUIRED_COLORS.intersection(classifier.prototypes))


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
    parser.add_argument(
        "--calibrate-colors",
        action="store_true",
        help="Capture the visible center sticker with W/Y/R/O/B/G keys.",
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


def run(
    source: str | int = 0,
    debug_colors: bool = False,
    calibrate_colors: bool = False,
) -> int:
    """Run the display loop until the source ends or the user quits."""
    counter = FPSCounter()
    detector = CubeFaceDetector()
    classifier = HSVColorClassifier(calibration_path=CALIBRATION_PATH)
    tracker = TemporalColorTracker()
    timer = SolveTimer()
    solved_detector = SolvedStateDetector()
    completion_announced = False

    try:
        with VideoCapture(source) as capture:
            print("RubikVision is running. Focus the video window and press Q or Esc to quit.")
            if calibrate_colors:
                print(
                    "Calibration: show a face, then type W, Y, R, O, B, or G "
                    "in Terminal and press Enter."
                )
            elif classifier.is_calibrated:
                print("Loaded saved six-color camera calibration.")
            if not calibrate_colors:
                print("Timer controls: type S + Enter to start/stop, X + Enter to reset.")
            pending_calibration_label = None
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                center_color = None
                detection = detector.detect(frame)
                if detection is not None:
                    normalized_face = warp_face(frame, detection.corners)
                    stickers = extract_stickers(normalized_face)
                    draw_sticker_regions(normalized_face, stickers)
                    center_color = stickers[4].median_bgr
                    if calibrate_colors and pending_calibration_label is not None:
                        count = capture_color_reference(
                            classifier,
                            tracker,
                            pending_calibration_label,
                            center_color,
                        )
                        samples = classifier.sample_count(pending_calibration_label)
                        print(
                            f"Captured {pending_calibration_label} "
                            f"({count}/6, {samples} samples)."
                        )
                        pending_calibration_label = None
                    predictions = classifier.classify_regions(stickers)
                    predictions = tracker.update(predictions)
                    face_state = FaceState.from_predictions(predictions)
                    progress = calculate_visible_face_progress(face_state)
                    draw_face_progress(frame, face_state, progress)
                    average_confidence = sum(
                        prediction.confidence for prediction in predictions
                    ) / len(predictions)
                    solved_status = solved_detector.update(
                        face_state,
                        average_confidence=average_confidence,
                    )
                    draw_color_predictions(
                        normalized_face,
                        stickers,
                        predictions,
                        show_hsv=debug_colors,
                    )
                    cv2.imshow(FACE_WINDOW_NAME, normalized_face)
                else:
                    tracker.mark_missing()
                    solved_detector.mark_missing()
                    solved_status = solved_detector.status()
                draw_detection(frame, detection)
                draw_fps(frame, counter.update())
                draw_timer(frame, timer.snapshot())
                draw_solved_status(frame, solved_status)
                if solved_status.cube_solved and not completion_announced:
                    timer.stop()
                    completion_announced = True
                    print(f"Cube solved in {timer.snapshot().display_time}.")
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                terminal_key = poll_terminal_key()
                if terminal_key is not None:
                    key = terminal_key
                if key in (ord("q"), ord("Q"), 27):
                    break
                calibration_label = CALIBRATION_KEYS.get(key | 32)
                if calibrate_colors and calibration_label:
                    if center_color is None:
                        pending_calibration_label = calibration_label
                        print(
                            f"Waiting to capture {calibration_label}: "
                            "hold that face still until detection returns."
                        )
                    else:
                        count = capture_color_reference(
                            classifier,
                            tracker,
                            calibration_label,
                            center_color,
                        )
                        samples = classifier.sample_count(calibration_label)
                        print(
                            f"Captured {calibration_label} "
                            f"({count}/6, {samples} samples)."
                        )
                elif not calibrate_colors and (key | 32) == ord("s"):
                    if timer.snapshot().state != "running":
                        solved_detector.reset()
                        completion_announced = False
                    timer.toggle()
                    print(f"Timer {timer.snapshot().state}.")
                elif not calibrate_colors and (key | 32) == ord("x"):
                    timer.reset()
                    solved_detector.reset()
                    completion_announced = False
                    print("Timer reset.")
    finally:
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(
            args.source,
            debug_colors=args.debug_colors,
            calibrate_colors=args.calibrate_colors,
        )
    except KeyboardInterrupt:
        print("\nRubikVision stopped.")
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"RubikVision error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
