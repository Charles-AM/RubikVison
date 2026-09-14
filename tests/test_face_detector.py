from __future__ import annotations

import cv2
import numpy as np

from src.face_detector import CubeFaceDetector, draw_detection, order_corners


def make_cube_face_image() -> np.ndarray:
    image = np.full((500, 700, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (200, 100), (500, 400), (225, 225, 225), -1)
    cv2.rectangle(image, (200, 100), (500, 400), (5, 5, 5), 8)
    for offset in (300, 400):
        cv2.line(image, (offset, 100), (offset, 400), (10, 10, 10), 5)
    for offset in (200, 300):
        cv2.line(image, (200, offset), (500, offset), (10, 10, 10), 5)
    return image


def make_low_grayscale_contrast_red_face() -> np.ndarray:
    """Create red stickers whose grayscale value nearly matches the background."""
    image = np.full((500, 700, 3), 65, dtype=np.uint8)
    for row in range(3):
        for column in range(3):
            x = 200 + column * 100
            y = 100 + row * 100
            cv2.rectangle(image, (x + 5, y + 5), (x + 95, y + 95), (0, 0, 220), -1)
    return image


def test_order_corners_is_deterministic() -> None:
    shuffled = np.array([[80, 90], [10, 20], [10, 90], [80, 20]])
    ordered = order_corners(shuffled)
    np.testing.assert_array_equal(
        ordered,
        np.array([[10, 20], [80, 20], [80, 90], [10, 90]], dtype=np.float32),
    )


def test_detector_finds_synthetic_cube_face() -> None:
    detection = CubeFaceDetector().detect(make_cube_face_image())

    assert detection is not None
    assert detection.area > 80_000
    np.testing.assert_allclose(detection.corners[0], [200, 100], atol=12)
    np.testing.assert_allclose(detection.corners[2], [500, 400], atol=12)


def test_detector_rejects_blank_frame() -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    assert CubeFaceDetector().detect(frame) is None


def test_detector_finds_red_face_with_low_grayscale_contrast() -> None:
    detection = CubeFaceDetector().detect(make_low_grayscale_contrast_red_face())

    assert detection is not None
    assert detection.area > 75_000


def test_draw_detection_modifies_frame() -> None:
    frame = make_cube_face_image()
    detection = CubeFaceDetector().detect(frame)
    before = frame.copy()

    result = draw_detection(frame, detection)

    assert result is frame
    assert np.any(frame != before)
