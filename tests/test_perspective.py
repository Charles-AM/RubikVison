from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.perspective import warp_face


def make_skewed_face() -> tuple[np.ndarray, np.ndarray]:
    normalized = np.zeros((180, 180, 3), dtype=np.uint8)
    normalized[:90, :90] = (0, 0, 255)
    normalized[:90, 90:] = (0, 255, 0)
    normalized[90:, :90] = (255, 0, 0)
    normalized[90:, 90:] = (0, 255, 255)

    frame = np.zeros((320, 420, 3), dtype=np.uint8)
    source = np.array([[0, 0], [179, 0], [179, 179], [0, 179]], dtype=np.float32)
    corners = np.array([[100, 60], [330, 90], [300, 275], [70, 240]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, corners)
    projected = cv2.warpPerspective(normalized, transform, (420, 320))
    frame = cv2.add(frame, projected)
    return frame, corners


def test_warp_face_returns_requested_square_size() -> None:
    frame, corners = make_skewed_face()

    warped = warp_face(frame, corners, output_size=180)

    assert warped.shape == (180, 180, 3)


def test_warp_face_restores_corner_regions() -> None:
    frame, corners = make_skewed_face()

    warped = warp_face(frame, corners, output_size=180)

    np.testing.assert_allclose(warped[45, 45], (0, 0, 255), atol=12)
    np.testing.assert_allclose(warped[45, 135], (0, 255, 0), atol=12)
    np.testing.assert_allclose(warped[135, 45], (255, 0, 0), atol=12)
    np.testing.assert_allclose(warped[135, 135], (0, 255, 255), atol=12)


def test_warp_face_rejects_invalid_output_size() -> None:
    frame, corners = make_skewed_face()
    with pytest.raises(ValueError, match="at least 2"):
        warp_face(frame, corners, output_size=1)

