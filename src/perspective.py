"""Perspective correction for detected Rubik's Cube faces."""

from __future__ import annotations

import cv2
import numpy as np

from src.face_detector import order_corners


DEFAULT_FACE_SIZE = 300


def warp_face(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: int = DEFAULT_FACE_SIZE,
) -> np.ndarray:
    """Transform a four-corner face region into a square image."""
    if frame.size == 0:
        raise ValueError("Cannot warp an empty frame.")
    if output_size < 2:
        raise ValueError("output_size must be at least 2 pixels.")

    source = order_corners(corners)
    edge = float(output_size - 1)
    destination = np.array(
        [[0, 0], [edge, 0], [edge, edge], [0, edge]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame, transform, (output_size, output_size))

