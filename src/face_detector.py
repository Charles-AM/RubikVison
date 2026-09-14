"""Traditional OpenCV detector for a visible Rubik's Cube face."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """A candidate cube face represented by four clockwise corner points."""

    corners: np.ndarray
    area: float
    score: float


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral corners in top-left, top-right, bottom-right order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    coordinate_sums = points.sum(axis=1)
    coordinate_differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(coordinate_sums)]
    ordered[2] = points[np.argmax(coordinate_sums)]
    ordered[1] = points[np.argmin(coordinate_differences)]
    ordered[3] = points[np.argmax(coordinate_differences)]
    return ordered


def _side_lengths(corners: np.ndarray) -> np.ndarray:
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def _maximum_corner_cosine(corners: np.ndarray) -> float:
    """Return 0 for right angles and values approaching 1 for sharp angles."""
    cosines: list[float] = []
    for index in range(4):
        vertex = corners[index]
        previous = corners[(index - 1) % 4] - vertex
        following = corners[(index + 1) % 4] - vertex
        denominator = np.linalg.norm(previous) * np.linalg.norm(following)
        if denominator == 0:
            return 1.0
        cosines.append(abs(float(np.dot(previous, following) / denominator)))
    return max(cosines)


class CubeFaceDetector:
    """Find the strongest square-like contour in a BGR frame."""

    def __init__(
        self,
        min_area_ratio: float = 0.03,
        max_area_ratio: float = 0.85,
        min_side_ratio: float = 0.45,
        max_corner_cosine: float = 0.55,
    ) -> None:
        if not 0 < min_area_ratio < max_area_ratio <= 1:
            raise ValueError("Area ratios must satisfy 0 < min < max <= 1.")
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_side_ratio = min_side_ratio
        self.max_corner_cosine = max_corner_cosine

    def edge_map(self, frame: np.ndarray) -> np.ndarray:
        """Build a color-aware edge image used for contour detection.

        Using the brightest BGR channel preserves boundaries that have little
        grayscale contrast, especially red stickers beside black gaps, while
        requiring only one Canny pass per frame.
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        blue, green, red = cv2.split(blurred)
        brightest_channel = cv2.max(cv2.max(blue, green), red)
        edges = cv2.Canny(brightest_channel, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    def detect(self, frame: np.ndarray) -> FaceDetection | None:
        """Return the best cube-face candidate, or ``None`` when none is found."""
        if frame.size == 0:
            return None

        frame_area = float(frame.shape[0] * frame.shape[1])
        min_area = frame_area * self.min_area_ratio
        max_area = frame_area * self.max_area_ratio
        contours, _ = cv2.findContours(
            self.edge_map(frame), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        best: FaceDetection | None = None
        sticker_candidates: list[tuple[np.ndarray, float, np.ndarray, float]] = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, closed=True)
            polygon = cv2.approxPolyDP(contour, 0.03 * perimeter, closed=True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue

            area = abs(float(cv2.contourArea(polygon)))
            if not frame_area * 0.001 <= area <= max_area:
                continue

            corners = order_corners(polygon)
            sides = _side_lengths(corners)
            longest_side = float(sides.max())
            if longest_side == 0 or float(sides.min()) / longest_side < self.min_side_ratio:
                continue

            corner_cosine = _maximum_corner_cosine(corners)
            if corner_cosine > self.max_corner_cosine:
                continue

            shape_quality = 1.0 - corner_cosine
            side_quality = float(sides.min()) / longest_side
            if area >= min_area:
                score = (area / frame_area) * shape_quality * side_quality
                detection = FaceDetection(corners=corners, area=area, score=score)
                if best is None or detection.score > best.score:
                    best = detection

            # A cube's outer edge is not always continuous. Retain smaller,
            # regular squares so a 3x3 group can provide the face boundary.
            if side_quality >= 0.65 and corner_cosine <= 0.35:
                center = corners.mean(axis=0)
                sticker_candidates.append((corners, area, center, longest_side))

        grouped = self._detect_from_sticker_group(sticker_candidates, frame_area)
        if grouped is not None and (best is None or grouped.score > best.score):
            best = grouped

        return best

    def _detect_from_sticker_group(
        self,
        candidates: list[tuple[np.ndarray, float, np.ndarray, float]],
        frame_area: float,
    ) -> FaceDetection | None:
        """Estimate a face boundary from nearby, similarly sized stickers."""
        unique: list[tuple[np.ndarray, float, np.ndarray, float]] = []
        for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
            _, _, center, side = candidate
            if any(np.linalg.norm(center - item[2]) < side * 0.3 for item in unique):
                continue
            unique.append(candidate)

        best: FaceDetection | None = None
        for _, seed_area, seed_center, seed_side in unique:
            group = [
                item
                for item in unique
                if 0.5 <= item[1] / seed_area <= 2.0
                and np.linalg.norm(item[2] - seed_center) <= seed_side * 5.0
            ]
            if len(group) < 4:
                continue

            # Nearby background squares can exist, so use at most the nine
            # closest candidates expected for a standard cube face.
            group.sort(key=lambda item: float(np.linalg.norm(item[2] - seed_center)))
            group = group[:9]
            all_corners = np.vstack([item[0] for item in group]).astype(np.float32)
            rectangle = cv2.minAreaRect(all_corners)
            corners = order_corners(cv2.boxPoints(rectangle))
            area = abs(float(cv2.contourArea(corners)))
            if not frame_area * self.min_area_ratio <= area <= frame_area * self.max_area_ratio:
                continue

            sides = _side_lengths(corners)
            side_quality = float(sides.min() / sides.max())
            support = min(len(group) / 9.0, 1.0)
            score = (area / frame_area) * side_quality * support
            detection = FaceDetection(corners=corners, area=area, score=score)
            if best is None or detection.score > best.score:
                best = detection

        return best


def draw_detection(frame: np.ndarray, detection: FaceDetection | None) -> np.ndarray:
    """Overlay the detected quadrilateral and current detector status."""
    if detection is None:
        label = "Cube face: searching"
        color = (0, 190, 255)
    else:
        corners = detection.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [corners], isClosed=True, color=(70, 255, 90), thickness=3)
        center = detection.corners.mean(axis=0).astype(int)
        cv2.drawMarker(
            frame,
            tuple(center),
            (70, 255, 90),
            cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        label = "Cube face: detected"
        color = (70, 255, 90)

    cv2.putText(
        frame,
        label,
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame
