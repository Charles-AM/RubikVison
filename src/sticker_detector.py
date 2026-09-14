"""Extract the nine sticker regions from a normalized cube face."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StickerRegion:
    """One sticker's position, sampling bounds, and representative BGR color."""

    row: int
    column: int
    bounds: tuple[int, int, int, int]
    sample_bounds: tuple[int, int, int, int]
    median_bgr: tuple[int, int, int]

    @property
    def index(self) -> int:
        """Return the sticker's one-based row-major index."""
        return self.row * 3 + self.column + 1


def extract_stickers(
    normalized_face: np.ndarray,
    margin_ratio: float = 0.2,
) -> list[StickerRegion]:
    """Split a normalized face into a 3x3 grid and sample sticker centers."""
    if normalized_face.size == 0:
        raise ValueError("Cannot extract stickers from an empty face.")
    if normalized_face.ndim != 3 or normalized_face.shape[2] != 3:
        raise ValueError("normalized_face must be a three-channel BGR image.")
    if not 0 <= margin_ratio < 0.5:
        raise ValueError("margin_ratio must be in the range [0, 0.5).")

    height, width = normalized_face.shape[:2]
    x_edges = [round(index * width / 3) for index in range(4)]
    y_edges = [round(index * height / 3) for index in range(4)]
    regions: list[StickerRegion] = []

    for row in range(3):
        for column in range(3):
            x1, x2 = x_edges[column], x_edges[column + 1]
            y1, y2 = y_edges[row], y_edges[row + 1]
            x_margin = round((x2 - x1) * margin_ratio)
            y_margin = round((y2 - y1) * margin_ratio)
            sx1, sx2 = x1 + x_margin, x2 - x_margin
            sy1, sy2 = y1 + y_margin, y2 - y_margin
            sample = normalized_face[sy1:sy2, sx1:sx2]
            median = np.median(sample.reshape(-1, 3), axis=0).astype(np.uint8)

            regions.append(
                StickerRegion(
                    row=row,
                    column=column,
                    bounds=(x1, y1, x2, y2),
                    sample_bounds=(sx1, sy1, sx2, sy2),
                    median_bgr=tuple(int(value) for value in median),
                )
            )

    return regions


def draw_sticker_regions(
    normalized_face: np.ndarray,
    regions: list[StickerRegion],
) -> np.ndarray:
    """Draw grid cells, center sampling boxes, and sticker indices."""
    for region in regions:
        x1, y1, x2, y2 = region.bounds
        sx1, sy1, sx2, sy2 = region.sample_bounds
        cv2.rectangle(normalized_face, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
        cv2.rectangle(normalized_face, (sx1, sy1), (sx2 - 1, sy2 - 1), (70, 255, 90), 2)
        cv2.putText(
            normalized_face,
            str(region.index),
            (sx1 + 5, sy1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            normalized_face,
            str(region.index),
            (sx1 + 5, sy1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return normalized_face

