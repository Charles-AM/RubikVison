from __future__ import annotations

import numpy as np
import pytest

from src.sticker_detector import draw_sticker_regions, extract_stickers


COLORS = [
    (255, 255, 255),
    (0, 255, 255),
    (0, 0, 255),
    (0, 128, 255),
    (255, 0, 0),
    (0, 180, 0),
    (80, 80, 80),
    (160, 80, 30),
    (30, 140, 210),
]


def make_nine_color_face() -> np.ndarray:
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    for index, color in enumerate(COLORS):
        row, column = divmod(index, 3)
        y1, y2 = row * 100, (row + 1) * 100
        x1, x2 = column * 100, (column + 1) * 100
        face[y1:y2, x1:x2] = color
        # Simulate a dark sticker border which center sampling should ignore.
        face[y1 : y1 + 12, x1:x2] = 0
        face[y2 - 12 : y2, x1:x2] = 0
        face[y1:y2, x1 : x1 + 12] = 0
        face[y1:y2, x2 - 12 : x2] = 0
    return face


def test_extract_stickers_returns_row_major_grid() -> None:
    regions = extract_stickers(make_nine_color_face())

    assert len(regions) == 9
    assert [region.index for region in regions] == list(range(1, 10))
    assert [(region.row, region.column) for region in regions] == [
        (row, column) for row in range(3) for column in range(3)
    ]


def test_extract_stickers_ignores_dark_borders() -> None:
    regions = extract_stickers(make_nine_color_face())

    assert [region.median_bgr for region in regions] == COLORS


def test_extract_stickers_supports_non_divisible_dimensions() -> None:
    face = np.full((302, 299, 3), 127, dtype=np.uint8)
    regions = extract_stickers(face)

    assert regions[0].bounds[:2] == (0, 0)
    assert regions[-1].bounds[2:] == (299, 302)


def test_extract_stickers_rejects_invalid_margin() -> None:
    with pytest.raises(ValueError, match="margin_ratio"):
        extract_stickers(make_nine_color_face(), margin_ratio=0.5)


def test_draw_sticker_regions_modifies_face() -> None:
    face = make_nine_color_face()
    regions = extract_stickers(face)
    before = face.copy()

    result = draw_sticker_regions(face, regions)

    assert result is face
    assert np.any(face != before)

