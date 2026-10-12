"""
Drawn rules: separator lines a display draws between its fields.

The E-Jets RADIO page has two across and one down.  They are not text, and
both the grid detector and the parser go wrong when they are taken for it:
the vertical one joins every text row it crosses into one band, so the
detector cannot see the row pitch, and the parser read the lines as a dash
in every blank cell they crossed and a '1' wherever the vertical one ran.

A rule is told from text by three things together:

* length: a straight run of ink longer than any glyph, or any row of them,
  since the gaps between neighbouring glyphs break a run;
* thickness: a stroke, not a solid band;
* dark on both sides: the margin of a reverse-video block is a long thin
  bright run too, but the block itself lies along one side of it.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

#: The most ink either side of a rule may carry, as a fraction of its length.
MAX_SIDE_INK = 0.3


def find_rules(ink: np.ndarray, min_length: Tuple[int, int],
               max_thickness: Tuple[float, float]) -> np.ndarray:
    """Pixels of *ink* that belong to drawn rules.

    Args:
        ink: 2-D mask, non-zero where there is ink.
        min_length: Shortest run that counts, ``(across, down)`` in pixels.
        max_thickness: Thickest stroke that counts, ``(across, down)``:
            the height of a horizontal rule, the width of a vertical one.

    Returns:
        A uint8 mask, 1 on rule pixels.
    """
    ink = (ink > 0).astype(np.uint8)
    rules = np.zeros_like(ink)
    last_row, last_col = ink.shape[0] - 1, ink.shape[1] - 1

    for horizontal, length, thickest in ((True, min_length[0], max_thickness[0]),
                                         (False, min_length[1], max_thickness[1])):
        length = int(round(length))
        if length < 3:
            continue
        kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
        runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(runs, 8)
        for i in range(1, count):
            x, y, w, h = stats[i, :4]
            if (h if horizontal else w) > thickest:
                continue
            if horizontal:
                sides = (ink[max(0, y - 2), x:x + w],
                         ink[min(last_row, y + h + 1), x:x + w])
            else:
                sides = (ink[y:y + h, max(0, x - 2)],
                         ink[y:y + h, min(last_col, x + w + 1)])
            if all(side.mean() <= MAX_SIDE_INK for side in sides):
                rules[labels == i] = 1
    return rules
