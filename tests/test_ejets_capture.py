"""
Embraer E-Jets MCDU, from a real capture.

tests/data/ejets_mcdu.png is the RADIO page: large green and white values
under small labels, solid triangle arrows, symbols the CDU cannot draw, and -
unlike every other capture here - separator lines drawn across the display.

What it exposed, all of it now read correctly:

* the lines were read as text - 36 characters in cells that are blank - and
  the vertical one merged the rows it crossed, so Auto Detect cut the title
  in half and every row after it straddled two lines of text;
* the solid triangles read as '→' and '>' where they point left and right;
* the bold '0' and 'N' fill enough of their cells to pass for reverse video,
  and were read inside out;
* the swap icons, the diamond and the filled square came out as '4', 'M'
  and 'O' instead of blank.
"""

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import mcdu_parser
import mcdu_templates
from aircraft_profiles import PROFILES
from mcdu_detector import detect_mcdu_region
from mcdu_parser import MCDUParser, erase_rules
from mcdu_templates import TemplateMatcher

CAPTURE = Path(__file__).parent / "data" / "ejets_mcdu.png"

#: Where the 24x14 grid lies, (x, y, w, h).  Found by a search over origin
#: and width for the crop that reads best; Auto Detect has to land near it.
GRID = (7, 24, 790, 618)

#: The page as displayed.  '?' marks a glyph the CDU cannot draw - the swap
#: icons, the diamond and the filled square - which must come out blank.
TRUTH = [
    "         RADIO       1/2",
    " COM1              COM2 ",
    "?124.000        124.850?",
    "                        ",
    " 124.850        124.850 ",
    " NAV1   FMS  FMS   NAV2 ",
    "?111.50 AUTOAUTO 111.50?",
    "                        ",
    " 113.90          113.90 ",
    "          738      XPDR ",
    "←TCAS/XPDR         1234→",
    "                        ",
    "?STBY TA/RA       IDENT?",
    "YTJI                    ",
]


def load():
    from PIL import Image
    return np.array(Image.open(CAPTURE).convert("RGB"))


class TestEraseRules(unittest.TestCase):
    """What counts as a drawn line, on a 24x14 grid of 20x30 cells."""

    CELL_W, CELL_H = 20, 30

    def _blank(self):
        return np.zeros((14 * self.CELL_H, 24 * self.CELL_W, 3), np.uint8)

    def _erase(self, image):
        return erase_rules(image, self.CELL_W, self.CELL_H, ink_level=80)

    def test_lines_across_and_down_are_erased(self):
        image = self._blank()
        image[100:102, :] = 255
        image[:300, 240:242] = 255
        self.assertFalse(self._erase(image).any())

    def test_glyphs_beside_a_line_survive(self):
        image = self._blank()
        image[100:102, :] = 255
        image[70:95, 45:48] = 255       # a '1' standing on the line
        cleaned = self._erase(image)
        self.assertFalse(cleaned[100:102].any())
        self.assertTrue(cleaned[70:95, 45:48].all())

    def test_a_row_of_dashes_is_not_a_line(self):
        """The ATR draws a full row of dashes; each sits in its own cell."""
        image = self._blank()
        for col in range(24):
            image[50:52, col * 20 + 4:col * 20 + 16] = 255
        self.assertIs(self._erase(image), image)

    def test_reverse_video_block_survives(self):
        """A block's top and bottom margins are long thin runs too."""
        image = self._blank()
        image[60:90, 20:160] = 255
        for col in range(1, 8):
            # Glyphs reach to 2px of the edge: margins thin enough to pass
            # for a line, so only the bright block beside them saves them.
            image[62:88, col * 20 + 8:col * 20 + 12] = 0
        self.assertIs(self._erase(image), image)

    def test_nothing_to_erase_returns_the_same_image(self):
        image = self._blank()
        image[70:95, 45:48] = 255
        self.assertIs(self._erase(image), image)


@unittest.skipIf(not CAPTURE.exists(), "E-Jets capture missing")
class TestEjetsDetection(unittest.TestCase):

    def assertNearGrid(self, found, offset=(0, 0)):
        self.assertIsNotNone(found, "nothing detected")
        want = (GRID[0] + offset[0], GRID[1] + offset[1], GRID[2], GRID[3])
        for got, expected, name in zip(found, want, "xywh"):
            self.assertLessEqual(abs(got - expected), 4,
                                 f"{name}: got {found}, want about {want}")

    def test_finds_the_grid(self):
        self.assertNearGrid(detect_mcdu_region(load()))

    def test_finds_the_grid_inside_a_larger_window(self):
        """As captured live: the pop-out is wider than the display."""
        image = load()
        window = np.zeros((800, 1700, 3), np.uint8)
        window[51:51 + image.shape[0], 882:882 + image.shape[1]] = image
        self.assertNearGrid(detect_mcdu_region(window), offset=(882, 51))


@unittest.skipIf(not CAPTURE.exists(), "E-Jets capture missing")
class TestEjetsCapture(unittest.TestCase):
    """A cold first read on the detected grid, as a first frame would be."""

    @classmethod
    def setUpClass(cls):
        for line in TRUTH:
            assert len(line) == 24, line
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._saved = (mcdu_templates._template_matcher,
                      dict(mcdu_parser._prev_row_imgs),
                      dict(mcdu_parser._prev_row_ocr))
        mcdu_parser._prev_row_imgs.clear()
        mcdu_parser._prev_row_ocr.clear()
        mcdu_templates._template_matcher = TemplateMatcher(
            template_path=Path(cls._tmpdir.name) / "ejets.npz",
            seed_dir=PROFILES["ejets"].seed_dir())
        image = load()
        x, y, w, h = detect_mcdu_region(image)
        cls.parser = MCDUParser(image[y:y + h, x:x + w], columns=24, rows=14,
                                source_id="ejets")
        cls.parsed = cls.parser.parse_grid()

    @classmethod
    def tearDownClass(cls):
        matcher, imgs, ocr = cls._saved
        mcdu_templates._template_matcher = matcher
        mcdu_parser._prev_row_imgs.clear()
        mcdu_parser._prev_row_ocr.clear()
        mcdu_parser._prev_row_imgs.update(imgs)
        mcdu_parser._prev_row_ocr.update(ocr)
        cls._tmpdir.cleanup()

    def _got(self, row, col):
        cell = self.parsed[row * 24 + col]
        return cell[0] if cell else " "

    def _row(self, row):
        return "".join(self._got(row, col) for col in range(24))

    def test_characters_read_back(self):
        errors = [f"R{row:02d}C{col:02d} {want!r}->{self._got(row, col)!r}"
                  for row, line in enumerate(TRUTH)
                  for col, want in enumerate(line)
                  if want not in " ?" and self._got(row, col) != want]
        # 119 of 130 when this capture was added; every one since the
        # lines, the arrows, reverse video and Auto Detect were fixed.
        self.assertEqual(errors, [])

    def test_separator_lines_are_not_read_as_text(self):
        junk = [f"R{row:02d}C{col:02d} {self._got(row, col)!r}"
                for row, line in enumerate(TRUTH)
                for col, want in enumerate(line)
                if want == " " and self._got(row, col) != " "]
        self.assertEqual(junk, [])

    def test_undrawable_symbols_come_out_blank(self):
        stray = [f"R{row:02d}C{col:02d} {self._got(row, col)!r}"
                 for row, line in enumerate(TRUTH)
                 for col, want in enumerate(line)
                 if want == "?" and self._got(row, col) != " "]
        self.assertEqual(stray, [])

    def test_solid_triangles_are_arrows(self):
        self.assertEqual(self._got(10, 0), "←")
        self.assertEqual(self._got(10, 23), "→")

    def test_bold_glyphs_are_not_reverse_video(self):
        inverted = [(row, col) for row in range(14) for col in range(24)
                    if len(self.parsed[row * 24 + col]) > 3]
        self.assertEqual(inverted, [])
