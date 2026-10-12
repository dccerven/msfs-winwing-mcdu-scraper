"""
Embraer E-Jets MCDU, from a real capture.

tests/data/ejets_mcdu.png is the RADIO page: large green and white values
under small labels, solid triangle arrows, and - unlike every other capture
here - separator lines drawn across the display.  The WinWing hardware has
no lines to draw, so they must not reach it as characters.  Before they were
erased they read as 36 characters in cells that are blank, a dash in every
blank cell they crossed and a '1' wherever the vertical one ran.

The crop is fixed rather than detected: Auto Detect currently cuts the
title row in half on this page, because its label rows sit closer to their
value rows than a uniform pitch implies.
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
from mcdu_parser import MCDUParser, erase_rules
from mcdu_templates import TemplateMatcher

CAPTURE = Path(__file__).parent / "data" / "ejets_mcdu.png"

#: (x, y, w, h) of the 24x14 grid, measured by eye from the text rows.
CROP = (0, 25, 808, 617)

#: The page as displayed.  '?' marks a glyph the CDU cannot draw - the swap
#: icons, the diamond and the filled square - which is not scored.
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


def load_crop():
    from PIL import Image
    image = np.array(Image.open(CAPTURE).convert("RGB"))
    x, y, w, h = CROP
    return image[y:y + h, x:x + w]


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
class TestEjetsCapture(unittest.TestCase):
    """A cold first read, as a new user's first frame would be."""

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
        cls.crop = load_crop()
        cls.parsed = MCDUParser(cls.crop, columns=24, rows=14,
                                source_id="ejets").parse_grid()

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
        total = correct = 0
        errors = []
        for row, line in enumerate(TRUTH):
            for col, want in enumerate(line):
                if want in " ?":
                    continue
                total += 1
                got = self._got(row, col)
                if got == want:
                    correct += 1
                else:
                    errors.append(f"R{row:02d}C{col:02d} {want!r}->{got!r}")
        # 119/130 before the separator lines were erased, 123 after.  The
        # rest are geometry: '.' read as '/', and the solid arrows.
        self.assertGreaterEqual(correct, 123,
                                f"{correct}/{total}; errors: {errors}")

    def test_separator_lines_are_not_read_as_text(self):
        junk = [f"R{row:02d}C{col:02d} {self._got(row, col)!r}"
                for row, line in enumerate(TRUTH)
                for col, want in enumerate(line)
                if want == " " and self._got(row, col) != " "]
        self.assertLessEqual(len(junk), 2, f"{len(junk)} in blank cells: {junk}")

    def test_value_rows_under_the_lines(self):
        """Rows 5 and 9 sit just below a line; row 8 just above one."""
        self.assertIn("NAV1", self._row(5))
        self.assertIn("NAV2", self._row(5))
        self.assertIn("XPDR", self._row(9))
        self.assertIn("738", self._row(9))
