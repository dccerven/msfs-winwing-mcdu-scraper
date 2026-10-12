"""
Embraer E-Jets MCDU, from real captures.

The tests/data/ejets_window_*.png captures are the whole pop-out window as
Auto Detect sees it: the CDU unit with its bezel, line-select keys and
keyboard, the screen in the top half.  tests/data/ejet_mcdu.png is the RADIO
screen alone, cropped by hand.

What these exposed, all of it now read correctly:

* the keyboard is a character grid of its own, and pulled Auto Detect to
  half the screen's pitch: the grid landed over the screen's top-left
  quarter, so the box had to be drawn by hand;
* a page that writes only every other row (PERF INDEX) had no one-row
  spacing at all, and its two-row gaps were taken for the pitch;
* the RADIO page's separator lines were read as text - 36 characters in
  blank cells - and the vertical one merged the rows it crossed;
* the solid triangles read as '→' and '>' where they point left and right;
* the bold '0' and 'N' fill enough of their cells to pass for reverse
  video, and were read inside out;
* grey (unavailable) prompts peak at 78-89, under the fixed ink threshold,
  and DES FORECAST read as OES CAAT;
* the fringe of thin white glyphs coloured them grey;
* the swap icons, the diamond, the filled square and '%' came out as
  letters instead of blank.
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
from mcdu_detector import _find_screen, detect_mcdu_region
from mcdu_parser import MCDUParser, erase_rules
from mcdu_templates import TemplateMatcher

DATA = Path(__file__).parent / "data"
SCREEN = DATA / "ejet_mcdu.png"

# Each page as displayed.  '?' marks a glyph the CDU cannot draw - the swap
# icons, the diamond, the filled square, '%' - which must come out blank.

RADIO = [
    "         RADIO       1/2",
    " COM1              COM2 ",
    "?124.000        124.850?",
    "                        ",
    " 124.850        124.850 ",
    " NAV1   FMS  FMS   NAV2 ",
    "?111.50 AUTOAUTO 111.50?",
    "                        ",
    " 113.90          113.90 ",
    "         1661      XPDR ",
    "←TCAS/XPDR         1234→",
    "                        ",
    "?STBY TA/RA       IDENT?",
    "KABQ                    ",
]

PERF_INDEX = [
    "       PERF INDEX    1/2",
    "                        ",
    "←PERF INIT      TAKEOFF→",
    "                        ",
    "←DEP LIM          CLIMB→",
    "                        ",
    "←CRUISE    DES FORECAST→",
    "                        ",
    "←DESCENT        APP SPD→",
    "                        ",
    "←LANDING         GA LIM→",
    "                        ",
    "←PERF PLAN    PERF DATA→",
    "                        ",
]

#: Colours on PERF INDEX: 'e' is grey, an unavailable prompt.
PERF_INDEX_COLOURS = [
    ".......wwww.wwwww....www",
    "........................",
    "wwwww.wwww......wwwwwwww",
    "........................",
    "wwww.www..........wwwwww",
    "........................",
    "wwwwwww....eee.eeeeeeeee",
    "........................",
    "wwwwwwww........www.wwww",
    "........................",
    "wwwwwwww.........ee.eeee",
    "........................",
    "eeeee.eeee....wwww.wwwww",
    "........................",
]

PERF_INIT = [
    " PERFORMANCE INIT-LB 2/3",
    " CRZ ALT       STEP INC ",
    "☐☐☐☐☐                  0",
    " CRZ WINDS      ISA DEV ",
    "000T/00             +0°C",
    "                    ZFW ",
    "                   ☐☐☐☐☐",
    "           (GAUGE) FUEL ",
    "            (5000)  5000",
    " TO CG        GROSS WT  ",
    "☐☐.☐?                   ",
    "                        ",
    "                        ",
    "                        ",
]

#: window capture -> (grid Auto Detect must land near, page text)
WINDOWS = {
    "ejets_window_radio.png": ((883, 57, 790, 618), RADIO),
    "ejets_window_perf_index.png": ((880, 57, 794, 621), PERF_INDEX),
    "ejets_window_perf_init.png": ((881, 57, 794, 618), PERF_INIT),
}


def load(path):
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


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


class TestEjetsDetection(unittest.TestCase):

    def test_finds_the_grid_not_the_keyboard(self):
        for name, (grid, _) in WINDOWS.items():
            if not (DATA / name).exists():
                continue
            with self.subTest(name):
                image = load(DATA / name)
                x, y, w, h = _find_screen(image)
                gx, gy, gw, gh = grid
                self.assertTrue(x <= gx and y <= gy and x + w >= gx + gw
                                and y + h >= gy + gh, f"screen {(x, y, w, h)}")
                found = detect_mcdu_region(image)
                self.assertIsNotNone(found, "nothing detected")
                for got, want, axis in zip(found, grid, "xywh"):
                    self.assertLessEqual(abs(got - want), 4,
                                         f"{axis}: got {found}, want about {grid}")


class _ReadsThePage:
    """A cold first read, as a new user's first frame would be."""

    truth = RADIO

    @classmethod
    def crop(cls):
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        for line in cls.truth:
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
        cls.parsed = MCDUParser(cls.crop(), columns=24, rows=14,
                                source_id=cls.__name__).parse_grid()

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

    def _cells(self, wanted):
        return [(row, col, want) for row, line in enumerate(self.truth)
                for col, want in enumerate(line) if wanted(want)]

    def test_characters_read_back(self):
        errors = [f"R{row:02d}C{col:02d} {want!r}->{self._got(row, col)!r}"
                  for row, col, want in self._cells(lambda c: c not in " ?")
                  if self._got(row, col) != want]
        self.assertEqual(errors, [])

    def test_blank_cells_stay_blank(self):
        junk = [f"R{row:02d}C{col:02d} {self._got(row, col)!r}"
                for row, col, _ in self._cells(lambda c: c == " ")
                if self._got(row, col) != " "]
        self.assertEqual(junk, [])

    def test_undrawable_symbols_come_out_blank(self):
        stray = [f"R{row:02d}C{col:02d} {self._got(row, col)!r}"
                 for row, col, _ in self._cells(lambda c: c == "?")
                 if self._got(row, col) != " "]
        self.assertEqual(stray, [])

    def test_nothing_reads_as_reverse_video(self):
        inverted = [(row, col) for row in range(14) for col in range(24)
                    if len(self.parsed[row * 24 + col]) > 3]
        self.assertEqual(inverted, [])


class _WindowRead(_ReadsThePage):
    """The live path: Auto Detect on the whole window, then read."""

    window = ""

    @classmethod
    def setUpClass(cls):
        if not (DATA / cls.window).exists():
            raise unittest.SkipTest(f"{cls.window} missing")
        cls.truth = WINDOWS[cls.window][1]
        super().setUpClass()

    @classmethod
    def crop(cls):
        image = load(DATA / cls.window)
        x, y, w, h = detect_mcdu_region(image)
        return image[y:y + h, x:x + w]


class TestRadioWindow(_WindowRead, unittest.TestCase):
    window = "ejets_window_radio.png"

    def test_solid_triangles_are_arrows(self):
        self.assertEqual(self._got(10, 0), "←")
        self.assertEqual(self._got(10, 23), "→")


class TestPerfInitWindow(_WindowRead, unittest.TestCase):
    window = "ejets_window_perf_init.png"


class TestPerfIndexWindow(_WindowRead, unittest.TestCase):
    window = "ejets_window_perf_index.png"

    def test_grey_prompts_are_grey_and_white_ones_white(self):
        wrong = []
        for row, line in enumerate(PERF_INDEX_COLOURS):
            for col, want in enumerate(line):
                cell = self.parsed[row * 24 + col]
                if want != "." and (not cell or cell[1] != want):
                    got = cell[1] if cell else "blank"
                    wrong.append(f"R{row:02d}C{col:02d} {want}->{got}")
        self.assertEqual(wrong, [])


@unittest.skipIf(not SCREEN.exists(), "E-Jets screen capture missing")
class TestRadioHandCrop(_ReadsThePage, unittest.TestCase):
    """A box drawn by hand around the text, a few pixels off the grid."""

    truth = RADIO

    @classmethod
    def crop(cls):
        return load(SCREEN)
