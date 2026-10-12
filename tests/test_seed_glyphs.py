"""
Seed glyphs: hand-captured PNGs matched alongside the learned store.

What matters is that seeds name their own glyphs, never leak into the saved
store, survive a reset, and stop the learner filing a glyph under the wrong
label - a learned store has been found holding a '4' labelled 'R'.
"""

import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import mcdu_templates
from aircraft_profiles import PROFILES
from mcdu_charset import DIGITS, LETTERS, RENDERABLE, SUBSTITUTIONS, UNDRAWN
from mcdu_templates import SEED_ROOT, TemplateMatcher, seed_binaries, seed_char

EJETS = SEED_ROOT / "ejets"

#: Seed files for glyphs the CDU cannot draw: matched so they read as blank.
UNDRAWN_FILES = {"ampersand", "ampersand_large", "percent", "square_filled",
                 "swap_arrows", "arrow_left_right_small"}


def _png_binary(name):
    """The seed's ink at the middle threshold, as the loader reads it."""
    path = next(EJETS.rglob(f"{name}.png"))
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3 and image.shape[2] == 4:
        ink = image[..., 3]
    elif image.ndim == 3:
        ink = image.max(axis=2)
    else:
        ink = image
    return (ink > 127).astype(np.uint8) * 255


class TestSeedNames(unittest.TestCase):

    def test_names(self):
        self.assertEqual(seed_char("A"), "A")
        self.assertEqual(seed_char("A_large"), "A")
        self.assertEqual(seed_char("7_large"), "7")
        self.assertEqual(seed_char("degree"), "°")
        self.assertEqual(seed_char("paren_left_large"), "(")
        self.assertEqual(seed_char("ampersand"), UNDRAWN)
        self.assertEqual(seed_char("swap_arrows"), UNDRAWN)
        # Not drawable, but the CDU shows '-' for it (SUBSTITUTIONS).
        self.assertEqual(seed_char("equals"), "=")
        self.assertIsNone(seed_char("no_such_symbol"))

    @unittest.skipIf(not EJETS.is_dir(), "E-Jets seeds missing")
    def test_every_ejets_file_is_understood(self):
        """A misspelt file name would be skipped without a word."""
        for path in EJETS.rglob("*.png"):
            char = seed_char(path.stem)
            if path.stem in UNDRAWN_FILES:
                self.assertEqual(char, UNDRAWN, path.name)
            else:
                self.assertTrue(char in RENDERABLE or char in SUBSTITUTIONS,
                                f"{path.name} -> {char!r}")

    def test_alpha_channel_carries_the_ink(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = np.full((30, 20, 4), 255, dtype=np.uint8)
            image[..., 3] = 0
            image[5:25, 8:12, 3] = 255
            cv2.imwrite(str(Path(tmp) / "I.png"), image)
            char, binary = next(seed_binaries(Path(tmp)))
            self.assertEqual(char, "I")
            self.assertEqual(int(binary[15, 10]), 255)
            self.assertEqual(int(binary[15, 2]), 0)


@unittest.skipIf(not EJETS.is_dir(), "E-Jets seeds missing")
class TestSeededMatcher(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "store.npz"
        self.matcher = TemplateMatcher(self.path, seed_dir=EJETS)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_covers_letters_and_digits(self):
        missing = set(LETTERS + DIGITS) - set(self.matcher._seeds)
        self.assertFalse(missing, f"no seed for {sorted(missing)}")

    def test_seeds_name_their_own_glyphs(self):
        for path in EJETS.rglob("*.png"):
            char = seed_char(path.stem)
            if char is None:
                continue
            result = self.matcher.recognize(_png_binary(path.stem))
            self.assertIsNotNone(result, path.name)
            self.assertEqual(result[0], char, path.name)

    def test_seeds_are_not_saved(self):
        self.assertEqual(self.matcher.template_count, 0)
        self.matcher.save()
        self.assertFalse(self.path.exists(),
                         "seeds alone made the store dirty")

    def test_reset_keeps_seeds(self):
        seeds = self.matcher.seed_count
        self.matcher.reset()
        self.assertEqual(self.matcher.seed_count, seeds)

    def test_seed_vetoes_a_conflicting_label(self):
        four = _png_binary("4")
        for _ in range(5):
            self.matcher.learn("R", four, confidence=1.0)
        self.assertNotIn("R", self.matcher._templates)

    def test_learning_still_works_for_unseeded_glyphs(self):
        plus = np.zeros((28, 20), dtype=np.uint8)
        plus[13:15, 4:16] = 255
        plus[8:20, 9:11] = 255
        self.matcher.learn("+", plus, confidence=1.0)
        self.matcher.learn("+", plus, confidence=1.0)
        self.assertEqual(self.matcher.template_count, 1)


class TestProfileSeeds(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._saved = (mcdu_templates._template_matcher,
                       mcdu_templates._template_store_path,
                       mcdu_templates._template_seed_dir)

    def tearDown(self):
        (mcdu_templates._template_matcher,
         mcdu_templates._template_store_path,
         mcdu_templates._template_seed_dir) = self._saved
        self._tmpdir.cleanup()

    def test_only_ejets_has_seeds(self):
        self.assertEqual(PROFILES["ejets"].seed_dir(), EJETS)
        for profile in PROFILES.values():
            if profile.id != "ejets":
                self.assertIsNone(profile.seed_dir(), profile.id)

    @unittest.skipIf(not EJETS.is_dir(), "E-Jets seeds missing")
    def test_store_switch_carries_seeds(self):
        store = Path(self._tmpdir.name) / "ejets.npz"
        self.assertTrue(mcdu_templates.set_store(store))
        self.assertEqual(mcdu_templates._get_template_matcher().seed_count, 0)
        self.assertTrue(mcdu_templates.set_store(store, EJETS),
                        "adding seeds to the same store was a no-op")
        self.assertGreater(mcdu_templates._get_template_matcher().seed_count, 0)
        self.assertFalse(mcdu_templates.set_store(store, EJETS))


if __name__ == "__main__":
    unittest.main()
