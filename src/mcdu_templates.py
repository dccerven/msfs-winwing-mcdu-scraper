"""
The learned-glyph store.

Recognition on a fixed-pitch display does not need an OCR engine once it
has seen the font.  Every glyph is normalised to a fixed size and kept, and
a later cell is named by finding the closest one - a hash lookup when the
shape is identical, normalised cross-correlation when it is not.  EasyOCR
is only the bootstrap: it teaches this store during warmup and is then
asked about whatever the store cannot name.

Two rules here were learned the hard way and are worth knowing before
changing anything:

* A stored label is never second-guessed.  Re-testing a template match
  against a geometric rule of thumb caps accuracy at the accuracy of that
  rule and makes learning pointless, because the correction is re-applied
  on every frame (ISSUES.md #5).

* A glyph is only stored once more than one independent reading agrees on
  it, *including* after warmup.  Committing a single unverified guess was
  once thought safe on the grounds that the character set was by then
  mostly covered; a real session grew its store from 80 to 126 templates
  while it ran and read worse with every one (ISSUES.md #25).

A profile may also ship *seed* glyphs: PNGs captured by hand from the real
display, under templates/<profile>/.  They are matched alongside the learned
store but never saved into it, so the PNGs stay the source of truth, and a
seed vetoes learning a glyph under a different label - a learned store has
been found holding a '4' filed under 'R'.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from mcdu_charset import (ARROW_LEFT, ARROW_RIGHT, BALLOT_BOX, DEGREE,
                          RENDERABLE, SUBSTITUTIONS, UNDRAWN)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Seed glyphs
# ═══════════════════════════════════════════════════════════════════════════

#: Where seed folders live.  A frozen build unpacks bundled data under
#: _MEIPASS rather than beside the source.
SEED_ROOT = (Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
             / "templates")

#: File stems for characters a filename cannot hold.  A stem that is a
#: single character names itself; a '_large' suffix marks the large font
#: and is dropped, since size is read from the row and not the glyph.
SEED_NAMES: Dict[str, str] = {
    "arrow_left": ARROW_LEFT,
    "arrow_right": ARROW_RIGHT,
    "box": BALLOT_BOX,
    "dash": "-",
    "degree": DEGREE,
    "greater_than": ">",
    "less_than": "<",
    "paren_left": "(",
    "paren_right": ")",
    "period": ".",
    "plus": "+",
    "slash": "/",
    "bracket_left": "[",
    "bracket_right": "]",
    # Not drawable, but SUBSTITUTIONS says what the CDU shows instead.
    "equals": "=",
}

#: Seed files for glyphs the CDU has nothing to show for.  They are still
#: loaded: a cell that matches one is left blank, where otherwise EasyOCR
#: put a letter there ('4' for the E-Jets swap icon) and could learn it.
UNDRAWN_NAMES = frozenset({
    "ampersand", "arrow_left_right_small", "percent", "square_filled",
    "swap_arrows",
})

#: Alpha levels a seed is binarised at.  A capture is anti-aliased and the
#: parser's threshold is not, so one cut-off gives one stroke weight; three
#: cover the weights real cells come out at.  Measured against 121 glyphs
#: learned in a real session, three cut-offs accepted 38 where one
#: accepted 26.  The one disagreement either made was that store's own
#: '4' filed as 'R', not a seed's mistake.
_SEED_THRESHOLDS = (64, 127, 192)


def seed_char(stem: str) -> Optional[str]:
    """The character a seed file names, or None if it names nothing known.

    UNDRAWN for a glyph the CDU cannot show at all.
    """
    if stem.endswith("_large"):
        stem = stem[: -len("_large")]
    if stem in UNDRAWN_NAMES:
        return UNDRAWN
    char = stem.upper() if len(stem) == 1 else SEED_NAMES.get(stem)
    return char if char in RENDERABLE or char in SUBSTITUTIONS else None


def seed_binaries(seed_dir: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield ``(char, binary)`` for every usable PNG under *seed_dir*.

    Ink is taken from the alpha channel when there is one - captures cut
    out of the display carry the glyph there over a flat colour - and from
    brightness otherwise.
    """
    for path in sorted(Path(seed_dir).rglob("*.png")):
        char = seed_char(path.stem)
        if char is None:
            logger.debug("Seed %s names no known character; skipped", path.name)
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning("Could not read seed %s", path)
            continue
        if image.ndim == 3 and image.shape[2] == 4:
            ink = image[..., 3]
        elif image.ndim == 3:
            ink = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            ink = image
        for level in _SEED_THRESHOLDS:
            yield char, (ink > level).astype(np.uint8) * 255


# ═══════════════════════════════════════════════════════════════════════════
#  Template Matcher
# ═══════════════════════════════════════════════════════════════════════════

class TemplateMatcher:
    """
    Template-based character recognition for MCDU fixed-font displays.

    Learns character patterns from confirmed OCR/contour results and then
    uses hash + normalised cross-correlation matching for instant
    recognition on subsequent frames.  CPU-only — no GPU required.
    """

    NORM_SIZE = (20, 28)       # (width, height) of normalised glyph
    MATCH_THRESHOLD = 0.85     # min NCC score to accept
    FALLBACK_THRESHOLD = 0.70  # ...and to accept as a last resort

    #: An undrawn glyph is also accepted from FALLBACK_THRESHOLD up, when it
    #: beats every drawable template by this much.  The E-Jets swap icons
    #: score only 0.77-0.83 against their own seeds, and EasyOCR then named
    #: them '4' and 'M'; but they beat every letter by 0.34 or more, where
    #: no real character on any of the eight captures comes within 0.10 of
    #: preferring an undrawn seed.
    UNDRAWN_MARGIN = 0.25
    MAX_TEMPLATES = 5          # max variants stored per character
    CONSENSUS_MIN = 2          # min votes to promote a candidate template
    LATE_CONSENSUS_MIN = 3     # ...and after warmup, when evidence is thinner

    #: Bumped whenever the normalisation changes shape-compatibility, or
    #: whenever what a stored label means changes.  Templates saved under an
    #: older format are discarded on load rather than silently compared
    #: against glyphs normalised a different way.
    #:
    #: Version 4 is the second kind.  Stores written before it were built by
    #: a learner that relabelled glyphs on the way in and, once warmup was
    #: over, committed a single unverified guess straight to disk - a real
    #: session grew one from 80 to 126 templates while it ran, and read
    #: worse with every one.  Those labels cannot be trusted, so every store
    #: is rebuilt on first run rather than asking anyone to find and delete
    #: the file.
    FORMAT_VERSION = 4

    #: Where learned glyphs are persisted when no explicit path is given.
    DEFAULT_TEMPLATE_PATH = (
        Path(__file__).resolve().parent.parent / "templates" / "mcdu_templates.npz"
    )

    def __init__(self, template_path: Optional[Path] = None,
                 seed_dir: Optional[Path] = None) -> None:
        """
        Args:
            template_path: Where to load/save learned glyphs.  Defaults to
                ``DEFAULT_TEMPLATE_PATH``.  Tests must pass a temp path —
                otherwise they inherit whatever the user learned by running
                the app, and their results depend on the host machine.
            seed_dir: Folder of hand-captured glyph PNGs for this profile,
                or None.  Matched alongside the learned store, never saved.
        """
        self._hash_cache: Dict[bytes, str] = {}
        self._templates: Dict[str, List[np.ndarray]] = {}
        self._seeds: Dict[str, List[np.ndarray]] = {}
        self._bank_key: Optional[tuple] = None   # see _scores
        self._candidates: Dict[bytes, Dict[str, int]] = {}
        self._dirty = False
        self._warmup_complete = False
        self._template_path = Path(
            template_path if template_path is not None
            else self.DEFAULT_TEMPLATE_PATH
        )
        self._load()
        if seed_dir is not None:
            self._load_seeds(Path(seed_dir))

    def _all_templates(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Every learned template, then every seed."""
        for store in (self._templates, self._seeds):
            for char, templates in store.items():
                for tmpl in templates:
                    yield char, tmpl

    def _scores(self, norm: np.ndarray) -> Tuple[List[str], np.ndarray, int]:
        """NCC of *norm* against every template, as one matrix product.

        Gives exactly what :meth:`_ncc` gives pair by pair, which cost
        ~30 us a template - fine for a hundred, not for the several hundred
        a seeded store holds, on every cell the hash cache misses.

        Returns:
            ``(labels, scores, n_learned)`` in :meth:`_all_templates` order;
            the first *n_learned* entries are learned, the rest seeds.
        """
        key = (tuple((c, len(v)) for c, v in self._templates.items()),
               tuple((c, len(v)) for c, v in self._seeds.items()))
        if key != self._bank_key:
            shape = (self.NORM_SIZE[1], self.NORM_SIZE[0])
            labels, rows, flat = [], [], []
            for char, tmpl in self._all_templates():
                row = np.zeros(shape[0] * shape[1], dtype=np.float32)
                is_flat = False
                # A template of another shape scores 0, as in _ncc.
                if tmpl.shape == shape:
                    centred = tmpl.ravel().astype(np.float32)
                    centred -= centred.mean()
                    length = float(np.linalg.norm(centred))
                    if length > self._FLAT:
                        row = centred / length
                    else:
                        is_flat = True   # blank or solid: _ncc's zero-std case
                labels.append(char)
                rows.append(row)
                flat.append(is_flat)
            bank = (np.stack(rows) if rows
                    else np.zeros((0, shape[0] * shape[1]), np.float32))
            self._bank = (labels, bank, np.array(flat, dtype=bool),
                          self.template_count)
            self._bank_key = key

        labels, bank, flat, n_learned = self._bank
        query = norm.ravel().astype(np.float32)
        query -= query.mean()
        length = float(np.linalg.norm(query))
        if length <= self._FLAT:
            return labels, flat.astype(np.float32), n_learned
        return labels, bank @ (query / length), n_learned

    #: Below this vector length a glyph has no variance to correlate.
    #: Matches _ncc's std < 1e-6 over NORM_SIZE pixels.
    _FLAT = 1e-6 * (NORM_SIZE[0] * NORM_SIZE[1]) ** 0.5

    def _best_seed(self, norm: np.ndarray) -> Optional[Tuple[str, float]]:
        labels, scores, n_learned = self._scores(norm)
        if len(labels) == n_learned:
            return None
        i = n_learned + int(np.argmax(scores[n_learned:]))
        return labels[i], float(scores[i])

    # ----- recognition ---------------------------------------------------

    def recognize(self, cell_binary: np.ndarray) -> Optional[Tuple[str, float]]:
        """Return ``(char, confidence)`` or ``None``."""
        glyph = self._extract_glyph(cell_binary)
        if glyph is None:
            return None

        norm = self._normalize(glyph)
        key = norm.tobytes()

        # Fast: exact hash
        if key in self._hash_cache:
            return (self._hash_cache[key], 1.0)

        # Slower: NCC against all templates
        best_char: Optional[str] = None
        best_score = 0.0
        labels, scores, _ = self._scores(norm)
        if labels:
            i = int(np.argmax(scores))
            if scores[i] > best_score:
                best_score, best_char = float(scores[i]), labels[i]

        if (best_char == UNDRAWN and best_score >= self.FALLBACK_THRESHOLD
                and best_score < self.MATCH_THRESHOLD):
            drawable = max((float(s) for c, s in zip(labels, scores)
                            if c != UNDRAWN), default=0.0)
            if best_score - drawable >= self.UNDRAWN_MARGIN:
                self._hash_cache[key] = UNDRAWN
                return (UNDRAWN, best_score)

        if best_score >= self.MATCH_THRESHOLD and best_char is not None:
            # No geometry heuristic here.  A template match is evidence from
            # the pixels of a glyph already confirmed by consensus; running
            # _disambiguate_confusables over it would let a rule of thumb
            # overrule that, and it would disagree with the hash-cache path
            # above, which returns the stored label untouched.
            self._hash_cache[key] = best_char
            return (best_char, best_score)

        return None

    def best_match(self, cell_binary: np.ndarray,
                   exclude: Optional[frozenset] = None,
                   ) -> Optional[Tuple[str, float]]:
        """The closest template regardless of MATCH_THRESHOLD.

        Used only as a last resort, for a cell that plainly holds ink
        but that neither engine would name.  Leaving such a cell blank
        throws away what the store already knows about the glyph, and a
        hole in the middle of a word reads worse on the CDU than a
        plausible letter does.  Never learned from.

        Args:
            cell_binary: The preprocessed cell.
            exclude: Characters this cell cannot hold.  The caller knows
                that when geometry has already ruled a shape out - and
                the store may well hold a template mislabelled as that
                very shape, which is what put a bracket inside
                CONSUMPTION.  Skipping it leaves the next best answer
                rather than no answer.
        """
        glyph = self._extract_glyph(cell_binary)
        if glyph is None:
            return None
        norm = self._normalize(glyph)
        cached = self._hash_cache.get(norm.tobytes())
        if cached is not None and not (exclude and cached in exclude):
            return (cached, 1.0)
        best_char, best_score = None, 0.0
        labels, scores, _ = self._scores(norm)
        if exclude:
            scores = np.where([c in exclude for c in labels], -np.inf, scores)
        if labels:
            i = int(np.argmax(scores))
            if scores[i] > best_score:
                best_score, best_char = float(scores[i]), labels[i]
        if best_char is None:
            return None
        return (best_char, best_score)

    # ----- learning ------------------------------------------------------

    def learn(self, char: str, cell_binary: np.ndarray,
              confidence: float = 1.0) -> None:
        """Record a confirmed character template (consensus-based).

        During warmup (before ``_warmup_complete``), each glyph shape
        accumulates votes.  A template is only promoted once the same
        character reaches ``CONSENSUS_MIN`` votes with a clear majority.
        After warmup, new characters are accepted directly (the bulk of
        the character set is already safely templated).
        """
        if confidence < 0.60 or not char or not char.strip():
            return
        char = char.upper()
        if len(char) != 1:
            return

        glyph = self._extract_glyph(cell_binary)
        if glyph is None:
            return

        norm = self._normalize(glyph)
        key = norm.tobytes()

        # Already known — skip
        if key in self._hash_cache:
            return

        # A seed was captured and named by hand.  A glyph that plainly is
        # one seed must not be learned as something else: a learned store
        # has been found holding a '4' labelled 'R', and this stops that.
        seed = self._best_seed(norm)
        if seed and seed[0] != char and seed[1] >= self.MATCH_THRESHOLD:
            logger.debug("Not learning '%s': it matches seed '%s' (%.2f)",
                         char, seed[0], seed[1])
            return

        # ---- candidate voting ----
        #
        # This applies after warmup as well.  It used to commit a glyph
        # straight to the store once warmup was over, on the reasoning that
        # the character set was by then mostly covered - but that is exactly
        # backwards.  A warmup vote is one of fifteen passes over the same
        # pixels; a later one is a single unverified guess, so it needs more
        # corroboration, not less.  A real session showed the cost: the
        # store grew 80 -> 90 -> 122 -> 126 templates while it ran, and the
        # display got steadily worse as wrong glyphs were matched and
        # re-matched from the cache.  Templates are permanent, so an
        # unconfirmed one is not a small mistake.
        if key not in self._candidates:
            self._candidates[key] = {}
        votes = self._candidates[key]
        votes[char] = votes.get(char, 0) + 1

        needed = (self.CONSENSUS_MIN if not self._warmup_complete
                  else self.LATE_CONSENSUS_MIN)
        best_char = max(votes, key=votes.get)
        total = sum(votes.values())
        if (votes[best_char] >= needed
                and votes[best_char] > total * 0.6):
            self._commit_template(best_char, norm)
            del self._candidates[key]
            logger.debug(
                "Promoted '%s' (%d/%d votes, %d total templates)",
                best_char, votes[best_char], total, self.template_count,
            )

    def _commit_template(self, char: str, norm: np.ndarray) -> None:
        """Directly commit a normalised glyph as a template."""
        key = norm.tobytes()
        self._hash_cache[key] = char

        if char not in self._templates:
            self._templates[char] = []

        for existing in self._templates[char]:
            if self._ncc(norm, existing) > 0.95:
                return

        if len(self._templates[char]) >= self.MAX_TEMPLATES:
            return

        self._templates[char].append(norm)
        self._dirty = True

    # ----- persistence ---------------------------------------------------

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self._template_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"__format__": np.array([self.FORMAT_VERSION])}
            for char, templates in self._templates.items():
                # Use hex-encoded UTF-8 to handle multi-character keys
                # (e.g. "<>" from OCR) as well as single characters.
                hex_key = char.encode("utf-8").hex()
                for i, tmpl in enumerate(templates):
                    data[f"h{hex_key}_{i}"] = tmpl
            np.savez_compressed(str(self._template_path), **data)
            self._dirty = False
            logger.info(
                "Saved %d templates for %d characters → %s",
                self.template_count, len(self._templates), self._template_path,
            )
        except Exception as exc:
            logger.warning("Failed to save templates: %s", exc)

    def _load(self) -> None:
        if not self._template_path.exists():
            return
        try:
            data = np.load(str(self._template_path))

            stored_format = int(data["__format__"][0]) if "__format__" in data.files else 1
            if stored_format != self.FORMAT_VERSION:
                logger.warning(
                    "Discarding templates from %s: saved in format v%d, this "
                    "build normalises glyphs differently (v%d). They will be "
                    "relearned on the next warmup.",
                    self._template_path, stored_format, self.FORMAT_VERSION,
                )
                return

            for key in data.files:
                if key == "__format__":
                    continue
                prefix = key.split("_")[0]
                if prefix.startswith("h"):
                    # New format: hex-encoded UTF-8
                    char = bytes.fromhex(prefix[1:]).decode("utf-8")
                else:
                    # Legacy format: cXXXX (single Unicode codepoint)
                    char = chr(int(prefix[1:], 16))
                if char not in self._templates:
                    self._templates[char] = []
                self._templates[char].append(data[key])
            if self.template_count > 0:
                self._warmup_complete = True
            logger.info(
                "Loaded %d templates for %d characters",
                self.template_count, len(self._templates),
            )
        except Exception as exc:
            logger.warning("Failed to load templates: %s", exc)

    def _load_seeds(self, seed_dir: Path) -> None:
        if not seed_dir.is_dir():
            logger.warning("Seed folder %s not found; no seeds loaded", seed_dir)
            return
        for char, binary in seed_binaries(seed_dir):
            glyph = self._extract_glyph(binary)
            if glyph is None:
                continue
            norm = self._normalize(glyph)
            variants = self._seeds.setdefault(char, [])
            # Thresholds that agree give the same shape; keep one of each.
            if all(self._ncc(norm, v) <= 0.95 for v in variants):
                variants.append(norm)
        logger.info("Loaded %d seed glyphs for %d characters from %s",
                    self.seed_count, len(self._seeds), seed_dir)

    # ----- helpers -------------------------------------------------------

    def _normalize(self, glyph: np.ndarray) -> np.ndarray:
        """Scale a glyph into NORM_SIZE, preserving its aspect ratio.

        Stretching each glyph to fill the box destroyed the one feature that
        separates the thin symbols: a dash, a period, an underscore and a
        solid block all became the same all-white rectangle and matched each
        other with NCC 1.0.  Scaling by the smaller factor and centring the
        result on a blank canvas keeps them distinct.
        """
        target_w, target_h = self.NORM_SIZE
        h, w = glyph.shape[:2]
        if w < 1 or h < 1:
            return np.zeros((target_h, target_w), dtype=np.uint8)

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        _, binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def _extract_glyph(binary: np.ndarray) -> Optional[np.ndarray]:
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        # A hyphen renders as few as 6x1 px in a 20x24 cell.  Requiring 2px
        # in both axes silently discarded every dash on the display.
        if w < 1 or h < 1:
            return None
        # 1-px padding so glyphs at slightly different positions within
        # the cell normalise to a consistent shape after resize.
        H, W = binary.shape
        pad = 1
        return binary[max(0, y - pad) : min(H, y + h + pad),
                      max(0, x - pad) : min(W, x + w + pad)]

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        """Normalised cross-correlation (same-size images)."""
        if a.shape != b.shape:
            return 0.0
        af = a.ravel().astype(np.float32)
        bf = b.ravel().astype(np.float32)
        am, bm = af.mean(), bf.mean()
        astd, bstd = af.std(), bf.std()
        if astd < 1e-6 or bstd < 1e-6:
            return 1.0 if (astd < 1e-6 and bstd < 1e-6) else 0.0
        return float(np.dot(af - am, bf - bm) / (len(af) * astd * bstd))

    @property
    def template_count(self) -> int:
        return sum(len(v) for v in self._templates.values())

    @property
    def seed_count(self) -> int:
        return sum(len(v) for v in self._seeds.values())

    def reset(self) -> None:
        """Wipe all learned templates and candidates.  Seeds are kept: they
        come from files, not from learning."""
        self._hash_cache.clear()
        self._templates.clear()
        self._candidates.clear()
        self._warmup_complete = False
        self._dirty = False
        logger.info("Template matcher reset — all templates cleared")


# Active matcher.  One store is live at a time: glyphs learned from one
# font must never be matched against another, so switching aircraft profile
# switches the store (see set_template_store).
_template_matcher: Optional[TemplateMatcher] = None
_template_store_path: Optional[Path] = None
_template_seed_dir: Optional[Path] = None


def _get_template_matcher() -> TemplateMatcher:
    global _template_matcher
    if _template_matcher is None:
        _template_matcher = TemplateMatcher(template_path=_template_store_path,
                                            seed_dir=_template_seed_dir)
    return _template_matcher


def set_store(path, seed_dir=None) -> bool:
    """Make *path* the active learned-glyph store.

    Saves the outgoing store first.  Glyphs learned from one font must never
    be matched against another, so each aircraft profile gets its own file
    and switching profile switches the store.

    Args:
        path: The learned-glyph store.
        seed_dir: The profile's hand-captured seed glyphs, or None.

    Returns:
        True if the active store changed.  The caller uses this to discard
        anything derived from the old one - the parser's row-level OCR cache
        holds readings produced by the old store's glyphs, and serving those
        after a switch would show the previous aircraft's font decisions.
    """
    global _template_matcher, _template_store_path, _template_seed_dir
    path = Path(path)
    seed_dir = Path(seed_dir) if seed_dir is not None else None
    if (_template_matcher is not None and _template_store_path == path
            and _template_seed_dir == seed_dir):
        return False
    if _template_matcher is not None:
        _template_matcher.save()
    _template_store_path = path
    _template_seed_dir = seed_dir
    _template_matcher = TemplateMatcher(template_path=path, seed_dir=seed_dir)
    logger.info("Template store switched to %s (%d templates, %d seeds)",
                path.name, _template_matcher.template_count,
                _template_matcher.seed_count)
    return True
