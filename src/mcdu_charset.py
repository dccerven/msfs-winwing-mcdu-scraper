"""
Single source of truth for the MCDU character set.

Three separate and disagreeing definitions used to exist: a validity set and
an EasyOCR allowlist in the parser, and a renderable set in the WebSocket
client.  They differed on '+', '*', ':' and the degree sign, so the parser
could accept characters the display cannot draw, and could forbid EasyOCR
from proposing ones it can.

The hardware is the authority.  RENDERABLE is what the CDU font can draw;
everything else is derived from it, so widening the display's repertoire is a
one-line change here rather than three edits that can drift apart.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional

# ---------------------------------------------------------------------------
#  What the WinWing CDU's AirbusThales font is expected to draw.
#
#  Checked against MobiFlight's own source.  There is no allowlist on their
#  side: WinCtrlCduController.ConvertAndSendCduData does
#
#      byteList.AddRange(Encoding.UTF8.GetBytes(new char[] { currentChar }));
#
#  so whatever character it receives is UTF-8 encoded and forwarded to the
#  device unvalidated, and the reference headwind_a33_winwing_cdu.py script
#  filters nothing either -- its REPLACED_CHARS table only translates the
#  conventions its data source uses (for example '_' for an entry box) into
#  the matching glyphs.
#
#  So an earlier warning here that an unsupported character "can freeze the
#  display" has no basis in the implementation.  The real limit is which
#  glyphs the font file carries, and those .dat files are encrypted, so this
#  set stays the conservative one that reference integrations are known to
#  emit.  Widening it is a question of what the font has, not of protocol
#  safety.
# ---------------------------------------------------------------------------
BALLOT_BOX = "☐"   # small square marking a selectable field
ARROW_LEFT = "←"
ARROW_RIGHT = "→"
ARROW_UP = "↑"
ARROW_DOWN = "↓"
DELTA = "Δ"        # overfly marker
DEGREE = "°"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
# '+' is here on two pieces of evidence: a real A330 capture showing
# +0.0/+0.0 on the IDLE/PERF line, and MobiFlight passing every character
# through unfiltered (see above), exactly as the reference A330 script does.
# Mapping it to a space lost the sign; mapping it to '-' would invert it.
PUNCTUATION = " .-+/<>[]()" + DEGREE

RENDERABLE: FrozenSet[str] = frozenset(
    LETTERS + DIGITS + PUNCTUATION
    + BALLOT_BOX + ARROW_LEFT + ARROW_RIGHT + ARROW_UP + ARROW_DOWN + DELTA
)

#: A glyph the display draws and the CDU cannot: the E-Jets' swap icons, the
#: diamond beside STBY and the filled square after IDENT.  Read as a letter
#: they put a stray '4' or 'O' on the hardware, so the recognisers name them
#: with this instead, and the parser leaves the cell blank.  Private-use, so
#: no OCR engine or font produces it, and sanitise_char turns any that leak
#: into a space.
UNDRAWN = ""

# ---------------------------------------------------------------------------
#  What EasyOCR is permitted to output.
# ---------------------------------------------------------------------------
#  Restricted to the ASCII subset of RENDERABLE.  A CRNN cannot reliably
#  produce the box-drawing and arrow glyphs, which reach the display through
#  contour detection and the '<'/'>' substitutions instead.  Letting the
#  engine propose characters the CDU cannot draw only invites a wrong guess
#  where a right one was available.
OCR_ALLOWLIST: str = "".join(
    sorted(c for c in RENDERABLE if c.isascii() and c != " ")
) + " "

# ---------------------------------------------------------------------------
#  Substitutions applied on the way to the display.
# ---------------------------------------------------------------------------
#  Anything neither in this table nor in RENDERABLE becomes a space.
SUBSTITUTIONS: Dict[str, str] = {
    "(": "[",
    ")": "]",
    "*": ".",
    ":": ".",
    "_": "-",
    "~": "-",
    "=": "-",
    "|": "1",
    "\\": "/",
    ",": ".",
    ";": ".",
    "{": "[",
    "}": "]",
    # '<' and '>' are deliberately absent.  They used to be folded onto the
    # arrows, but the MCDU draws both, and on the same page: the real capture
    # has a true left arrow (shaft plus solid head) at R06C00 and a plain
    # chevron at the end of STATUS/XLOAD> on R12.  Rewriting one into the
    # other changes what the CDU shows.
    #
    # MobiFlight's reference headwind_a33_winwing_cdu.py agrees: it maps
    # '{' and '}' onto the arrows and lets '<' and '>' through untouched.
    # Both are in PUNCTUATION, so they pass through here as themselves.
}


# ---------------------------------------------------------------------------
#  What a particular hardware font draws beyond RENDERABLE.
# ---------------------------------------------------------------------------
#  Checked on a WinWing CDU by sending each candidate straight to the device
#  and looking.  Only listed glyphs are trusted: everything else in the test
#  - { } _ ~ | \ ^ `, lower case, and the arrow and box symbols beyond the
#  ones in RENDERABLE - came out blank or garbled.  A font not listed here
#  has not been tested, and keeps the conservative set and substitutions.
FONT_EXTRAS: Dict[str, FrozenSet[str]] = {
    # Tested 2026-10-12.  The E-Jets shows RNP=1.00, (GAUGE) and TO CG %,
    # which reached the CDU as RNP-1.00, [GAUGE] and a blank.
    "Boeing": frozenset("()*:,;%&#@!?'\"$="),
}

#: Every character some font can draw - what a recogniser may name.
DRAWABLE: FrozenSet[str] = RENDERABLE.union(*FONT_EXTRAS.values())


def sanitise_char(char: str, font: Optional[str] = None) -> str:
    """Map a single character onto something the CDU can render.

    Args:
        char: The character read off the screen.
        font: The hardware font loaded, if known.  Characters it is known
            to draw (FONT_EXTRAS) are sent as they are.
    """
    if not char:
        return " "
    if len(char) > 1:
        # Multi-character strings can arrive from the contour fallback.
        char = char[0]
    if font and char in FONT_EXTRAS.get(font, ()):
        return char
    if char in SUBSTITUTIONS:
        return SUBSTITUTIONS[char]
    if char in RENDERABLE:
        return char
    upper = char.upper()
    if upper in SUBSTITUTIONS:
        return SUBSTITUTIONS[upper]
    return upper if upper in RENDERABLE else " "
