"""
SPAD.next WebSocket client: drives a WinWing CDU through SPAD.next instead of
MobiFlight.

SPAD.next runs its own SimBridge-style relay and reads CDU screens from it in
the format dementedmonkey's CJ4/737 MCDU mod uses
(https://github.com/dementedmonkey/cj4-mcdu).  This client speaks that
protocol exactly as the mod's 737 sender (dm_cdu.js) does:

* connect to ``ws://localhost:8088`` and announce itself with
  ``mcduConnected``;
* send the screen as ``update:wt737:{json}`` whenever it changes;
* resend it whenever anyone on the relay asks with ``requestUpdate``.

The 737 format is the one that fits a scraped grid: it is the Working Title
SimpleFmcRenderer's own cell data - 14 rows of 24 ``{content, styles}``
cells, ``styles`` being CSS class names such as ``"green s-text"`` - so
every character keeps its own colour and size with no layout to guess.

Each CDU has its own socket and sends only its own side (``left`` for the
captain, ``right`` for the first officer), as each CDU instance in the sim
does.  The relay echoes every message to every client, our own included, so
anything that is not ``requestUpdate`` is ignored.
"""

import json
import logging
from typing import Optional

from mcdu_charset import BALLOT_BOX
from mobiflight_client import MobiFlightClient, sanitise_display_data

logger = logging.getLogger(__name__)

HARDWARE_COLUMNS = 24
HARDWARE_ROWS = 14

#: Grid colour code -> Working Title FMC colour class.  'e' (grey) is the
#: renderer's ``disabled`` style; khaki has no class of its own.
COLOUR_CLASSES = {
    "a": "amber",
    "c": "cyan",
    "e": "disabled",
    "g": "green",
    "k": "yellow",
    "m": "magenta",
    "o": "blue",
    "r": "red",
    "w": "white",
    "y": "yellow",
}

#: Glyphs the Working Title FMCs spell differently: their empty-field box
#: is U+25A1, not the ballot box the parser reads it as.
WT_GLYPHS = {
    BALLOT_BOX: "□",
}

#: An empty cell, as the renderer initialises one.
_BLANK_CELL = {"content": " ", "styles": ""}


def _row_style(row: int) -> str:
    """The class the renderer gives a row: alternating large and small."""
    return "fmc-row s-text" if row % 2 else "fmc-row d-text"


def _encode_cell(cell: list) -> dict:
    """One ``[char, colour, size(, inverted)]`` cell as ``{content, styles}``."""
    if not cell or cell[0] == " ":
        return dict(_BLANK_CELL)
    char = WT_GLYPHS.get(cell[0], cell[0])
    styles = [COLOUR_CLASSES.get(cell[1], "white"),
              "s-text" if cell[2] else "d-text"]
    if len(cell) > 3 and cell[3]:
        styles.append("inverse")
    return {"content": char, "styles": " ".join(styles)}


def encode_screen(display_data: list, font: Optional[str] = None) -> dict:
    """Turn a 24x14 hardware grid into one side of a ``wt737`` update.

    Every cell names its size explicitly, so the row's own default (which
    alternates large/small, as on the real CDU) never decides it.

    Args:
        display_data: 336 cells, each ``[]`` or ``[char, colour, size]``
            with an optional fourth reverse-video flag.
        font: The hardware font, for character sanitising.
    """
    cells = sanitise_display_data(display_data, font)
    lines = []
    for row in range(HARDWARE_ROWS):
        start = row * HARDWARE_COLUMNS
        cols = [_encode_cell(cell)
                for cell in cells[start:start + HARDWARE_COLUMNS]]
        cols += [dict(_BLANK_CELL)
                 for _ in range(HARDWARE_COLUMNS - len(cols))]
        lines.append({"rowStyle": _row_style(row), "cols": cols})
    return {"lines": lines, "exec": False, "power": True}


class SpadClient(MobiFlightClient):
    """Sends one CDU's screen to SPAD.next, reconnecting as MobiFlight's does."""

    SERVICE = "SPAD.next"

    def __init__(self, websocket_uri: str, side: str = "left",
                 message_type: str = "wt737", aircraft: str = "",
                 font: Optional[str] = None, max_retries: int = 3):
        """
        Args:
            websocket_uri: SPAD.next's relay, e.g. ws://localhost:8088.
            side: ``left`` (captain) or ``right`` (first officer).
            message_type: The aircraft type tag in ``update:<type>:``.
            aircraft: Sent as the ``aircraft`` field, where the mod sends
                the sim's ATC MODEL.
            font: Hardware font, only used to sanitise characters - SPAD.next
                loads the font itself.
            max_retries: See MobiFlightClient.
        """
        super().__init__(websocket_uri, font=font, max_retries=max_retries)
        self.side = side
        self.message_type = message_type
        self.aircraft = aircraft
        #: The last update built, resent on reconnect and on requestUpdate.
        self._last_update: Optional[str] = None

    async def _on_open(self):
        """Announce ourselves as a sim CDU and restore the current screen."""
        await self.websocket.send("mcduConnected")
        if self._last_update is not None:
            await self.websocket.send(self._last_update)

    async def _on_message(self, message) -> None:
        if message == "requestUpdate" and self._last_update is not None:
            await self.send(self._last_update)

    def build_update(self, display_data: list) -> str:
        """The ``update:`` message for a grid, without sending it."""
        payload = {"aircraft": self.aircraft,
                   self.side: encode_screen(display_data, self.font)}
        return f"update:{self.message_type}:{json.dumps(payload)}"

    async def send_display_data(self, display_data: list) -> bool:
        """Send a grid to SPAD.next.  Returns True if it was sent."""
        self._last_update = self.build_update(display_data)
        return await self.send(self._last_update)
