"""
The SPAD.next output: dementedmonkey's CJ4/737 MCDU mod protocol.

The mod's 737 sender (dm_cdu.js) is the reference: it announces itself with
mcduConnected, sends update:wt737:{json} carrying the Working Title
renderer's 14x24 {content, styles} cells for its own side, and resends on
requestUpdate.  SPAD.next's relay echoes every message to every client.
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mcdu_charset import BALLOT_BOX
from spad_client import SpadClient, encode_screen

CELLS = 24 * 14


def grid(**cells):
    """A blank 336-cell grid with cells set by index, e.g. c0=['A','w',0]."""
    data = [[] for _ in range(CELLS)]
    for key, cell in cells.items():
        data[int(key[1:])] = cell
    return data


class TestEncodeScreen(unittest.TestCase):

    def test_shape_is_fourteen_rows_of_twenty_four(self):
        screen = encode_screen(grid())
        self.assertEqual(len(screen["lines"]), 14)
        for line in screen["lines"]:
            self.assertEqual(len(line["cols"]), 24)

    def test_screen_is_powered_and_exec_off(self):
        screen = encode_screen(grid())
        self.assertTrue(screen["power"])
        self.assertFalse(screen["exec"])

    def test_blank_cell_matches_the_renderers_initial_cell(self):
        cell = encode_screen(grid())["lines"][0]["cols"][0]
        self.assertEqual(cell, {"content": " ", "styles": ""})

    def test_cell_lands_at_its_row_and_column(self):
        screen = encode_screen(grid(c49=["K", "g", 0]))   # row 2, col 1
        self.assertEqual(screen["lines"][2]["cols"][1]["content"], "K")

    def test_colour_and_size_become_classes(self):
        screen = encode_screen(grid(c0=["A", "c", 1], c1=["B", "m", 0]))
        cols = screen["lines"][0]["cols"]
        self.assertEqual(cols[0]["styles"], "cyan s-text")
        self.assertEqual(cols[1]["styles"], "magenta d-text")

    def test_every_parser_colour_has_a_class(self):
        expected = {"w": "white", "c": "cyan", "g": "green", "y": "yellow",
                    "a": "amber", "r": "red", "m": "magenta",
                    "e": "disabled"}
        for code, name in expected.items():
            cell = encode_screen(grid(c0=["A", code, 0]))["lines"][0]["cols"][0]
            self.assertEqual(cell["styles"].split()[0], name, code)

    def test_reverse_video_is_kept(self):
        cell = encode_screen(grid(c0=["A", "w", 0, True]))["lines"][0]["cols"][0]
        self.assertIn("inverse", cell["styles"].split())

    def test_row_styles_alternate_like_the_renderer(self):
        lines = encode_screen(grid())["lines"]
        self.assertEqual(lines[0]["rowStyle"], "fmc-row d-text")
        self.assertEqual(lines[1]["rowStyle"], "fmc-row s-text")

    def test_box_uses_the_working_title_glyph(self):
        cell = encode_screen(grid(c0=[BALLOT_BOX, "a", 0]))["lines"][0]["cols"][0]
        self.assertEqual(cell["content"], "□")

    def test_characters_are_sanitised(self):
        cell = encode_screen(grid(c0=["é", "w", 0]))["lines"][0]["cols"][0]
        self.assertEqual(cell["content"], " ")


class TestBuildUpdate(unittest.TestCase):

    def test_message_prefix_and_side(self):
        client = SpadClient("ws://localhost:1", side="right", aircraft="E175")
        message = client.build_update(grid(c0=["A", "w", 0]))
        self.assertTrue(message.startswith("update:wt737:"))
        payload = json.loads(message[len("update:wt737:"):])
        self.assertEqual(payload["aircraft"], "E175")
        self.assertIn("right", payload)
        self.assertNotIn("left", payload)

    def test_message_type_is_configurable(self):
        client = SpadClient("ws://localhost:1", message_type="wt21")
        self.assertTrue(client.build_update(grid()).startswith("update:wt21:"))

    def test_no_font_message_is_ever_built(self):
        """SPAD.next loads the font itself; MobiFlight's Font target is not
        part of this protocol."""
        client = SpadClient("ws://localhost:1", font="Boeing")
        self.assertNotIn('"Target"', client.build_update(grid()))


class TestAgainstRelay(unittest.IsolatedAsyncioTestCase):
    """A local relay that echoes like SPAD.next's and the mod's server."""

    async def asyncSetUp(self):
        self.clients = set()
        self.seen = asyncio.Queue()

        async def relay(ws):
            self.clients.add(ws)
            try:
                async for message in ws:
                    await self.seen.put(message)
                    for client in list(self.clients):
                        await client.send(message)
            finally:
                self.clients.discard(ws)

        self.server = await websockets.serve(relay, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"
        self.client = SpadClient(self.url, side="left")
        self.task = asyncio.create_task(self.client.run())
        await asyncio.wait_for(self.client.connected.wait(), 5)

    async def asyncTearDown(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        await self.client.close()
        self.server.close()
        await self.server.wait_closed()

    async def next_seen(self):
        return await asyncio.wait_for(self.seen.get(), 5)

    async def test_announces_itself_on_connect(self):
        self.assertEqual(await self.next_seen(), "mcduConnected")

    async def test_sends_the_screen(self):
        await self.next_seen()                      # mcduConnected
        sent = await self.client.send_display_data(grid(c0=["A", "w", 0]))
        self.assertTrue(sent)
        message = await self.next_seen()
        payload = json.loads(message[len("update:wt737:"):])
        self.assertEqual(payload["left"]["lines"][0]["cols"][0]["content"], "A")

    async def test_answers_request_update(self):
        await self.next_seen()                      # mcduConnected
        await self.client.send_display_data(grid(c0=["A", "w", 0]))
        first = await self.next_seen()
        async with websockets.connect(self.url) as spad:
            await spad.send("requestUpdate")
            self.assertEqual(await self.next_seen(), "requestUpdate")
            self.assertEqual(await self.next_seen(), first)

    async def test_echoes_of_its_own_update_are_ignored(self):
        """The relay sends our update back to us; replying would loop."""
        await self.next_seen()                      # mcduConnected
        await self.client.send_display_data(grid(c0=["A", "w", 0]))
        await self.next_seen()
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(self.seen.get(), 1.5)


if __name__ == "__main__":
    unittest.main()
