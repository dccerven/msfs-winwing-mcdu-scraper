# Templates Directory

The parser already recognises characters by template matching. You do not
need to change `mcdu_parser.py` to use templates. This folder holds two kinds
of glyph data:

| What | Where | Made by | In git |
|---|---|---|---|
| Learned store | `mcdu_templates*.npz`, one per aircraft profile | The app, at runtime | No (`.gitignore`) |
| Seed glyphs | `<profile>/` folders of PNGs, e.g. `ejets/` | Captured by hand | Yes |

## How recognition works

For each non-empty cell, [`parse_grid`](../src/mcdu_parser.py) tries these in
order:

1. **Geometry**: entry boxes, dashes, slashes, brackets and similar shapes.
2. **Template matching** ([`TemplateMatcher`](../src/mcdu_templates.py)):
   - The cell's glyph is cropped and scaled to 20×28.
   - It is first looked up by exact hash, then compared by normalised
     cross-correlation (NCC) against every learned template and every seed.
   - A score of 0.85 or more is accepted.
3. **EasyOCR**, for rows that templates could not finish. During warmup its
   readings are voted on, and a glyph joins the learned store only once
   several readings agree.

## Learned store

- Each aircraft profile has its own `.npz` file, because glyphs from one font
  must never be matched against another font.
- The app builds it on first run and keeps adding to it.
- To rebuild it, use **Delete Templates** in the GUI, which deletes the
  current profile's file.

## Seed glyphs

Seeds are PNGs captured from the real display. They are matched alongside
the learned store but never saved into it.

- A glyph a seed matches gets a hand-checked label from the first frame,
  with no warmup needed.
- A seed also stops the learner filing a glyph under the wrong label.
  - A learned store has been found holding a `4` filed as `R`, and a broken
    fragment filed as `7`.
  - With seeds loaded, a glyph that matches the `4` seed at 0.85 or more can
    no longer be learned as anything else.

Seeds belong to a profile, not to a font family. The E-Jets and Fokker
profiles both load the hardware's "Boeing" font, but the Fokker's glyphs are
squatter and do not match the E-Jets seeds. A profile's seed folder is set by `seed_dirname` in
[`aircraft_profiles.py`](../src/aircraft_profiles.py).

### Layout

```
templates/
└── ejets/
    ├── characters/   A.png, A_large.png, …
    ├── numbers/      0.png, 0_large.png, …
    └── symbols/      degree.png, arrow_left.png, …
```

Subfolders are only for tidiness: everything under the profile folder is
loaded.

### Capturing a seed

- One character per PNG, cropped with a little margin. The size doesn't
  matter, because glyphs are scaled to 20×28 before matching.
- Ink can be in the **alpha channel** (a transparent background, which is how
  the E-Jets set was captured) or in brightness (light ink on a dark
  background).
- Take the glyph from the simulator screen at its normal size, and don't
  redraw or retouch it.

### File names

- A single-character name means that character: `A.png`, `7.png`.
- A `_large` suffix marks the large font. It is dropped when the file is
  loaded, because font size is read from the row, not the glyph.
- Characters that cannot appear in a file name use these names:

| Name | Char | Name | Char |
|---|---|---|---|
| `arrow_left` | ← | `less_than` | < |
| `arrow_right` | → | `greater_than` | > |
| `box` | ☐ | `paren_left` / `paren_right` | ( ) |
| `dash` | - | `bracket_left` / `bracket_right` | [ ] |
| `degree` | ° | `period` | . |
| `plus` | + | `slash` | / |

- Some glyphs on the display have nothing the CDU can draw for them.
  Their seeds are still loaded, so a cell that matches one is left blank
  instead of EasyOCR putting a wrong letter there. In the E-Jets set these
  are `ampersand`, `percent`, `square_filled`, `swap_arrows` and
  `arrow_left_right_small`; the list is `UNDRAWN_NAMES` in
  [`mcdu_templates.py`](../src/mcdu_templates.py).
- `equals` names `=`, which the CDU shows as `-` (see `SUBSTITUTIONS` in
  [`mcdu_charset.py`](../src/mcdu_charset.py)). The drawable set is
  `RENDERABLE` in the same file.
- `tests/test_seed_glyphs.py` fails if a file name in `ejets/` is not
  understood, so a misspelt name doesn't get skipped without anyone noticing.

### Adding seeds for another profile

1. Capture the PNGs into `templates/<name>/`.
2. Set `seed_dirname="<name>"` on that profile in `aircraft_profiles.py`.
3. Add `templates/<name>` to the `--add-data` lines in `build_exe.bat`,
   `MSFS-CDU-Scraper-GUI.spec` and `.github/workflows/build-executable*.yml`,
   so the built app includes it.
