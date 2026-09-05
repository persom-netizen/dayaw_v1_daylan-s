# baybayin_custom.ttf — glyph tooling

`baybayin_custom.ttf` is hand-edited (no source `.glyphs`/`.ufo`). These scripts
are the record of every change made after the original, so the font can be
rebuilt from the base if it is ever lost. They need `fonttools`
(`pip install fonttools`); `render_sample.py` also needs `matplotlib`.

Run them **in order**, from anywhere:

```
python add_iu_glyphs.py       # 370 -> 372 glyphs
python tune_diacritics.py     # 372 -> 382 glyphs
```

Both are idempotent — each keeps a one-time pristine snapshot next to the font
(`*.bak`, `*.prediac`) and rebuilds its glyphs from that, so re-running never
stacks contours. Delete the snapshot files before committing the font.

## What each script does

### `add_iu_glyphs.py`
Adds two full glyphs so a typed standalone **i / u** is visually distinct from
**e / o** (they share a base shape and otherwise round-trip through the BTL OCR
as e/o):

| glyph | = base + added stroke | codepoint | consumed by |
|---|---|---|---|
| `uniE010` | `uni1701` (i/e) + short vertical line centred above | U+E010 | `tagalog_to_baybayin.py` `base_map['i']` |
| `uniE011` | `uni1702` (u/o) + vertical tick bottom-right (50×300) | U+E011 | `tagalog_to_baybayin.py` `base_map['u']` |

### `tune_diacritics.py`
1. Enlarges the 5 shared combining marks U+1712–U+1716 about their centroid
   (virama "x" +34 %, dots +24 %, dashes thicker) so they survive screenshot
   compression.
2. Adds **ha-context** marks U+E012–U+E019 that sit ~96 units from `uni1711`
   (ha is a thin mid-height stroke, so the default marks float far away).
3. Adds **na-context** marks U+E015–U+E01B whose below-marks sit lower and whose
   virama is nudged right, clear of `uni1708`'s descender (otherwise the virama
   merges into the loop and "n" reads as "na").

`tagalog_to_baybayin.py` picks the ha/na variant in `_mark(key, slot)`; every
other base keeps the enlarged shared marks. Tune the gaps via `GAP_HA`,
`GAP_NA_ABOVE`, `GAP_NA_BELOW`, `NA_VIRAMA_DX` and the `ENLARGE` table at the
top of the script.

### `render_sample.py`
`python render_sample.py <font.ttf> <out.png>` — draws base+mark combinations
the way this GPOS-less font actually stacks them, for eyeballing spacing.
