"""Diacritic tuning for baybayin_custom.ttf.

1. Enlarge the 5 shared combining marks (U+1712..U+1716) about their centroid
   so they survive screenshot compression - especially the virama 'x'.
2. Add ha-context PUA marks (U+E012..) sitting CLOSER to U+1711 (ha is a thin
   mid-height stroke, so the default marks float far away).
3. Add na-context PUA marks (U+E015..) sitting further from U+1708's descender
   so the virama/below marks stop merging into the glyph ("n" -> "na").

The font has no GPOS/GSUB; marks are zero-advance and placed by their own
coordinates. tagalog_to_baybayin.py picks the ha/na variant by base glyph.
Idempotent: always rebuilds from *pristine* marks stored on first run.
"""
import copy
import json
import os
import shutil
from array import array

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates

FONT = os.path.join(os.path.dirname(__file__), "..", "baybayin_custom.ttf")
PRISTINE = FONT + ".prediac"          # snapshot of marks before any tuning

# ---- tunables -------------------------------------------------------------
# per shared mark: (scale_x, scale_y) about centroid
ENLARGE = {
    0x1712: (1.24, 1.24),   # dot above
    0x1713: (1.24, 1.24),   # dot below
    0x1714: (1.34, 1.34),   # virama x  - biggest bump (often invisible)
    0x1715: (1.10, 1.38),   # dash above - mainly thicker
    0x1716: (1.10, 1.38),   # dash below
}
# desired clear gap (font units, UPM 1000) between base edge and mark near-edge
GAP_HA = 96      # ha: pull marks in to this gap
GAP_NA_ABOVE = 105   # na: small outward nudge on the above marks
GAP_NA_BELOW = 92    # na: push below marks clear of the -66 descender
NA_VIRAMA_DX = 70    # also shove the na virama right, off the descender tail

HA_TOP, HA_BOT = 390, 236
NA_TOP, NA_BOT = 634, -66

# mark codepoint -> (above?/below?, ha PUA, na PUA)
MARKS = {
    0x1712: ("above", 0xE012, 0xE015),   # dot above   (i)
    0x1713: ("below", 0xE013, 0xE016),   # dot below   (o)
    0x1714: ("below", 0xE014, 0xE017),   # virama x
    0x1715: ("above", 0xE018, 0xE01A),   # dash above  (e)
    0x1716: ("below", 0xE019, 0xE01B),   # dash below  (u)
}
# ------------------------------------------------------------------------

if not shutil.os.path.exists(PRISTINE):
    shutil.copyfile(FONT, PRISTINE)

pris = TTFont(PRISTINE)
pris_glyf = pris["glyf"]
pris_cmap = pris.getBestCmap()

f = TTFont(FONT)
glyf, hmtx, vmtx = f["glyf"], f["hmtx"], f.get("vmtx")
cmap_tables = [t for t in f["cmap"].tables if t.isUnicode()]
order = f.getGlyphOrder()


def scaled_coords(coords, sx, sy):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return [(cx + (x - cx) * sx, cy + (y - cy) * sy) for x, y in coords]


def translated(coords, dx, dy):
    return [(x + dx, y + dy) for x, y in coords]


def put_glyph(name, cp, src_glyph, coords, src_name_for_vmtx):
    g = copy.deepcopy(src_glyph)
    g.coordinates = GlyphCoordinates([(round(x), round(y)) for x, y in coords])
    glyf[name] = g
    g.recalcBounds(glyf)
    hmtx[name] = (0, g.xMin)          # zero advance - combining
    if vmtx is not None:
        vmtx[name] = vmtx.metrics.get(src_name_for_vmtx, (0, 0))
    if name not in order:
        order.append(name)
        f.setGlyphOrder(order)
    for t in cmap_tables:
        t.cmap[cp] = name


report = {}
for cp, (scale) in ENLARGE.items():
    src_name = pris_cmap[cp]
    pg = pris_glyf[src_name]
    base_coords = list(pg.coordinates)
    big = scaled_coords(base_coords, *scale)

    # 1. write the enlarged shared mark back in place
    tgt_name = f.getBestCmap()[cp]
    put_glyph(tgt_name, cp, pg, big, src_name)
    eg = glyf[tgt_name]
    report[f"U+{cp:04X} shared"] = (eg.xMin, eg.yMin, eg.xMax, eg.yMax)

    where, ha_cp, na_cp = MARKS[cp]

    # 2. ha variant - move the ENLARGED mark to GAP_HA from ha's edge
    if where == "above":
        near = min(y for _, y in big)                 # bottom edge of mark
        dy = (HA_TOP + GAP_HA) - near
    else:
        near = max(y for _, y in big)                 # top edge of mark
        dy = (HA_BOT - GAP_HA) - near
    put_glyph(f"uni{ha_cp:04X}", ha_cp, pg, translated(big, 0, dy), src_name)
    hg = glyf[f"uni{ha_cp:04X}"]
    report[f"U+{ha_cp:04X} ha"] = (hg.xMin, hg.yMin, hg.xMax, hg.yMax)

    # 3. na variant
    if where == "above":
        near = min(y for _, y in big)
        dy = (NA_TOP + GAP_NA_ABOVE) - near
        dx = 0
    else:
        near = max(y for _, y in big)
        dy = (NA_BOT - GAP_NA_BELOW) - near
        dx = NA_VIRAMA_DX if cp == 0x1714 else 0
    put_glyph(f"uni{na_cp:04X}", na_cp, pg, translated(big, dx, dy), src_name)
    ng = glyf[f"uni{na_cp:04X}"]
    report[f"U+{na_cp:04X} na"] = (ng.xMin, ng.yMin, ng.xMax, ng.yMax)

f["maxp"].numGlyphs = len(f.getGlyphOrder())
f.save(FONT)

print("saved", FONT)
print("numGlyphs", f["maxp"].numGlyphs)
print(json.dumps(report, indent=1))

# reload sanity
g2 = TTFont(FONT)
c2 = g2.getBestCmap()
missing = [hex(cp) for cp in (list(range(0x1712, 0x1717))
                              + list(range(0xE012, 0xE01C))) if cp not in c2]
print("missing after reload:", missing or "none")
