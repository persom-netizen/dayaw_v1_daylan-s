"""Horizontally centre the BELOW kudlit marks in baybayin_custom.ttf.

The marks are zero-advance combining glyphs placed purely by their own
coordinates (the font has no GPOS). They were drawn sitting ~107 units left of
where a typical consonant's ink centre falls, so "ko" / "go" / "do" render with
the dot pushed to the left instead of under the middle of the body.

This shifts every below mark (+ its ha / na context variants) right by DX so the
dot / dash / virama sits under the centre of the base. Above marks (i, e) are
left untouched. BTL is unaffected: the V7 mark descriptor measures the mark's
size / shape / vertical side, not its horizontal offset.

Runs AFTER tune_diacritics.py (it shifts whatever marks are currently in the
font). Idempotent: first run snapshots the marks to baybayin_custom.ttf.premarkx
and every run rebuilds from that snapshot, so re-running never compounds.
"""
import copy
import json
import os
import shutil

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates

FONT = os.path.join(os.path.dirname(__file__), "..", "baybayin_custom.ttf")
SNAPSHOT = FONT + ".premarkx"

# +x shift, font units (UPM 1000). +107 puts the o-dot dead centre under Ka
# (ink centre 492, advance 984 -> wanted mark-ink-centre -492; it sits at -598).
DX = 107

# below marks + their ha / na PUA variants (see tune_diacritics.py MARKS)
BELOW = {
    0x1713: (0xE013, 0xE016),   # dot below   (o)
    0x1714: (0xE014, 0xE017),   # virama x
    0x1716: (0xE019, 0xE01B),   # dash below  (u)
}

if not os.path.exists(SNAPSHOT):
    shutil.copyfile(FONT, SNAPSHOT)

src = TTFont(SNAPSHOT)
src_glyf, src_cmap = src["glyf"], src.getBestCmap()

f = TTFont(FONT)
glyf, hmtx = f["glyf"], f["hmtx"]
cmap_tables = [t for t in f["cmap"].tables if t.isUnicode()]

report = {}
for base_cp, variants in BELOW.items():
    for cp in (base_cp, *variants):
        name = src_cmap.get(cp)
        if name is None or name not in src_glyf.glyphs:
            report[f"U+{cp:04X}"] = "missing - skipped"
            continue
        g = copy.deepcopy(src_glyf[name])
        g.coordinates = GlyphCoordinates(
            [(x + DX, y) for x, y in g.coordinates])
        g.recalcBounds(glyf)
        glyf[name] = g
        hmtx[name] = (0, g.xMin)                       # stay zero-advance
        for t in cmap_tables:
            t.cmap[cp] = name
        report[f"U+{cp:04X} {name}"] = (g.xMin, g.yMin, g.xMax, g.yMax)

f.save(FONT)
print(f"saved {FONT}  (DX = +{DX})")
print(json.dumps(report, indent=1))

g2 = TTFont(FONT)
gs2, c2 = g2.getGlyphSet(), g2.getBestCmap()
adv = g2["hmtx"]["uni1703"][0]
for cp, tag in [(0x1713, "o dot"), (0x1716, "u dash"), (0x1714, "virama")]:
    from fontTools.pens.boundsPen import BoundsPen
    bp = BoundsPen(gs2)
    gs2[c2[cp]].draw(bp)
    if bp.bounds:
        mcx = (bp.bounds[0] + bp.bounds[2]) / 2
        print(f"  {tag:8s} under Ka: mark ink cx {adv + mcx:+.0f}  "
              f"(Ka ink centre 492)")
print("reload OK, numGlyphs", g2["maxp"].numGlyphs)
