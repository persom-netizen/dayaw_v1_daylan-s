"""Add two dedicated glyphs to baybayin_custom.ttf so the standalone vowels
I and U are visually distinct from E and O:

  uniE010  =  the I/E base glyph (uni1701)  +  a short vertical line, CENTRED ABOVE
  uniE011  =  the U/O base glyph (uni1702)  +  a short vertical tick BOTTOM-RIGHT
             (enlarged to 50 x 300 font units after field feedback)

Full glyphs with their own advance width (not combining marks), so the shaper
renders them exactly as drawn. tagalog_to_baybayin.py maps standalone 'i' ->
U+E010 and 'u' -> U+E011; nothing else changes.

Idempotent: if the glyphs already exist they are rebuilt from the base glyph,
so re-running never stacks contours.
"""
import copy
import os
import shutil
from array import array

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.pens.boundsPen import BoundsPen

FONT = os.path.join(os.path.dirname(__file__), "..", "baybayin_custom.ttf")

if not shutil.os.path.exists(FONT + ".bak"):
    shutil.copyfile(FONT, FONT + ".bak")

f = TTFont(FONT)
glyf, hmtx = f["glyf"], f["hmtx"]
vmtx = f.get("vmtx")


def add_stroke(src_name, new_name, cp, rect, advance):
    """new_name = deepcopy of src glyph + one rectangular contour `rect`
    (4 on-curve (x, y) corners). Always rebuilt from src so it's idempotent."""
    g = copy.deepcopy(glyf[src_name])
    coords = list(g.coordinates) + list(rect)
    g.coordinates = GlyphCoordinates(coords)
    g.flags = array("B", list(g.flags) + [1, 1, 1, 1])
    g.endPtsOfContours = list(g.endPtsOfContours) + [len(coords) - 1]
    g.numberOfContours = len(g.endPtsOfContours)

    glyf[new_name] = g
    hmtx[new_name] = (advance, min(x for x, _ in coords))
    if vmtx is not None:
        vmtx[new_name] = vmtx.metrics.get(src_name, (0, 0))
    order = f.getGlyphOrder()
    if new_name not in order:
        f.setGlyphOrder(order + [new_name])
    for t in f["cmap"].tables:
        if t.isUnicode():
            t.cmap[cp] = new_name


# I : vertical line centred above (base uni1701 x[32,978] y[78,579] adv 1011)
add_stroke("uni1701", "uniE010", 0xE010,
           rect=[(492, 615), (492, 835), (520, 835), (520, 615)], advance=1011)

# U : vertical tick bottom-right, 50 x 300 (base uni1702 x[48,487] y[0,634] adv 535)
add_stroke("uni1702", "uniE011", 0xE011,
           rect=[(444, 60), (444, -240), (494, -240), (494, 60)], advance=535)

f["maxp"].numGlyphs = len(f.getGlyphOrder())
f.save(FONT)
print("saved", FONT)

g = TTFont(FONT)
gs, cmap = g.getGlyphSet(), g.getBestCmap()
for cp in (0xE010, 0xE011):
    n = cmap[cp]
    bp = BoundsPen(gs)
    gs[n].draw(bp)
    print(f"U+{cp:04X} -> {n}  bounds {tuple(round(v) for v in bp.bounds)}  "
          f"adv {g['hmtx'][n][0]}  contours {g['glyf'][n].numberOfContours}")
print("numGlyphs", g["maxp"].numGlyphs, "loads OK")
