import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

FONT = sys.argv[1]
OUT = sys.argv[2]


class MplPen(BasePen):
    def __init__(self, glyphSet, dx=0):
        super().__init__(glyphSet)
        self.verts, self.codes, self.dx = [], [], dx
    def _moveTo(self, p):
        self.verts.append((p[0] + self.dx, p[1])); self.codes.append(Path.MOVETO)
    def _lineTo(self, p):
        self.verts.append((p[0] + self.dx, p[1])); self.codes.append(Path.LINETO)
    def _curveToOne(self, p1, p2, p3):
        for p in (p1, p2, p3):
            self.verts.append((p[0] + self.dx, p[1])); self.codes.append(Path.CURVE4)
    def _qCurveToOne(self, p1, p2):
        self.verts.append((p1[0] + self.dx, p1[1])); self.codes.append(Path.CURVE3)
        self.verts.append((p2[0] + self.dx, p2[1])); self.codes.append(Path.CURVE3)
    def _closePath(self):
        pass


f = TTFont(FONT)
gs = f.getGlyphSet()
cmap = f.getBestCmap()
hmtx = f["hmtx"]

SAMPLES = [
    ("hi  1711+E012", [0x1711, 0xE012]),
    ("ho  1711+E013", [0x1711, 0xE013]),
    ("h   1711+E014", [0x1711, 0xE014]),
    ("he  1711+E018", [0x1711, 0xE018]),
    ("hu  1711+E019", [0x1711, 0xE019]),
    ("ni  1708+E015", [0x1708, 0xE015]),
    ("no  1708+E016", [0x1708, 0xE016]),
    ("n   1708+E017", [0x1708, 0xE017]),
    ("nu  1708+E01B", [0x1708, 0xE01B]),
    ("k   1703+1714 (enlarged, ctl)", [0x1703, 0x1714]),
    ("ki  1703+1712 (enlarged, ctl)", [0x1703, 0x1712]),
    ("ko  1703+1713 (enlarged, ctl)", [0x1703, 0x1713]),
]

n = len(SAMPLES)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.9))
axes = axes.flatten()
for ax, (title, cps) in zip(axes, SAMPLES):
    penx = 0
    for i, cp in enumerate(cps):
        gname = cmap.get(cp)
        pen = MplPen(gs, dx=penx)
        gs[gname].draw(pen)
        if pen.verts:
            color = "#222" if i == 0 else "#c0392b"
            ax.add_patch(PathPatch(Path(pen.verts, pen.codes), fc=color, ec="none"))
        penx += hmtx[gname][0]
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(-200, 1400); ax.set_ylim(-500, 1050)
    ax.set_aspect("equal")
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[n:]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT, dpi=110)
print("wrote", OUT)
