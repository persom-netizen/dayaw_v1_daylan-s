import 'package:flutter/material.dart';

/// Bottom sheet shown from the Baybayin->Latin capture screen: what to do and
/// what to avoid so a scan actually works, plus the one linguistic limitation.
Future<void> showScanGuide(BuildContext context) {
  return showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.white,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
    ),
    builder: (_) => const _ScanGuide(),
  );
}

class _Rule {
  final String good;
  final String bad;
  const _Rule(this.good, this.bad);
}

const _rules = <_Rule>[
  _Rule("Plain white or off-white paper",
      "No lined, grid, or boxed paper — the lines read as ink"),
  _Rule("Fill the frame — the writing spans most of the width",
      "Don't shoot from far; tiny glyphs break segmentation"),
  _Rule("A clear gap between every character",
      "Don't let glyphs touch or overlap — they merge into one box"),
  _Rule("Shapes close to the Baybayin chart",
      "Avoid heavy cursive, joined strokes, and big end-flourishes"),
  _Rule("Dark ballpen or, best, a marker",
      "No pencil — too faint to threshold"),
  _Rule("Paper flat and upright (under ~15° tilt)",
      "Don't photograph at an angle or rotated"),
];

class _ScanGuide extends StatelessWidget {
  const _ScanGuide();

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 20),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 14),
              const Text("How to scan for best results",
                  style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text("The system is tuned for a narrow set of conditions. "
                  "Outside them, accuracy drops.",
                  style: TextStyle(fontSize: 12.5, color: Colors.grey[600])),
              const SizedBox(height: 14),

              // spacing diagram — the rule people break most
              const _SpacingDiagram(),
              const SizedBox(height: 16),

              for (final r in _rules) ...[
                _ruleRow(true, r.good),
                _ruleRow(false, r.bad),
                const SizedBox(height: 10),
              ],

              const SizedBox(height: 4),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.teal.withValues(alpha: 0.07),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.teal.withValues(alpha: 0.3)),
                ),
                child: const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("Expected limitation: e/i and o/u",
                        style: TextStyle(
                            fontSize: 12.5, fontWeight: FontWeight.w600)),
                    SizedBox(height: 4),
                    Text(
                      "In Baybayin, e and i share one kudlit, and so do o and u. "
                      "The app writes them as i and u. Telling them apart needs "
                      "sentence context and is outside this system's scope — "
                      "that's how the script works, not a bug.",
                      style: TextStyle(fontSize: 12),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text("Got it"),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _ruleRow(bool good, String text) => Padding(
        padding: const EdgeInsets.only(top: 4),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(good ? Icons.check_circle : Icons.cancel,
                size: 17, color: good ? Colors.green[600] : Colors.red[400]),
            const SizedBox(width: 8),
            Expanded(
              child: Text(text,
                  style: TextStyle(
                      fontSize: 12.5,
                      color: good ? Colors.black87 : Colors.grey[700])),
            ),
          ],
        ),
      );
}

/// Two rows of glyph boxes: spaced (ok) vs overlapping (merges).
class _SpacingDiagram extends StatelessWidget {
  const _SpacingDiagram();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.grey[200]!),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Icon(Icons.check_circle, size: 15, color: Colors.green[600]),
              const SizedBox(width: 6),
              const Text("Spaced — each character its own box",
                  style: TextStyle(fontSize: 11.5)),
            ],
          ),
          const SizedBox(height: 6),
          SizedBox(
              height: 42,
              child: CustomPaint(
                  size: const Size(double.infinity, 42),
                  painter: _BoxRowPainter(overlap: false))),
          const SizedBox(height: 12),
          Row(
            children: [
              Icon(Icons.cancel, size: 15, color: Colors.red[400]),
              const SizedBox(width: 6),
              const Text("Touching — boxes intersect, glyphs merge into one",
                  style: TextStyle(fontSize: 11.5)),
            ],
          ),
          const SizedBox(height: 6),
          SizedBox(
              height: 42,
              child: CustomPaint(
                  size: const Size(double.infinity, 42),
                  painter: _BoxRowPainter(overlap: true))),
        ],
      ),
    );
  }
}

class _BoxRowPainter extends CustomPainter {
  final bool overlap;
  _BoxRowPainter({required this.overlap});

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = overlap ? Colors.red.shade400 : Colors.green.shade600;
    final ink = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round
      ..color = Colors.black87;

    const n = 4;
    final boxW = size.width / (n + 1.2);
    final gap = overlap ? -boxW * 0.28 : boxW * 0.30;
    var x = (size.width - (n * boxW + (n - 1) * gap)) / 2;
    for (var i = 0; i < n; i++) {
      final r = Rect.fromLTWH(x, 4, boxW, size.height - 8);
      canvas.drawRect(r, stroke);
      // a little squiggle to stand for a glyph
      final p = Path()
        ..moveTo(r.left + r.width * 0.2, r.center.dy + 6)
        ..cubicTo(r.left + r.width * 0.1, r.top + 4, r.right - r.width * 0.1,
            r.top + 4, r.right - r.width * 0.2, r.center.dy + 6)
        ..lineTo(r.right - r.width * 0.35, r.bottom - 5);
      canvas.drawPath(p, ink);
      x += boxW + gap;
    }
  }

  @override
  bool shouldRepaint(covariant _BoxRowPainter old) => old.overlap != overlap;
}
