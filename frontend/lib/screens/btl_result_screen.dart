import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import '../services/api_service.dart';
import '../services/scan_logger.dart';

/// Full-page result for a Baybayin -> Tagalog scan: the captured image with the
/// per-glyph bounding boxes on top, then the transliteration (with a copyable
/// lower-case line), the technical scores, tap-to-correct glyph chips, and the
/// archival card.
class BtlResultScreen extends StatefulWidget {
  final Uint8List imageBytes;
  final Size? imageSize;
  final Map<String, dynamic> response;
  final String? scanId;

  const BtlResultScreen({
    super.key,
    required this.imageBytes,
    required this.imageSize,
    required this.response,
    this.scanId,
  });

  @override
  State<BtlResultScreen> createState() => _BtlResultScreenState();
}

class _BtlResultScreenState extends State<BtlResultScreen> {
  bool _isArchived = false;
  bool _isProcessing = false;

  /// Mutable working copy of the detections — corrections land here.
  late final List<Map<String, dynamic>> _working;

  int get _sessionId => (widget.response['session_id'] as num?)?.toInt() ?? 0;

  @override
  void initState() {
    super.initState();
    _working = (widget.response['individual_detections'] as List? ?? [])
        .whereType<Map>()
        .map((d) => Map<String, dynamic>.from(d))
        .toList();
  }

  bool _visible(Map<String, dynamic> d) =>
      d['corrected'] == true || (d['confidence'] as num).toDouble() >= 23.0;

  List<Map<String, dynamic>> get _shown => _working.where(_visible).toList();

  double get _average {
    final list = _shown;
    if (list.isEmpty) return 0.0;
    final total = list.fold<double>(
        0.0, (s, d) => s + (d['confidence'] as num).toDouble());
    return total / list.length;
  }

  String get _resultText {
    final list = _shown;
    if (list.isEmpty) return "—";
    final buffers = <int, StringBuffer>{};
    for (final d in list) {
      final line = (d['line'] as num?)?.toInt() ?? 0;
      final buf = buffers.putIfAbsent(line, () => StringBuffer());
      if (d['space_before'] == true && buf.isNotEmpty) buf.write(' ');
      buf.write(d['char']?.toString() ?? '');
    }
    return (buffers.keys.toList()..sort())
        .map((k) => _titleCaseWords(buffers[k]!.toString().trim()))
        .where((s) => s.isNotEmpty)
        .join('  |  ');
  }

  String _titleCaseWords(String s) => s
      .split(RegExp(r'\s+'))
      .where((w) => w.isNotEmpty)
      .map((w) => w[0].toUpperCase() + w.substring(1).toLowerCase())
      .join(' ');

  /// All-lower-case transliteration for copying.
  String get _plainResult {
    final list = _shown;
    if (list.isEmpty) return "";
    final buffers = <int, StringBuffer>{};
    for (final d in list) {
      final line = (d['line'] as num?)?.toInt() ?? 0;
      final buf = buffers.putIfAbsent(line, () => StringBuffer());
      if (d['space_before'] == true && buf.isNotEmpty) buf.write(' ');
      buf.write((d['char']?.toString() ?? '').toLowerCase());
    }
    return (buffers.keys.toList()..sort())
        .map((k) => buffers[k]!.toString().trim())
        .where((s) => s.isNotEmpty)
        .join('\n');
  }

  void _copyResult() {
    final text = _plainResult.replaceAll('\n', ' ');
    if (text.isEmpty) return;
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text("Result copied"), duration: Duration(seconds: 1)));
  }

  Future<void> _pickAlternative(Map<String, dynamic> d) async {
    final alts = (d['alternatives'] as List? ?? [])
        .whereType<List>()
        .map((a) => [a[0].toString(), (a[1] as num).toDouble()])
        .toList();

    final chosen = await showModalBottomSheet<String>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Text("Correct this character",
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ),
            for (final a in alts)
              ListTile(
                title: Text(a[0] as String, style: const TextStyle(fontSize: 18)),
                trailing: Text("${(a[1] as double).toStringAsFixed(1)}%"),
                onTap: () => Navigator.pop(ctx, a[0] as String),
              ),
            const Divider(height: 1),
            ListTile(
              leading: const Icon(Icons.keyboard),
              title: const Text("Type it manually"),
              onTap: () async {
                final typed = await _promptManual(ctx);
                if (typed != null && typed.isNotEmpty && ctx.mounted) {
                  Navigator.pop(ctx, typed);
                }
              },
            ),
          ],
        ),
      ),
    );

    if (chosen == null || chosen == d['char']) return;
    final from = d['char']?.toString() ?? '';
    setState(() {
      d['char'] = chosen;
      d['corrected'] = true;
      d['is_eligible'] = true;
      final match =
          alts.firstWhere((a) => a[0] == chosen, orElse: () => ['', 100.0]);
      d['confidence'] = match[1] == 0.0 ? 100.0 : match[1];
    });
    ScanLogger.instance.logCorrection(
      scanId: widget.scanId,
      index: _working.indexOf(d),
      from: from,
      to: chosen,
      alternatives: d['alternatives'] as List?,
    );
  }

  Future<String?> _promptManual(BuildContext ctx) {
    final controller = TextEditingController();
    return showDialog<String>(
      context: ctx,
      builder: (c) => AlertDialog(
        title: const Text("Type the character"),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(hintText: "e.g. Ko, Nga, Ba"),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c), child: const Text("Cancel")),
          TextButton(
              onPressed: () => Navigator.pop(c, controller.text.trim()),
              child: const Text("OK")),
        ],
      ),
    );
  }

  Future<void> _handleBulkArchive(List<Map<String, dynamic>> eligible) async {
    if (_isArchived || _isProcessing) return;
    final messenger = ScaffoldMessenger.of(context);
    setState(() => _isProcessing = true);

    final success =
        await ApiService().archiveBulkCharacters(eligible, _sessionId);

    if (!mounted) return;
    setState(() {
      _isProcessing = false;
      if (success) _isArchived = true;
    });
    messenger.showSnackBar(SnackBar(
      content: Text(success ? "Salamat! Data archived." : "Failed to archive."),
      backgroundColor: success ? Colors.green : Colors.red,
    ));
  }

  @override
  Widget build(BuildContext context) {
    final shown = _shown;
    final eligibleForArchive = shown
        .where((d) =>
            d['corrected'] == true ||
            (d['confidence'] as num).toDouble() >= 90.0)
        .toList();
    final correctedCount = _working.where((d) => d['corrected'] == true).length;
    final imgH = MediaQuery.of(context).size.height * 0.38;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Scan Result"),
        backgroundColor: Colors.white,
        foregroundColor: Colors.brown,
        elevation: 1,
      ),
      body: Column(
        children: [
          // --- captured image + detection boxes (pinch to zoom) ---
          Container(
            height: imgH,
            width: double.infinity,
            color: const Color(0xFFEDEDED),
            child: InteractiveViewer(
              maxScale: 5,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Center(
                    child: Image.memory(widget.imageBytes, fit: BoxFit.contain),
                  ),
                  Positioned.fill(
                    child: IgnorePointer(
                      child: CustomPaint(
                        painter: DetectionOverlayPainter(
                          _working,
                          widget.imageSize ?? Size.infinite,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const Divider(height: 1),

          // --- results ---
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("Translation Result",
                      style:
                          TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                  const SizedBox(height: 6),
                  Text(
                    _resultText,
                    style: const TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                        color: Colors.brown),
                  ),
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.fromLTRB(12, 8, 4, 8),
                    decoration: BoxDecoration(
                      color: Colors.grey[100],
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: SelectableText(
                            "Result: ${_plainResult.replaceAll('\n', ' ')}",
                            style: const TextStyle(fontSize: 15),
                          ),
                        ),
                        IconButton(
                          visualDensity: VisualDensity.compact,
                          icon: const Icon(Icons.copy,
                              size: 18, color: Colors.brown),
                          tooltip: "Copy result",
                          onPressed: _copyResult,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  const Text("Tap any character below to fix a wrong reading.",
                      style: TextStyle(fontSize: 12, color: Colors.grey)),
                  const SizedBox(height: 14),
                  _statRow("Detected Characters:", "${shown.length}"),
                  _statRow("Average Confidence:",
                      "${_average.toStringAsFixed(1)}%"),
                  _statRow("Scan Confidence:",
                      "${(widget.response['confidence'] as num? ?? 0).toStringAsFixed(1)}%"),
                  if (correctedCount > 0)
                    _statRow("Corrected by you:", "$correctedCount"),
                  const Divider(height: 32),
                  shown.isEmpty
                      ? const Padding(
                          padding: EdgeInsets.all(20),
                          child: Text(
                            "No characters detected above 23% confidence.",
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Colors.grey),
                          ),
                        )
                      : Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: shown.map(_glyphChip).toList(),
                        ),
                  const SizedBox(height: 26),
                  if (eligibleForArchive.isNotEmpty)
                    _buildArchiveCard(eligibleForArchive)
                  else
                    const Center(
                      child: Text(
                        "No high-confidence characters eligible for archival.",
                        textAlign: TextAlign.center,
                        style: TextStyle(color: Colors.grey, fontSize: 12),
                      ),
                    ),
                  const SizedBox(height: 24),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.grey[100],
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Text(
                        "Tip: Clear handwriting improves AI learning!",
                        style: TextStyle(fontSize: 12)),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _glyphChip(Map<String, dynamic> d) {
    final conf = (d['confidence'] as num).toDouble();
    final corrected = d['corrected'] == true;
    final Color bg = corrected
        ? Colors.blue[50]!
        : (conf >= 90 ? Colors.green[50]! : Colors.orange[50]!);
    final Color border = corrected
        ? Colors.blue
        : (conf >= 90 ? Colors.green : Colors.orange);

    return InkWell(
      onTap: () => _pickAlternative(d),
      borderRadius: BorderRadius.circular(10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: border.withValues(alpha: 0.6)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(d['char']?.toString() ?? '?',
                    style: const TextStyle(
                        fontSize: 20, fontWeight: FontWeight.bold)),
                if (corrected) ...[
                  const SizedBox(width: 4),
                  const Icon(Icons.edit, size: 12, color: Colors.blue),
                ],
              ],
            ),
            Text(corrected ? "fixed" : "${conf.toStringAsFixed(0)}%",
                style: TextStyle(fontSize: 10, color: border)),
          ],
        ),
      ),
    );
  }

  Widget _statRow(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(
          children: [
            Text(label),
            const SizedBox(width: 10),
            Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
      );

  Widget _buildArchiveCard(List<Map<String, dynamic>> eligible) {
    final archived = _isArchived;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: archived ? Colors.grey[100] : Colors.orange[50],
        borderRadius: BorderRadius.circular(15),
        border: Border.all(
            color: archived ? Colors.grey[300]! : Colors.orange[200]!),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Icon(archived ? Icons.cloud_done : Icons.volunteer_activism,
                  color: archived ? Colors.grey : Colors.orange[800]),
              const SizedBox(width: 10),
              Text(archived ? "Data Saved to Archive" : "Help Dayaw Grow",
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: archived ? Colors.grey : Colors.black)),
            ],
          ),
          const SizedBox(height: 10),
          if (!archived)
            Text("We detected ${eligible.length} high-quality strokes "
                "(corrections included). Permit us to save them?"),
          const SizedBox(height: 15),
          ElevatedButton.icon(
            onPressed: (archived || _isProcessing)
                ? null
                : () => _handleBulkArchive(eligible),
            icon: _isProcessing
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
                : Icon(archived ? Icons.check : Icons.check_circle_outline),
            label: Text(archived
                ? "Archived Successfully"
                : "Archive All Eligible Strokes"),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.orange[800],
              foregroundColor: Colors.white,
              minimumSize: const Size(double.infinity, 45),
            ),
          ),
        ],
      ),
    );
  }
}

/// Draws the per-glyph detection boxes over the displayed photo.
///
/// The backend returns `bbox` as [x, y, w, h] fractions (0..1) of its own
/// processed image (deskewed, width-normalised). The photo is shown with
/// BoxFit.contain, so it only fills a letterboxed sub-rectangle of the paint
/// area — we reconstruct that rectangle from `imageSize`'s aspect ratio and map
/// the fractions onto it.
class DetectionOverlayPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  final Size imageSize;

  DetectionOverlayPainter(this.detections, this.imageSize);

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty ||
        !imageSize.isFinite ||
        imageSize.width <= 0 ||
        imageSize.height <= 0 ||
        size.isEmpty) {
      return;
    }

    final imgAspect = imageSize.width / imageSize.height;
    final boxAspect = size.width / size.height;
    final double renderW, renderH;
    if (imgAspect > boxAspect) {
      renderW = size.width;
      renderH = size.width / imgAspect;
    } else {
      renderH = size.height;
      renderW = size.height * imgAspect;
    }
    final offX = (size.width - renderW) / 2;
    final offY = (size.height - renderH) / 2;

    for (final d in detections) {
      final box = d['bbox'] as List<dynamic>?;
      if (box == null || box.length < 4) continue;

      var rect = Rect.fromLTWH(
        offX + (box[0] as num).toDouble() * renderW,
        offY + (box[1] as num).toDouble() * renderH,
        (box[2] as num).toDouble() * renderW,
        (box[3] as num).toDouble() * renderH,
      );
      rect = rect.intersect(Rect.fromLTWH(offX, offY, renderW, renderH));
      if (rect.width <= 1 || rect.height <= 1) continue;

      final conf = (d['confidence'] as num?)?.toDouble() ?? 0.0;
      final color = conf >= 90.0
          ? Colors.greenAccent
          : (conf >= 23.0 ? Colors.yellowAccent : Colors.orangeAccent);

      canvas.drawRect(
          rect,
          Paint()
            ..style = PaintingStyle.fill
            ..color = color.withValues(alpha: 0.10));
      canvas.drawRect(
          rect,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2.0
            ..color = color.withValues(alpha: 0.95));

      final ch = d['char']?.toString() ?? '';
      if (ch.isNotEmpty && rect.width > 14) {
        final tp = TextPainter(
          text: TextSpan(
            text: ch,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.bold,
              shadows: const [Shadow(color: Colors.black, blurRadius: 2)],
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas,
            Offset(rect.left, (rect.top - 13).clamp(0.0, size.height)));
      }
    }
  }

  @override
  bool shouldRepaint(covariant DetectionOverlayPainter old) =>
      old.detections != detections || old.imageSize != imageSize;
}
