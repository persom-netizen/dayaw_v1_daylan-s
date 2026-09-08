import 'dart:convert' show base64Decode;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show compute;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'package:share_plus/share_plus.dart';
import '../services/api_service.dart';
import '../services/lexicon.dart';
import '../services/result_exporter.dart';
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

  /// The server response currently on screen. Replaced when the user flips the
  /// processing-mode toggle and we re-scan the same image.
  late Map<String, dynamic> _response;
  String? _scanId;

  /// White-paper mode: backend skips the noise-robustness steps (downscale,
  /// dilation, small-object removal) that also erode thin-pen kudlit dots.
  late bool _whitePaper;
  /// 'marker' (default) or 'pen' — pen mode thickens thin strokes ~1px and
  /// keeps the noise floor low so a ballpen kudlit dot survives.
  late String _penType;
  bool _rescanning = false;

  /// Mutable working copy of the detections — corrections land here.
  late List<Map<String, dynamic>> _working;

  /// Decoded stage previews ("what the computer sees") + which one is shown.
  Map<String, Uint8List> _stages = {};
  String _view = '2_flattened'; // default to a processed frame, not the raw capture

  /// Full ordered pipeline walkthrough, fetched on demand.
  List<Map<String, dynamic>> _vizCards = [];
  bool _vizLoading = false;
  bool _vizLoaded = false;
  bool _saving = false;

  /// Optional Tagalog spell-snapping of the displayed result. Off by default.
  bool _lexiconOn = false;
  int _snapped = 0;

  static const _stageLabels = {
    '0_raw': 'Raw',
    '1_normalized': 'Normalized',
    '2_flattened': 'Flattened',
    '3_binary': 'Binary',
    '4_grouped': 'Grouped',
  };
  static const _stageHints = {
    '0_raw': 'Your capture, as sent.',
    '1_normalized': 'Resized to the working width.',
    '2_flattened': 'Lighting removed — what Otsu thresholds. Boxes sit on this frame.',
    '3_binary': 'Pure black/white — what findContours traces.',
    '4_grouped': 'After dilation / proximity-merge — one blob per glyph.',
  };

  int get _sessionId => (_response['session_id'] as num?)?.toInt() ?? 0;

  @override
  void initState() {
    super.initState();
    _response = widget.response;
    _scanId = widget.scanId;
    _whitePaper = _response['white_paper'] == true;
    _penType = _response['pen_type'] == 'pen' ? 'pen' : 'marker';
    _working = _detectionsFrom(_response);
    _decodeStages(_response);
    _hydrateViz(_response);
    Lexicon.instance.load().then((_) {
      if (mounted) setState(() {});
    });
  }

  /// The result text as shown / copied — with Tagalog snapping applied when the
  /// toggle is on.
  String _lex(String s) {
    if (!_lexiconOn) return s;
    return Lexicon.instance.correctPhrase(s).$1;
  }

  void _toggleLexicon(bool on) {
    final n = on ? Lexicon.instance.correctPhrase(_plainResult).$2 : 0;
    setState(() {
      _lexiconOn = on;
      _snapped = n;
    });
  }

  void _hydrateViz(Map<String, dynamic> resp) {
    final raw = resp['visualize_b64'];
    if (raw is List && raw.isNotEmpty) {
      _vizCards = raw.whereType<Map>().map((m) {
        final card = Map<String, dynamic>.from(m);
        final img = card['img'];
        if (img is String && img.isNotEmpty) {
          try {
            card['bytes'] = base64Decode(img);
          } catch (_) {}
        }
        return card;
      }).toList();
      _vizLoaded = _vizCards.isNotEmpty;
    }
  }

  Future<void> _saveResults() async {
    if (_saving) return;
    setState(() => _saving = true);
    final messenger = ScaffoldMessenger.of(context);
    try {
      if (!_vizLoaded && !_vizLoading) await _loadViz();
      if (!mounted) return;

      final proc = _response['processed_size'];
      final pw = (proc is List && proc.isNotEmpty) ? (proc[0] as num).toInt() : 0;
      final ph =
          (proc is List && proc.length > 1) ? (proc[1] as num).toInt() : 0;
      final shown = _shown;

      final payload = ExportPayload(
        heading: 'DAYAW  -  Baybayin to Latin',
        timestamp: DateTime.now().toString().split('.').first,
        title: _resultText,
        plain: _plainResult.replaceAll('\n', '   '),
        statLines: [
          'Detected characters: ${shown.length}',
          'Average confidence: ${_average.toStringAsFixed(1)}%',
          'Scan confidence: '
              '${(_response['confidence'] as num? ?? 0).toStringAsFixed(1)}%',
          'Processing mode: ${_whitePaper ? 'white-paper' : 'standard'} / $_penType',
        ],
        processed: _stages['2_flattened'],
        procW: pw,
        procH: ph,
        boxesPx: _working.map<List<int>>((d) {
          final b = d['bbox_px'];
          return (b is List && b.length >= 4)
              ? [
                  (b[0] as num).toInt(),
                  (b[1] as num).toInt(),
                  (b[2] as num).toInt(),
                  (b[3] as num).toInt(),
                ]
              : <int>[];
        }).toList(),
        boxLabels: _working.map((d) => d['char']?.toString() ?? '').toList(),
        viz: _vizCards
            .where((c) => c['bytes'] is Uint8List)
            .map((c) => ExportVizCard(
                  part: c['part']?.toString() ?? '',
                  title: c['title']?.toString() ?? '',
                  caption: c['caption']?.toString() ?? '',
                  bytes: c['bytes'] as Uint8List,
                ))
            .toList(),
      );

      final bytes = await compute(buildResultSheet, payload);
      if (!mounted) return;
      final name = 'dayaw_scan_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await Share.shareXFiles(
        [XFile.fromData(bytes, name: name, mimeType: 'image/jpeg')],
        text: 'Dayaw scan result',
      );
    } catch (e) {
      messenger.showSnackBar(
          SnackBar(content: Text('Could not save: $e')));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _loadViz() async {
    if (_vizLoading || _vizLoaded) return;
    setState(() => _vizLoading = true);
    final resp = await ApiService().uploadAndTranslateDetailed(
      null,
      'Baybayin to Tagalog',
      imageBytes: widget.imageBytes,
      whitePaper: _whitePaper,
      penType: _penType,
      visualize: true,
    );
    if (!mounted) return;
    if (resp == null) {
      setState(() => _vizLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text("Could not load the visualization."),
          duration: Duration(seconds: 2)));
      return;
    }
    setState(() {
      _hydrateViz(resp);
      _vizLoading = false;
    });
  }

  void _decodeStages(Map<String, dynamic> resp) {
    final out = <String, Uint8List>{};
    final s = resp['stages_b64'];
    if (s is Map) {
      s.forEach((k, v) {
        if (v is String && v.isNotEmpty) {
          try {
            out[k.toString()] = base64Decode(v);
          } catch (_) {}
        }
      });
    }
    _stages = out;
    if (!_stages.containsKey(_view)) {
      _view = _stages.containsKey('2_flattened')
          ? '2_flattened'
          : (_stages.isEmpty ? '0_raw' : _stages.keys.first);
    }
  }

  /// Size of the frame the bbox coords refer to (the flattened stage).
  Size get _procSize {
    final p = _response['processed_size'];
    if (p is List && p.length >= 2) {
      return Size((p[0] as num).toDouble(), (p[1] as num).toDouble());
    }
    return widget.imageSize ?? Size.infinite;
  }

  List<Map<String, dynamic>> _detectionsFrom(Map<String, dynamic> resp) =>
      (resp['individual_detections'] as List? ?? [])
          .whereType<Map>()
          .map((d) => Map<String, dynamic>.from(d))
          .toList();

  Future<void> _rescan({bool? whitePaper, String? penType}) async {
    if (_rescanning) return;
    final wp = whitePaper ?? _whitePaper;
    final pt = penType ?? _penType;
    setState(() {
      _rescanning = true;
      _whitePaper = wp;
      _penType = pt;
    });
    final resp = await ApiService().uploadAndTranslateDetailed(
      null,
      'Baybayin to Tagalog',
      imageBytes: widget.imageBytes,
      whitePaper: wp,
      penType: pt,
    );
    if (!mounted) return;
    if (resp == null) {
      setState(() => _rescanning = false);
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text("Re-scan failed — check the connection."),
          duration: Duration(seconds: 2)));
      return;
    }
    final newId = await ScanLogger.instance
        .logScan(imageBytes: widget.imageBytes, response: resp);
    if (!mounted) return;
    setState(() {
      _response = resp;
      _scanId = newId;
      _working = _detectionsFrom(resp);
      _decodeStages(resp);
      _isArchived = false;
      _rescanning = false;
    });
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
    final text = _lex(_plainResult).replaceAll('\n', ' ');
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
      scanId: _scanId,
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
        actions: [
          _saving
              ? const Padding(
                  padding: EdgeInsets.all(14),
                  child: SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.brown),
                  ),
                )
              : IconButton(
                  icon: const Icon(Icons.save_alt),
                  tooltip: "Save scanned results (JPG)",
                  onPressed: _saveResults,
                ),
        ],
      ),
      body: Column(
        children: [
          // --- stage preview ("what the computer sees") + detection boxes ---
          _stageViewer(imgH),
          _stagePicker(),
          const Divider(height: 1),

          // --- results ---
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (_qualityNotes.isNotEmpty) ...[
                    _qualityCard(),
                    const SizedBox(height: 12),
                  ],
                  const Text("Translation Result",
                      style:
                          TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                  const SizedBox(height: 6),
                  Text(
                    _lex(_resultText),
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
                            "Result: ${_lex(_plainResult).replaceAll('\n', ' ')}",
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
                  const SizedBox(height: 12),
                  _lexiconToggle(),
                  const SizedBox(height: 8),
                  _penMarkerToggle(),
                  const SizedBox(height: 8),
                  _modeToggle(),
                  const SizedBox(height: 8),
                  _visualizeSection(),
                  const SizedBox(height: 10),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: _saving ? null : _saveResults,
                      icon: _saving
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(
                                  strokeWidth: 2, color: Colors.brown))
                          : const Icon(Icons.save_alt, size: 18),
                      label: Text(_saving
                          ? "Rendering…"
                          : "Save scanned results (JPG)"),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.brown,
                        side: BorderSide(
                            color: Colors.brown.withValues(alpha: 0.5)),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text("Tap any character below to fix a wrong reading.",
                      style: TextStyle(fontSize: 12, color: Colors.grey)),
                  const SizedBox(height: 14),
                  _statRow("Detected Characters:", "${shown.length}"),
                  _statRow("Average Confidence:",
                      "${_average.toStringAsFixed(1)}%"),
                  _statRow("Scan Confidence:",
                      "${(_response['confidence'] as num? ?? 0).toStringAsFixed(1)}%"),
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

  Widget _stageViewer(double h) {
    final raw = _view == '0_raw' || _stages.isEmpty;
    final bytes =
        raw ? widget.imageBytes : (_stages[_view] ?? widget.imageBytes);
    final overlaySize = raw ? (widget.imageSize ?? Size.infinite) : _procSize;
    return Container(
      height: h,
      width: double.infinity,
      color: const Color(0xFFEDEDED),
      child: InteractiveViewer(
        maxScale: 5,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Center(
              child: Image.memory(bytes,
                  fit: BoxFit.contain, gaplessPlayback: true),
            ),
            Positioned.fill(
              child: IgnorePointer(
                child: CustomPaint(
                  painter: DetectionOverlayPainter(_working, overlaySize),
                ),
              ),
            ),
            if (_rescanning)
              const Positioned.fill(
                child: ColoredBox(
                  color: Color(0x22000000),
                  child: Center(
                      child: CircularProgressIndicator(color: Colors.brown)),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _stagePicker() {
    if (_stages.isEmpty) return const SizedBox.shrink();
    final keys = _stages.keys.toList()..sort();
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(10, 8, 10, 6),
      color: const Color(0xFFF7F7F7),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                for (final k in keys)
                  Padding(
                    padding: const EdgeInsets.only(right: 6),
                    child: ChoiceChip(
                      label: Text(_stageLabels[k] ?? k,
                          style: const TextStyle(fontSize: 12)),
                      selected: _view == k,
                      selectedColor: Colors.brown.withValues(alpha: 0.18),
                      visualDensity: VisualDensity.compact,
                      onSelected: (_) => setState(() => _view = k),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 2),
          Text(
            _stageHints[_view] ?? "What the computer sees at this stage.",
            style: const TextStyle(fontSize: 10.5, color: Colors.grey),
          ),
        ],
      ),
    );
  }

  Widget _visualizeSection() {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.brown.withValues(alpha: 0.25)),
        borderRadius: BorderRadius.circular(10),
      ),
      clipBehavior: Clip.antiAlias,
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(horizontal: 12),
          childrenPadding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
          leading: const Icon(Icons.blur_on, color: Colors.brown, size: 20),
          title: const Text("Visualize process",
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          subtitle: const Text("Every step, from raw image to model prediction",
              style: TextStyle(fontSize: 11, color: Colors.grey)),
          onExpansionChanged: (open) {
            if (open && !_vizLoaded && !_vizLoading) _loadViz();
          },
          children: [
            if (_vizLoading)
              const Padding(
                padding: EdgeInsets.all(20),
                child: Column(children: [
                  CircularProgressIndicator(color: Colors.brown),
                  SizedBox(height: 10),
                  Text("Rendering the pipeline…",
                      style: TextStyle(fontSize: 12, color: Colors.grey)),
                ]),
              )
            else if (_vizCards.isEmpty)
              const Padding(
                padding: EdgeInsets.all(16),
                child: Text("No visualization available.",
                    style: TextStyle(fontSize: 12, color: Colors.grey)),
              )
            else
              ..._buildVizCards(),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildVizCards() {
    final widgets = <Widget>[];
    String? part;
    for (final c in _vizCards) {
      final p = c['part']?.toString() ?? '';
      if (p != part) {
        part = p;
        widgets.add(Padding(
          padding: const EdgeInsets.fromLTRB(2, 12, 2, 6),
          child: Text(p.toUpperCase(),
              style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  color: Colors.brown,
                  letterSpacing: 0.5)),
        ));
      }
      final bytes = c['bytes'];
      widgets.add(Padding(
        padding: const EdgeInsets.only(bottom: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(c['title']?.toString() ?? '',
                style: const TextStyle(
                    fontSize: 12.5, fontWeight: FontWeight.w600)),
            const SizedBox(height: 4),
            if (bytes is Uint8List)
              ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: InteractiveViewer(
                  maxScale: 6,
                  child: Container(
                    color: const Color(0xFFECECEC),
                    width: double.infinity,
                    child: Image.memory(bytes, fit: BoxFit.contain),
                  ),
                ),
              ),
            const SizedBox(height: 4),
            Text(c['caption']?.toString() ?? '',
                style: const TextStyle(fontSize: 11, color: Colors.grey)),
          ],
        ),
      ));
    }
    return widgets;
  }

  List<String> get _qualityNotes {
    final raw = _response['quality_notes'];
    final out = <String>[];
    if (raw is List) {
      for (final n in raw) {
        if (n is String && n.trim().isNotEmpty) out.add(n.trim());
      }
    }
    if (out.isEmpty && _response['capture_warning'] is String) {
      out.add(_response['capture_warning'] as String);
    }
    return out;
  }

  Widget _qualityCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber[50],
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.amber[300]!),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.warning_amber_rounded, size: 18, color: Colors.amber[800]),
              const SizedBox(width: 8),
              const Text("This scan hit a known limitation",
                  style: TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600)),
            ],
          ),
          const SizedBox(height: 6),
          for (final note in _qualityNotes)
            Padding(
              padding: const EdgeInsets.only(left: 2, top: 3),
              child: Text("•  $note", style: const TextStyle(fontSize: 12)),
            ),
        ],
      ),
    );
  }

  Widget _lexiconToggle() {
    final ready = Lexicon.instance.isLoaded;
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 4, 8, 4),
      decoration: BoxDecoration(
        color: _lexiconOn ? Colors.teal.withValues(alpha: 0.06) : Colors.grey[100],
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: _lexiconOn
                ? Colors.teal.withValues(alpha: 0.4)
                : Colors.grey.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.spellcheck, size: 18, color: Colors.teal),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text("Snap to Tagalog words",
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                Text(
                  !ready
                      ? "Loading word list…"
                      : (_lexiconOn
                          ? (_snapped == 0
                              ? "On — no words needed snapping."
                              : "On — $_snapped word${_snapped == 1 ? '' : 's'} "
                                  "snapped to the nearest Tagalog word.")
                          : "Off — showing the raw model output. "
                              "Optional spelling fix, does not change detections."),
                  style: const TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
          ),
          Switch(
            value: _lexiconOn,
            activeThumbColor: Colors.teal,
            onChanged: ready ? _toggleLexicon : null,
          ),
        ],
      ),
    );
  }

  Widget _penMarkerToggle() {
    Widget seg(String value, String label, IconData icon) {
      final on = _penType == value;
      return Expanded(
        child: InkWell(
          onTap: (_rescanning || on) ? null : () => _rescan(penType: value),
          borderRadius: BorderRadius.circular(8),
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 7),
            decoration: BoxDecoration(
              color: on ? Colors.brown.withValues(alpha: 0.12) : Colors.transparent,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                  color: on
                      ? Colors.brown.withValues(alpha: 0.5)
                      : Colors.grey.withValues(alpha: 0.25)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon,
                    size: 15,
                    color: on ? Colors.brown : Colors.grey),
                const SizedBox(width: 5),
                Text(label,
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: on ? FontWeight.w600 : FontWeight.normal,
                        color: on ? Colors.brown : Colors.grey[700])),
              ],
            ),
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text("Writing tool",
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
            if (_rescanning) ...[
              const SizedBox(width: 8),
              const SizedBox(
                width: 12,
                height: 12,
                child: CircularProgressIndicator(
                    strokeWidth: 2, color: Colors.brown),
              ),
            ],
          ],
        ),
        const SizedBox(height: 4),
        Row(children: [
          seg('marker', 'Marker', Icons.brush),
          const SizedBox(width: 8),
          seg('pen', 'Pen', Icons.edit),
        ]),
        const SizedBox(height: 3),
        Text(
          _penType == 'pen'
              ? "Pen: thin strokes thickened ~1px, low noise floor — keeps ballpen kudlit dots."
              : "Marker: standard pipeline (dataset is marker-weight).",
          style: const TextStyle(fontSize: 11, color: Colors.grey),
        ),
      ],
    );
  }

  Widget _modeToggle() {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 4, 8, 4),
      decoration: BoxDecoration(
        color: _whitePaper ? Colors.brown.withValues(alpha: 0.06) : Colors.grey[100],
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: _whitePaper
                ? Colors.brown.withValues(alpha: 0.4)
                : Colors.grey.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.description_outlined, size: 18, color: Colors.brown),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Text("White-paper mode",
                        style: TextStyle(
                            fontSize: 13, fontWeight: FontWeight.w600)),
                    if (_rescanning) ...[
                      const SizedBox(width: 8),
                      const SizedBox(
                        width: 12,
                        height: 12,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.brown),
                      ),
                    ],
                  ],
                ),
                Text(
                  _rescanning
                      ? "Re-scanning this image…"
                      : (_whitePaper
                          ? "Keeps thin-pen marks. Re-scan of the same image."
                          : "Standard pipeline. Flip to re-scan without noise cleanup."),
                  style: const TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
          ),
          Switch(
            value: _whitePaper,
            activeThumbColor: Colors.brown,
            onChanged: _rescanning ? null : (v) => _rescan(whitePaper: v),
          ),
        ],
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
