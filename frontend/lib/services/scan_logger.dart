import 'dart:convert';
import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

/// On-device logger for Baybayin -> Tagalog scans, for field testing without a
/// backend. Each scan writes a folder under `dayaw_scan_logs/` in the app's
/// documents directory:
///
///   `<id>/image.png` - the cropped photo that was sent to the model
///   `<id>/scan.json` - the full response + any corrections the user made
///
/// "Export" zips the whole folder and opens the share sheet, so a tester can
/// email / AirDrop / save the logs off the phone. The user-corrected crops are
/// clean labelled training data for the next retrain.
///
/// Every method swallows its own errors — logging must never break a scan.
class ScanLogger {
  ScanLogger._();
  static final ScanLogger instance = ScanLogger._();

  static const _dirName = 'dayaw_scan_logs';
  Directory? _root;

  Future<Directory?> _dir() async {
    if (_root != null) return _root;
    try {
      final base = await getApplicationDocumentsDirectory();
      final d = Directory('${base.path}/$_dirName');
      if (!await d.exists()) await d.create(recursive: true);
      _root = d;
      return d;
    } catch (e) {
      debugPrint('ScanLogger: cannot open log dir: $e');
      return null;
    }
  }

  String _newId() {
    final now = DateTime.now();
    final ts = now.toIso8601String().replaceAll(RegExp(r'[:.]'), '-');
    return '${ts}_${now.microsecondsSinceEpoch % 100000}';
  }

  /// Records a scan. Returns the scan id (folder name) to pass to logCorrection,
  /// or null if logging is unavailable.
  Future<String?> logScan({
    required Uint8List imageBytes,
    required Map<String, dynamic> response,
  }) async {
    try {
      final root = await _dir();
      if (root == null) return null;
      final id = _newId();
      final folder = Directory('${root.path}/$id');
      await folder.create(recursive: true);

      await File('${folder.path}/image.png').writeAsBytes(imageBytes, flush: true);

      final record = <String, dynamic>{
        'id': id,
        'logged_at': DateTime.now().toIso8601String(),
        'translated_text': response['translated_text'],
        'confidence': response['confidence'],
        'status': response['status'],
        'session_id': response['session_id'],
        'processed_size': response['processed_size'],
        'deskew_angle': response['deskew_angle'],
        'individual_detections': response['individual_detections'],
        'corrections': <dynamic>[],
        'corrected_text': null,
      };
      await File('${folder.path}/scan.json')
          .writeAsString(const JsonEncoder.withIndent('  ').convert(record), flush: true);
      return id;
    } catch (e) {
      debugPrint('ScanLogger.logScan failed: $e');
      return null;
    }
  }

  /// Appends a user correction (glyph `index` changed `from` -> `to`) to a scan.
  Future<void> logCorrection({
    required String? scanId,
    required int index,
    required String from,
    required String to,
    List<dynamic>? alternatives,
  }) async {
    if (scanId == null) return;
    try {
      final root = await _dir();
      if (root == null) return;
      final f = File('${root.path}/$scanId/scan.json');
      if (!await f.exists()) return;
      final data = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      (data['corrections'] as List).add({
        'index': index,
        'from': from,
        'to': to,
        'alternatives': alternatives,
        'at': DateTime.now().toIso8601String(),
      });
      // rebuild corrected_text from the (edited) detections
      final dets = (data['individual_detections'] as List?) ?? const [];
      final buffers = <int, StringBuffer>{};
      for (var i = 0; i < dets.length; i++) {
        final d = dets[i] as Map;
        final ch = (i == index) ? to : (d['char']?.toString() ?? '');
        final line = (d['line'] as num?)?.toInt() ?? 0;
        final buf = buffers.putIfAbsent(line, () => StringBuffer());
        if (d['space_before'] == true && buf.isNotEmpty) buf.write(' ');
        buf.write(ch);
      }
      data['corrected_text'] = (buffers.keys.toList()..sort())
          .map((k) => buffers[k].toString().trim())
          .where((s) => s.isNotEmpty)
          .join('  |  ');
      // persist the char change too, so repeated corrections compound
      if (index >= 0 && index < dets.length) {
        (dets[index] as Map)['char'] = to;
      }
      await f.writeAsString(
          const JsonEncoder.withIndent('  ').convert(data), flush: true);
    } catch (e) {
      debugPrint('ScanLogger.logCorrection failed: $e');
    }
  }

  Future<int> scanCount() async {
    try {
      final root = await _dir();
      if (root == null) return 0;
      return root
          .listSync()
          .whereType<Directory>()
          .length;
    } catch (_) {
      return 0;
    }
  }

  /// Zips every logged scan and opens the share sheet. Returns false if there
  /// is nothing to export or the export failed.
  Future<bool> exportZip() async {
    try {
      final root = await _dir();
      if (root == null) return false;
      final entries = root.listSync().whereType<Directory>().toList();
      if (entries.isEmpty) return false;

      final encoder = ZipFileEncoder();
      final tmp = await getTemporaryDirectory();
      final ts = DateTime.now().toIso8601String().replaceAll(RegExp(r'[:.]'), '-');
      final zipPath = '${tmp.path}/dayaw_scan_logs_$ts.zip';
      encoder.create(zipPath);
      encoder.addDirectory(root, includeDirName: false);
      encoder.close();

      await Share.shareXFiles(
        [XFile(zipPath)],
        subject: 'DAYAW scan logs',
        text: 'DAYAW field-test scan logs (${entries.length} scans).',
      );
      return true;
    } catch (e) {
      debugPrint('ScanLogger.exportZip failed: $e');
      return false;
    }
  }

  Future<void> clear() async {
    try {
      final root = await _dir();
      if (root == null) return;
      if (await root.exists()) await root.delete(recursive: true);
      _root = null;
    } catch (e) {
      debugPrint('ScanLogger.clear failed: $e');
    }
  }
}
