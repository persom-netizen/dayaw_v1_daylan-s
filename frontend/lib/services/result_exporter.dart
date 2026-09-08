import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// One card in the "Visualize process" strip, ready to render into the sheet.
class ExportVizCard {
  final String part; // "PART 1 - find the glyphs", ...
  final String title;
  final String caption;
  final Uint8List bytes; // JPEG from the backend

  const ExportVizCard(
      {required this.part,
      required this.title,
      required this.caption,
      required this.bytes});
}

/// Everything needed to render the "Save scanned results" JPG. Kept as a plain
/// value object so the whole build can run in a background isolate via
/// `compute(buildResultSheet, payload)`.
class ExportPayload {
  final String heading;
  final String timestamp;
  final String title; // transliteration (title-cased)
  final String plain; // lower-case result line
  final List<String> statLines;
  final Uint8List? processed; // the "flattened" frame JPEG
  final int procW; // pixel size the bbox coords refer to
  final int procH;
  final List<List<int>> boxesPx; // [x, y, w, h] in proc space
  final List<String> boxLabels;
  final List<ExportVizCard> viz;

  const ExportPayload({
    required this.heading,
    required this.timestamp,
    required this.title,
    required this.plain,
    required this.statLines,
    required this.processed,
    required this.procW,
    required this.procH,
    required this.boxesPx,
    required this.boxLabels,
    required this.viz,
  });
}

const int _sheetW = 900;
const int _margin = 10;
const int _contentW = _sheetW - 2 * _margin;

final _brown = img.ColorRgb8(121, 85, 72);
final _black = img.ColorRgb8(30, 30, 30);
final _grey = img.ColorRgb8(120, 120, 120);
final _white = img.ColorRgb8(255, 255, 255);
final _rule = img.ColorRgb8(210, 205, 200);
final _box = img.ColorRgb8(40, 155, 40);

int _wrappedHeight(String text, {required bool big}) {
  final cpl = big ? 42 : 78; // chars per line at width ~980
  final lineH = big ? 30 : 19;
  var lines = 0;
  for (final para in text.split('\n')) {
    lines += para.isEmpty ? 1 : (para.length / cpl).ceil();
  }
  return lines * lineH + 8;
}

img.Image _textBlock(String text,
    {required bool big, img.Color? color, int padTop = 4, int padBottom = 4}) {
  final h = _wrappedHeight(text, big: big) + padTop + padBottom;
  final block = img.Image(width: _contentW, height: h);
  img.fill(block, color: _white);
  img.drawString(block, text,
      font: big ? img.arial24 : img.arial14,
      x: 2,
      y: padTop,
      color: color ?? _black,
      wrap: true);
  return block;
}

img.Image _fittedImage(Uint8List jpeg, {int maxW = _contentW}) {
  final decoded = img.decodeImage(jpeg);
  if (decoded == null) return img.Image(width: maxW, height: 4);
  if (decoded.width <= maxW) return decoded;
  return img.copyResize(decoded, width: maxW);
}

img.Image _boxedProcessed(ExportPayload p) {
  final base = _fittedImage(p.processed!);
  final scale = base.width / (p.procW == 0 ? base.width : p.procW);
  for (var i = 0; i < p.boxesPx.length; i++) {
    final b = p.boxesPx[i];
    if (b.length < 4) continue;
    final x1 = (b[0] * scale).round();
    final y1 = (b[1] * scale).round();
    final x2 = ((b[0] + b[2]) * scale).round();
    final y2 = ((b[1] + b[3]) * scale).round();
    final label = i < p.boxLabels.length ? p.boxLabels[i] : '';
    img.drawRect(base, x1: x1, y1: y1, x2: x2, y2: y2, color: _box, thickness: 2);
    if (label.isNotEmpty) {
      final ly = (y1 - 15).clamp(0, base.height - 1).toInt();
      img.drawString(base, label,
          font: img.arial14, x: x1 + 1, y: ly, color: _box);
    }
  }
  return base;
}

img.Image _hRule() {
  final r = img.Image(width: _contentW, height: 13);
  img.fill(r, color: _white);
  img.drawLine(r, x1: 0, y1: 6, x2: _contentW - 1, y2: 6, color: _rule);
  return r;
}

/// Renders the whole result + the visualize-process strip into one tall JPEG.
Uint8List buildResultSheet(ExportPayload p) {
  final blocks = <img.Image>[];

  blocks.add(_textBlock(p.heading, big: true, color: _brown, padTop: 8));
  blocks.add(_textBlock(p.timestamp, big: false, color: _grey));
  blocks.add(_textBlock('Translation', big: false, color: _grey, padTop: 8));
  blocks.add(_textBlock(p.title.isEmpty ? '—' : p.title, big: true));
  blocks.add(_textBlock('Result: ${p.plain}', big: false));
  for (final s in p.statLines) {
    blocks.add(_textBlock(s, big: false));
  }
  blocks.add(_hRule());

  if (p.processed != null) {
    blocks.add(_textBlock('Detected characters on the processed frame',
        big: false, color: _grey, padTop: 6));
    blocks.add(_boxedProcessed(p));
    blocks.add(_hRule());
  }

  String? part;
  for (final card in p.viz) {
    if (card.part.isNotEmpty && card.part != part) {
      part = card.part;
      blocks.add(_hRule());
      blocks.add(_textBlock(card.part.toUpperCase(),
          big: false, color: _brown, padTop: 8));
    }
    blocks.add(_textBlock(card.title, big: true, color: _black, padTop: 8));
    blocks.add(_fittedImage(card.bytes));
    if (card.caption.isNotEmpty) {
      blocks.add(_textBlock(card.caption, big: false, color: _grey));
    }
  }

  final totalH = blocks.fold<int>(24, (s, b) => s + b.height + 6);
  final sheet = img.Image(width: _sheetW, height: totalH);
  img.fill(sheet, color: _white);
  var y = 12;
  for (final b in blocks) {
    img.compositeImage(sheet, b, dstX: _margin, dstY: y);
    y += b.height + 6;
  }

  return Uint8List.fromList(img.encodeJpg(sheet, quality: 86));
}
