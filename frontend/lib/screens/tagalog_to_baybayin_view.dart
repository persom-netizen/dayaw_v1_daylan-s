import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../services/api_service.dart';

/// Forces typed input to lower case (Baybayin has no case, and the backend
/// lowercases anyway — this keeps the on-screen text consistent for screenshots).
class _LowercaseFormatter extends TextInputFormatter {
  @override
  TextEditingValue formatEditUpdate(
      TextEditingValue oldValue, TextEditingValue newValue) {
    return newValue.copyWith(text: newValue.text.toLowerCase());
  }
}

/// Handles the "Tagalog to Baybayin" mode: debounced auto-translate as the
/// user types, with confidence display. Fully self-contained — owns its
/// own state, independent of the image-translation mode.
class TagalogToBaybayinView extends StatefulWidget {
  const TagalogToBaybayinView({super.key});

  @override
  State<TagalogToBaybayinView> createState() => _TagalogToBaybayinViewState();
}

class _TagalogToBaybayinViewState extends State<TagalogToBaybayinView> {
  final ApiService _apiService = ApiService();
  final TextEditingController _textController = TextEditingController();

  // Baybayin rendering, sized up 3x so the glyphs + diacritics are big enough
  // to read (and screenshot for the Baybayin->Tagalog test flow). The gaps
  // scale with the glyph so the word/char spacing ratio stays the same.
  static const double _glyphSize = 66.0;
  static const double _charGap = 6.0; // spacing between glyphs within a word
  static const double _wordGap = 90.0; // large, unmistakable gap between words
  static const double _lineHeight = 1.5;

  Timer? _debounce;

  String _translatedResult = "Result will appear here";
  double _confidenceScore = 0.0;
  bool _isLoading = false;

  @override
  void dispose() {
    _debounce?.cancel();
    _textController.dispose();
    super.dispose();
  }

  void _onTextChanged(String text) {
    if (_debounce?.isActive ?? false) _debounce!.cancel();
    _debounce = Timer(const Duration(milliseconds: 350), () {
      _handleTextTranslation(text);
    });
  }

  Future<void> _handleTextTranslation(String text) async {
    if (text.trim().isEmpty) {
      setState(() {
        _translatedResult = "Result will appear here";
        _confidenceScore = 0.0;
        _isLoading = false;
      });
      return;
    }

    setState(() => _isLoading = true);

    final response = await _apiService.uploadAndTranslateDetailed(
      null,
      'Tagalog to Baybayin',
      text: text,
    );

    if (!mounted) return;

    setState(() {
      _isLoading = false;
      if (response != null) {
        _translatedResult = response['translated_text'] ?? "No result";
        _confidenceScore = (response['confidence'] as num).toDouble();
      } else {
        _translatedResult = "Error: Connection Failed";
        _confidenceScore = 0.0;
      }
    });
  }

  Color _getConfidenceColor() {
    if (_confidenceScore >= 95) return Colors.green;
    if (_confidenceScore >= 75) return Colors.orange;
    return Colors.red;
  }

  /// Renders the translation with a clear, wider gap between words than between
  /// the glyphs inside a word (a plain string can't do that — `letterSpacing`
  /// hits every character equally).
  Widget _buildBaybayin() {
    final isPlaceholder = _translatedResult == "Result will appear here" ||
        _translatedResult == "No result" ||
        _translatedResult.startsWith("Error");
    if (isPlaceholder) {
      return Text(_translatedResult,
          style: const TextStyle(color: Colors.grey, fontSize: 14));
    }

    const style = TextStyle(
      fontFamily: 'BaybayinCustom',
      fontSize: _glyphSize,
      color: Colors.black87,
      height: _lineHeight,
      letterSpacing: _charGap,
    );

    final lines = _translatedResult.split('\n');
    final spans = <InlineSpan>[];
    for (var li = 0; li < lines.length; li++) {
      if (li > 0) spans.add(const TextSpan(text: '\n'));
      final words = lines[li]
          .trim()
          .split(RegExp(r'\s+'))
          .where((w) => w.isNotEmpty)
          .toList();
      for (var wi = 0; wi < words.length; wi++) {
        if (wi > 0) {
          spans.add(const WidgetSpan(
            alignment: PlaceholderAlignment.middle,
            child: SizedBox(width: _wordGap),
          ));
        }
        spans.add(TextSpan(text: words[wi]));
      }
    }
    return Text.rich(TextSpan(style: style, children: spans));
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F5F5),
        borderRadius: BorderRadius.circular(15),
      ),
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            // Compact, self-scrolling input so the Baybayin output gets most
            // of the card height.
            SizedBox(
              height: 96,
              child: TextField(
                controller: _textController,
                onChanged: _onTextChanged,
                inputFormatters: [_LowercaseFormatter()],
                expands: true,
                maxLines: null,
                minLines: null,
                keyboardType: TextInputType.multiline,
                textCapitalization: TextCapitalization.none,
                textAlignVertical: TextAlignVertical.top,
                decoration: const InputDecoration(
                  hintText: "Enter Tagalog text here...",
                  border: InputBorder.none,
                  isDense: true,
                ),
                style: const TextStyle(fontSize: 15),
              ),
            ),
            const Divider(height: 12, color: Colors.grey),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Text(
                      "Baybayin Translation:",
                      style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.grey,
                          fontSize: 13),
                    ),
                    if (_isLoading) ...[
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
                if (_confidenceScore > 0.0)
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: _getConfidenceColor().withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      "Confidence: ${_confidenceScore.toStringAsFixed(1)}%",
                      style: TextStyle(
                          color: _getConfidenceColor(),
                          fontWeight: FontWeight.bold,
                          fontSize: 12),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 6),
            Expanded(
              child: Align(
                alignment: Alignment.topLeft,
                child: SingleChildScrollView(
                  child: _buildBaybayin(),
                ),
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                IconButton(
                  visualDensity: VisualDensity.compact,
                  icon: const Icon(Icons.clear, color: Colors.grey),
                  onPressed: () {
                    _debounce?.cancel();
                    _textController.clear();
                    _handleTextTranslation("");
                  },
                  tooltip: "Clear",
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
