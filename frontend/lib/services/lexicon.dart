import 'package:flutter/services.dart' show rootBundle;

/// Optional Tagalog spell-snapping for a BTL result. Off by default - the user
/// turns it on in the result panel. Snaps each word to the nearest dictionary
/// entry within a small *weighted* edit distance, where the swaps the classifier
/// actually confuses (s/y, r/g, o/u, e/i, ...) cost less than a normal edit.
class Lexicon {
  Lexicon._();
  static final Lexicon instance = Lexicon._();

  final Set<String> _words = {};
  final Map<int, List<String>> _byLen = {};
  bool _loaded = false;
  bool get isLoaded => _loaded;

  // Baybayin-transliteration look-alikes: cheaper substitution cost.
  static const _pairs = <String>[
    'sy', 'rg', 'rd', 'gd', 'dg', 'nm', 'pm', 'mb', 'kg', 'td', 'bd',
    'ao', 'ae', 'ou', 'ei', 'eo', 'ai', 'ln', 'wy', 'yn',
  ];
  static final Map<String, double> _sub = () {
    final m = <String, double>{};
    for (final p in _pairs) {
      m['${p[0]}${p[1]}'] = 0.45;
      m['${p[1]}${p[0]}'] = 0.45;
    }
    return m;
  }();

  Future<void> load() async {
    if (_loaded) return;
    try {
      final raw = await rootBundle.loadString('assets/tagalog_words.txt');
      for (var w in raw.split('\n')) {
        w = w.trim().toLowerCase();
        if (w.isEmpty || w.startsWith('#')) continue;
        _words.add(w);
        _byLen.putIfAbsent(w.length, () => []).add(w);
      }
    } catch (_) {
      // no asset -> correction is just a no-op
    }
    _loaded = true;
  }

  double _cost(String a, String b) => a == b ? 0.0 : (_sub['$a$b'] ?? 1.0);

  double _dist(String a, String b, double cap) {
    final n = a.length, m = b.length;
    var prev = List<double>.generate(m + 1, (j) => j.toDouble());
    for (var i = 1; i <= n; i++) {
      final cur = List<double>.filled(m + 1, 0);
      cur[0] = i.toDouble();
      var rowMin = cur[0];
      for (var j = 1; j <= m; j++) {
        final sub = prev[j - 1] + _cost(a[i - 1], b[j - 1]);
        final del = prev[j] + 1;
        final ins = cur[j - 1] + 1;
        var best = sub < del ? sub : del;
        if (ins < best) best = ins;
        cur[j] = best;
        if (best < rowMin) rowMin = best;
      }
      if (rowMin > cap) return cap + 1;
      prev = cur;
    }
    return prev[m];
  }

  /// (corrected, changed). Only snaps a plain alphabetic word of length >= 2,
  /// within the weighted cap, and refuses when two candidates tie.
  (String, bool) correct(String word) {
    if (!_loaded || word.length < 2) return (word, false);
    final w = word.toLowerCase();
    if (!RegExp(r'^[a-z]+$').hasMatch(w)) return (word, false);
    if (_words.contains(w)) return (word, false);

    const cap = 1.5;
    String? best;
    var bestD = cap + 1.0;
    var ties = 0;
    for (var l = w.length - 2; l <= w.length + 2; l++) {
      for (final cand in _byLen[l] ?? const <String>[]) {
        final d = _dist(w, cand, cap);
        if (d < bestD - 1e-9) {
          bestD = d;
          best = cand;
          ties = 0;
        } else if ((d - bestD).abs() < 1e-9) {
          ties++;
        }
      }
    }
    if (best == null || bestD > cap || ties > 0) return (word, false);

    final fixed = word[0] == word[0].toUpperCase() && word[0] != word[0].toLowerCase()
        ? best[0].toUpperCase() + best.substring(1)
        : best;
    return (fixed, fixed.toLowerCase() != w);
  }

  /// Correct every whitespace-separated token; preserve the spacing and any
  /// non-word separators (e.g. the " | " line joiner) untouched.
  (String, int) correctPhrase(String phrase) {
    var changed = 0;
    final out = phrase.splitMapJoin(
      RegExp(r'\S+'),
      onMatch: (mm) {
        final (fixed, did) = correct(mm[0]!);
        if (did) changed++;
        return fixed;
      },
      onNonMatch: (s) => s,
    );
    return (out, changed);
  }
}
