import 'package:flutter/material.dart';
import '../services/api_service.dart';

class EvaluationModal extends StatefulWidget {
  final List<dynamic> detections;
  final double averageConfidence;
  final String translatedText;
  final int sessionId;

  const EvaluationModal({
    super.key,
    required this.detections,
    required this.averageConfidence,
    required this.translatedText,
    required this.sessionId,
  });

  @override
  State<EvaluationModal> createState() => _EvaluationModalState();
}

class _EvaluationModalState extends State<EvaluationModal> {
  bool _isArchived = false;
  bool _isProcessing = false;

  /// --- 1. FILTER LOGIC (STRICT 23% THRESHOLD) ---
  /// Only detections >= 23% are shown in the UI and included in the result text.
  List<Map<String, dynamic>> get filteredDetections {
    return widget.detections
        .where((d) {
          if (d is! Map) return false;
          final conf = (d['confidence'] as num).toDouble();
          return conf >= 23.0; // STRICT: 23% and above only
        })
        .map((d) => Map<String, dynamic>.from(d as Map))
        .toList();
  }

  /// --- 2. RECALCULATE AVERAGE ---
  /// Recalculates the average based only on visible (>=23%) characters.
  double get filteredAverage {
    final list = filteredDetections;
    if (list.isEmpty) return 0.0;
    final total = list.fold(
      0.0,
      (sum, item) => sum + (item['confidence'] as num).toDouble(),
    );
    return total / list.length;
  }

  /// --- 3. REBUILD RESULT TEXT FROM FILTERED DETECTIONS ---
  /// Reassembles the translated text using only characters that pass the 23% filter.
  String get filteredResultText {
    final list = filteredDetections;
    if (list.isEmpty) return "—";
    return list.map((d) => d['char']?.toString() ?? '').join('');
  }

  Future<void> _handleBulkArchive(
    BuildContext context,
    List<Map<String, dynamic>> eligible,
  ) async {
    if (_isArchived || _isProcessing) return;
    setState(() => _isProcessing = true);

    final ApiService apiService = ApiService();
    bool success = await apiService.archiveBulkCharacters(
      eligible,
      widget.sessionId,
    );

    if (!mounted) return;

    setState(() {
      _isProcessing = false;
      if (success) _isArchived = true;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          success ? "Salamat! Data archived." : "Failed to archive.",
        ),
        backgroundColor: success ? Colors.green : Colors.red,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Only characters >= 23% shown in table and result
    final detectionsToShow = filteredDetections;

    // Archive list: characters that passed 23% filter AND are high-confidence (>=90%)
    final List<Map<String, dynamic>> eligibleForArchive = detectionsToShow
        .where((d) => (d['confidence'] as num).toDouble() >= 90.0)
        .toList();

    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.85,
      ),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(25)),
      ),
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40,
            height: 5,
            margin: const EdgeInsets.only(bottom: 10),
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(10),
            ),
          ),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Text(
                      "dayaw",
                      style: TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                        color: Colors.orange[700],
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),

                  const Text(
                    "Translation Result",
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 6),

                  // --- FIXED: Uses filteredResultText instead of widget.translatedText ---
                  Text(
                    filteredResultText,
                    style: const TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.brown,
                    ),
                  ),
                  const SizedBox(height: 20),

                  // Stats use filtered data only
                  _buildStatRow(
                    "Detected Characters:",
                    "${detectionsToShow.length}",
                  ),
                  _buildStatRow(
                    "Average Confidence:",
                    "${filteredAverage.toStringAsFixed(1)}%",
                  ),

                  const Divider(height: 40),

                  // --- TABLE: Only shows characters >= 23% ---
                  detectionsToShow.isEmpty
                      ? const Center(
                          child: Padding(
                            padding: EdgeInsets.all(20),
                            child: Text(
                              "No characters detected above 23% confidence.",
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.grey),
                            ),
                          ),
                        )
                      : Table(
                          defaultVerticalAlignment:
                              TableCellVerticalAlignment.middle,
                          children: [
                            const TableRow(
                              decoration: BoxDecoration(
                                border: Border(
                                  bottom: BorderSide(
                                    color: Colors.grey,
                                    width: 0.5,
                                  ),
                                ),
                              ),
                              children: [
                                Padding(
                                  padding: EdgeInsets.all(8.0),
                                  child: Text(
                                    "Char",
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.all(8.0),
                                  child: Text(
                                    "Conf.",
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.all(8.0),
                                  child: Text(
                                    "Status",
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold),
                                  ),
                                ),
                              ],
                            ),
                            ...detectionsToShow.map((d) => _buildTableRow(d)),
                          ],
                        ),

                  const SizedBox(height: 30),

                  if (eligibleForArchive.isNotEmpty)
                    _buildArchivePermissionCard(context, eligibleForArchive)
                  else
                    const Center(
                      child: Text(
                        "No high-confidence characters eligible for archival.",
                        textAlign: TextAlign.center,
                        style: TextStyle(color: Colors.grey, fontSize: 12),
                      ),
                    ),

                  const SizedBox(height: 30),
                  _buildLearningTip(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  TableRow _buildTableRow(Map<String, dynamic> d) {
    final double conf = (d['confidence'] as num).toDouble();
    final bool isExcellent = conf >= 90.0;

    return TableRow(
      children: [
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(
            d['char']?.toString() ?? '?',
            style: const TextStyle(fontSize: 18),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text("${conf.toStringAsFixed(1)}%"),
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(
            isExcellent ? "Excellent" : "Good",
            style: TextStyle(
              color: isExcellent ? Colors.green : Colors.blueGrey,
              fontWeight: FontWeight.bold,
              fontSize: 11,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text(label),
          const SizedBox(width: 10),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildArchivePermissionCard(
    BuildContext context,
    List<Map<String, dynamic>> eligible,
  ) {
    final bool archived = _isArchived;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: archived ? Colors.grey[100] : Colors.orange[50],
        borderRadius: BorderRadius.circular(15),
        border: Border.all(
          color: archived ? Colors.grey[300]! : Colors.orange[200]!,
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Icon(
                archived ? Icons.cloud_done : Icons.volunteer_activism,
                color: archived ? Colors.grey : Colors.orange[800],
              ),
              const SizedBox(width: 10),
              Text(
                archived ? "Data Saved to Archive" : "Help Dayaw Grow",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: archived ? Colors.grey : Colors.black,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          if (!archived)
            Text(
              "We detected ${eligible.length} high-quality strokes. Permit us to save them?",
            ),
          const SizedBox(height: 15),
          ElevatedButton.icon(
            onPressed: (archived || _isProcessing)
                ? null
                : () => _handleBulkArchive(context, eligible),
            icon: _isProcessing
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : Icon(archived ? Icons.check : Icons.check_circle_outline),
            label: Text(
              archived ? "Archived Successfully" : "Archive All Eligible Strokes",
            ),
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

  Widget _buildLearningTip() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(10),
      ),
      child: const Text(
        "Tip: Clear handwriting improves AI learning!",
        style: TextStyle(fontSize: 12),
      ),
    );
  }
}