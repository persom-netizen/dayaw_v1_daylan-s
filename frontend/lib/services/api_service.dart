import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';

class ApiService {
  static String get _baseUrl {
    const override = String.fromEnvironment('API_BASE_URL');
    if (override.isNotEmpty) {
      return override;
    }

    if (kIsWeb) {
      return 'http://localhost:5000';
    }

    return 'http://10.10.248.60:5000';
  }

  /// --- UNIVERSAL TRANSLATION METHOD ---
  Future<Map<String, dynamic>?> uploadAndTranslateDetailed(
    XFile? imageFile,
    String mode, {
    String? text,
    Uint8List? imageBytes,
    bool whitePaper = false,
    bool visualize = false,
  }) async {
    try {
      final uri = Uri.parse('$_baseUrl/api/translate');
      final request = http.MultipartRequest('POST', uri);

      request.fields['mode'] = mode;
      request.fields['white_paper'] = whitePaper.toString();
      request.fields['visualize'] = visualize.toString();

      if (mode == 'Tagalog to Baybayin' && text != null) {
        request.fields['text'] = text;
      } else if (imageBytes != null || imageFile != null) {
        final bytes = imageBytes ?? await imageFile!.readAsBytes();
        final extension = imageFile != null
            ? imageFile.path.split('.').last.toLowerCase()
            : 'jpg';

        request.files.add(
          http.MultipartFile.fromBytes(
            'file',
            bytes,
            filename: 'upload.$extension',
            contentType: MediaType(
              'image',
              extension == 'png' ? 'png' : 'jpeg',
            ),
          ),
        );
      } else {
        print("❌ Error: No input data provided for mode: $mode");
        return null;
      }

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 30),
      );
      final response = await http.Response.fromStream(streamedResponse);

      print("===== TRANSLATE RESPONSE DEBUG =====");
      print("Status Code: ${response.statusCode}");
      print("Response Body: ${response.body}");
      print("===== END TRANSLATE DEBUG =====");

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        print("❌ Server Error: ${response.statusCode} - ${response.body}");
        return null;
      }
    } catch (e) {
      print("❌ General Error in uploadAndTranslateDetailed: $e");
      return null;
    }
  }

  /// --- BULK ARCHIVAL METHOD (FIXED) ---
  Future<bool> archiveBulkCharacters(
    List<Map<String, dynamic>> eligibleDetections,
    int sessionId,
  ) async {
    try {
      print("===== FLUTTER ARCHIVE DEBUG =====");
      print("Session ID: $sessionId");
      print("Eligible Detections: $eligibleDetections");

      final payload = {
        "session_id": sessionId,
        "detections": eligibleDetections
            .map(
              (d) => {
                "char": d['char']?.toString() ?? '',
                "confidence": d['confidence'] ?? 0.0,
                "is_eligible": d['is_eligible'] == true,
                "temp_path": d['temp_path']?.toString() ?? '',
              },
            )
            .toList(),
      };

      print("Payload Sent: ${jsonEncode(payload)}");

      final response = await http
          .post(
            Uri.parse('$_baseUrl/api/archive_bulk'),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 15));

      print("Archive Response Code: ${response.statusCode}");
      print("Archive Response Body: ${response.body}");
      print("===== END FLUTTER ARCHIVE DEBUG =====");

      return response.statusCode == 200;
    } catch (e) {
      print("❌ Bulk Archival API Error: $e");
      return false;
    }
  }

  /// --- LEGACY/SINGLE ARCHIVAL ---
  Future<bool> archiveCharacter(
    String char,
    double confidence,
    int sessionId,
  ) async {
    try {
      final response = await http
          .post(
            Uri.parse('$_baseUrl/api/archive_character'),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({
              "char": char,
              "confidence": confidence,
              "session_id": sessionId,
            }),
          )
          .timeout(const Duration(seconds: 10));

      return response.statusCode == 200;
    } catch (e) {
      print("❌ Archival API Error: $e");
      return false;
    }
  }

  /// --- HELPERS ---
  Future<String> translateTagalogToBaybayin(String tagalogText) async {
    final data = await uploadAndTranslateDetailed(
      null,
      'Tagalog to Baybayin',
      text: tagalogText,
    );
    return data?['translated_text']?.toString() ?? "Translation failed";
  }

  Future<String> uploadAndTranslate(XFile imageFile, String mode) async {
    final data = await uploadAndTranslateDetailed(imageFile, mode);
    return data?['translated_text']?.toString() ?? "No translation found";
  }
}
