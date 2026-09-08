import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../services/scan_logger.dart';
import '../widgets/image_cropper_widget.dart';
import '../widgets/scan_guide.dart';
import 'btl_result_screen.dart';

/// "Baybayin to Tagalog" mode: capture / upload a photo, crop it, send it for
/// translation, then push a full result page (image with bounding boxes +
/// scores + tap-to-correct).
class BaybayinToTagalogView extends StatefulWidget {
  const BaybayinToTagalogView({super.key});

  @override
  State<BaybayinToTagalogView> createState() => _BaybayinToTagalogViewState();
}

class _BaybayinToTagalogViewState extends State<BaybayinToTagalogView> {
  final ApiService _apiService = ApiService();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _lastImage;
  bool _isLoading = false;
  String _status = "";

  Future<Size?> _decodeSize(Uint8List bytes) async {
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      return Size(frame.image.width.toDouble(), frame.image.height.toDouble());
    } catch (_) {
      return null;
    }
  }

  Future<void> _scan(Uint8List croppedBytes) async {
    setState(() {
      _isLoading = true;
      _status = "Processing image…";
      _lastImage = croppedBytes;
    });

    final size = await _decodeSize(croppedBytes);
    final response = await _apiService.uploadAndTranslateDetailed(
      null,
      'Baybayin to Tagalog',
      imageBytes: croppedBytes,
    );
    if (!mounted) return;

    if (response == null) {
      setState(() {
        _isLoading = false;
        _status = "Error: connection failed.";
      });
      return;
    }

    final detections =
        (response['individual_detections'] as List? ?? []).whereType<Map>();
    final status = response['status']?.toString().toLowerCase() ?? '';

    if (detections.isEmpty || status == 'no_characters') {
      setState(() {
        _isLoading = false;
        _status = "No Baybayin letters found. Try a clearer crop.";
      });
      return;
    }

    String? scanId;
    scanId = await ScanLogger.instance
        .logScan(imageBytes: croppedBytes, response: response);
    if (!mounted) return;

    setState(() {
      _isLoading = false;
      _status = "";
    });

    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => BtlResultScreen(
        imageBytes: croppedBytes,
        imageSize: size,
        response: response,
        scanId: scanId,
      ),
    ));
  }

  Future<void> _selectAndCropImage(ImageSource source) async {
    // Keep plenty of resolution: a downscaled paragraph leaves each glyph too
    // small for the segmentation / OCR.
    final XFile? photo = await _picker.pickImage(
      source: source,
      imageQuality: 95,
      maxWidth: 2400,
      maxHeight: 2400,
    );
    if (photo == null) return;

    final bytes = await photo.readAsBytes();
    if (!mounted) return;

    final Uint8List? croppedBytes = await Navigator.of(context).push<Uint8List>(
      MaterialPageRoute(builder: (_) => ImageCropperScreen(imageData: bytes)),
    );
    if (croppedBytes == null || !mounted) return;

    await _scan(croppedBytes);
  }

  Widget _buildImageArea() {
    return Stack(
      children: [
        Center(
          child: _lastImage != null
              ? Image.memory(_lastImage!, fit: BoxFit.contain)
              : GestureDetector(
                  onTap: () => showScanGuide(context),
                  child: Padding(
                    padding: const EdgeInsets.all(24.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.document_scanner,
                            size: 64, color: Colors.brown),
                        const SizedBox(height: 12),
                        const Text(
                          "Upload or scan a document containing Baybayin scripts to transcribe",
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Colors.grey, fontSize: 14),
                        ),
                        const SizedBox(height: 10),
                        Text("Tap for scan tips",
                            style: TextStyle(
                                color: Colors.brown[300],
                                fontSize: 12,
                                fontWeight: FontWeight.w600)),
                      ],
                    ),
                  ),
                ),
        ),
        if (_isLoading)
          Positioned.fill(
            child: ColoredBox(
              color: Colors.black.withValues(alpha: 0.24),
              child: const Center(
                child: Card(
                  child: Padding(
                    padding: EdgeInsets.all(20),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(color: Colors.brown),
                        SizedBox(height: 12),
                        Text("Processing text algorithm…",
                            style: TextStyle(fontWeight: FontWeight.w500)),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          )
        else if (_status.isNotEmpty)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(12),
              color: Colors.black54,
              child: Text(
                _status,
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
            ),
          ),
      ],
    );
  }

  Widget _circleButton(IconData icon, String label, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.brown.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, size: 28, color: Colors.brown),
          ),
          const SizedBox(height: 8),
          Text(label,
              style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                  color: Colors.black87)),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 20),
            decoration: BoxDecoration(
              color: const Color(0xFFF5F5F5),
              borderRadius: BorderRadius.circular(15),
            ),
            clipBehavior: Clip.antiAlias,
            child: _buildImageArea(),
          ),
        ),
        const SizedBox(height: 24),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 30),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _circleButton(Icons.photo_library, "Gallery",
                  () => _selectAndCropImage(ImageSource.gallery)),
              const SizedBox(width: 40),
              _circleButton(Icons.camera_alt, "Camera",
                  () => _selectAndCropImage(ImageSource.camera)),
            ],
          ),
        ),
        const SizedBox(height: 6),
        TextButton.icon(
          onPressed: () => showScanGuide(context),
          icon: const Icon(Icons.help_outline, size: 18),
          label: const Text("How to scan"),
          style: TextButton.styleFrom(foregroundColor: Colors.brown),
        ),
      ],
    );
  }
}
