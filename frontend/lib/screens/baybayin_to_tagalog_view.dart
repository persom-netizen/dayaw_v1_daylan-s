import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../widgets/image_cropper_widget.dart';
import '../widgets/evaluation_modal.dart';

/// Handles the "Baybayin to Tagalog" mode: capture/upload a photo, crop it,
/// send it for translation, and show the result. Fully self-contained —
/// owns its own state, independent of the text-translation mode.
class BaybayinToTagalogView extends StatefulWidget {
  const BaybayinToTagalogView({super.key});

  @override
  State<BaybayinToTagalogView> createState() => _BaybayinToTagalogViewState();
}

class _BaybayinToTagalogViewState extends State<BaybayinToTagalogView> {
  final ApiService _apiService = ApiService();
  final ImagePicker _picker = ImagePicker();

  String _translatedResult = "Result will appear here";
  bool _isLoading = false;
  Uint8List? _webImage;
  Size? _decodedImageSize;
  List<Map<String, dynamic>> _detectionBoxes = [];

  Future<void> _loadImageDimensions(Uint8List imageBytes) async {
    final codec = await ui.instantiateImageCodec(imageBytes);
    final frame = await codec.getNextFrame();
    if (!mounted) return;

    setState(() {
      _decodedImageSize = Size(
        frame.image.width.toDouble(),
        frame.image.height.toDouble(),
      );
    });
  }

  Future<void> _processCroppedImage(Uint8List imageBytes) async {
    setState(() {
      _isLoading = true;
      _translatedResult = 'Processing Image...';
    });

    await _loadImageDimensions(imageBytes);

    final response = await _apiService.uploadAndTranslateDetailed(
      null,
      'Baybayin to Tagalog',
      imageBytes: imageBytes,
    );

    if (!mounted) return;

    setState(() {
      _isLoading = false;
      if (response != null) {
        _translatedResult = response['translated_text'] ?? 'No result';
        _detectionBoxes = (response['individual_detections'] as List? ?? [])
            .whereType<Map>()
            .map((d) => Map<String, dynamic>.from(d))
            .toList();

        String status = response['status']?.toString().toLowerCase() ?? '';
        if (status == 'success' || status == 'low_confidence') {
          Future.delayed(const Duration(milliseconds: 500), () {
            if (mounted) _showEvaluation(response);
          });
        } else if (status == 'no_characters' || _translatedResult.isEmpty) {
          _translatedResult = 'No Baybayin letters found. Try a clearer crop.';
          _detectionBoxes = [];
        }
      } else {
        _translatedResult = 'Error: Connection Failed';
        _detectionBoxes = [];
      }
    });
  }

  Future<void> _selectAndCropImage(ImageSource source) async {
    final XFile? photo = await _picker.pickImage(
      source: source,
      imageQuality: 80,
      maxWidth: 700,
      maxHeight: 700,
    );
    if (photo == null) return;

    final bytes = await photo.readAsBytes();
    if (!mounted) return;

    final Uint8List? croppedBytes = await Navigator.of(context).push<Uint8List>(
      MaterialPageRoute(builder: (_) => ImageCropperScreen(imageData: bytes)),
    );

    if (croppedBytes == null) return;

    setState(() {
      _isLoading = true;
      _translatedResult = 'Processing Image...';
      _detectionBoxes = [];
      _decodedImageSize = null;
      // Send the raw cropped photo as-is — no client-side binarization.
      // app.py's Otsu-based pipeline is the single source of truth for
      // binarization and is tuned to handle lighting/shadows itself.
      _webImage = croppedBytes;
    });

    await _loadImageDimensions(croppedBytes);

    await _processCroppedImage(croppedBytes);
  }

  Future<void> _uploadFromGallery() async {
    await _selectAndCropImage(ImageSource.gallery);
  }

  Future<void> _captureFromCamera() async {
    await _selectAndCropImage(ImageSource.camera);
  }

  void _showEvaluation(Map<String, dynamic> data) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => EvaluationModal(
        detections: data['individual_detections'] ?? [],
        averageConfidence: (data['confidence'] as num).toDouble(),
        translatedText: data['translated_text'] ?? "",
        sessionId: data['session_id'] ?? 0,
      ),
    );
  }

  Widget _buildImageDisplay() {
    final imageContent = _webImage != null
        ? Image.memory(_webImage!, fit: BoxFit.contain)
        : Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                Icon(Icons.document_scanner, size: 64, color: Colors.brown),
                SizedBox(height: 12),
                Text(
                  "Upload or scan a document containing Baybayin scripts to transcribe",
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.grey, fontSize: 14),
                ),
              ],
            ),
          );

    return Stack(
      children: [
        Center(child: imageContent),
        if (_webImage != null && _detectionBoxes.isNotEmpty)
          Positioned.fill(
            child: IgnorePointer(
              child: CustomPaint(
                painter: DetectionOverlayPainter(
                  _detectionBoxes,
                  _decodedImageSize ?? Size.infinite,
                ),
              ),
            ),
          ),
        if (_isLoading)
          Positioned.fill(
            child: ColoredBox(
              color: Colors.black.withValues(alpha: 0.24),
              child: Center(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: const [
                        CircularProgressIndicator(color: Colors.brown),
                        SizedBox(height: 12),
                        Text(
                          "Processing text algorithm...",
                          style: TextStyle(fontWeight: FontWeight.w500),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          )
        else if (_webImage != null)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(12),
              color: Colors.black54,
              child: Text(
                _translatedResult,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildUploadWidget() {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.brown.withValues(alpha: 0.1),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.photo_library, size: 28, color: Colors.brown),
        ),
        const SizedBox(height: 8),
        const Text(
          "Gallery",
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
            color: Colors.black87,
          ),
        ),
      ],
    );
  }

  Widget _buildCameraWidget() {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.brown.withValues(alpha: 0.1),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.camera_alt, size: 28, color: Colors.brown),
        ),
        const SizedBox(height: 8),
        const Text(
          "Camera",
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
            color: Colors.black87,
          ),
        ),
      ],
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
            child: _buildImageDisplay(),
          ),
        ),
        const SizedBox(height: 30),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 30),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              GestureDetector(
                onTap: _uploadFromGallery,
                child: _buildUploadWidget(),
              ),
              const SizedBox(width: 40),
              GestureDetector(
                onTap: _captureFromCamera,
                child: _buildCameraWidget(),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class DetectionOverlayPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  final Size imageSize;

  DetectionOverlayPainter(this.detections, this.imageSize);

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.yellowAccent.withValues(alpha: 0.95);

    final fillPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.yellow.withValues(alpha: 0.08);

    for (final detection in detections) {
      final pixelBox = detection['bbox_px'] as List<dynamic>?;
      final normalizedBox = detection['bbox'] as List<dynamic>?;

      final box = pixelBox != null && pixelBox.length >= 4
          ? pixelBox
          : normalizedBox;

      if (box == null || box.length < 4) continue;

      final x = (box[0] as num).toDouble();
      final y = (box[1] as num).toDouble();
      final w = (box[2] as num).toDouble();
      final h = (box[3] as num).toDouble();

      if (imageSize.isFinite) {
        final rect = Rect.fromLTWH(
          (x / imageSize.width) * size.width,
          (y / imageSize.height) * size.height,
          (w / imageSize.width) * size.width,
          (h / imageSize.height) * size.height,
        );

        if (rect.width <= 0 || rect.height <= 0) continue;

        final insetX = rect.width * 0.05;
        final insetY = rect.height * 0.05;
        final tightRect = Rect.fromLTWH(
          rect.left + insetX,
          rect.top + insetY,
          (rect.width - insetX * 2).clamp(0.0, size.width),
          (rect.height - insetY * 2).clamp(0.0, size.height),
        );

        if (tightRect.width <= 0 || tightRect.height <= 0) continue;

        canvas.drawRect(tightRect, fillPaint);
        canvas.drawRect(tightRect, paint);
        continue;
      }

      final rect = Rect.fromLTWH(
        x * size.width,
        y * size.height,
        w * size.width,
        h * size.height,
      );

      final insetX = rect.width * 0.05;
      final insetY = rect.height * 0.05;
      final tightRect = Rect.fromLTWH(
        rect.left + insetX,
        rect.top + insetY,
        (rect.width - insetX * 2).clamp(0.0, size.width),
        (rect.height - insetY * 2).clamp(0.0, size.height),
      );

      if (tightRect.width <= 0 || tightRect.height <= 0) continue;

      canvas.drawRect(tightRect, fillPaint);
      canvas.drawRect(tightRect, paint);

      if (rect.width <= 0 || rect.height <= 0) continue;

      canvas.drawRect(rect, fillPaint);
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant DetectionOverlayPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.imageSize != imageSize;
  }
}
