import 'dart:typed_data';

import 'package:crop_your_image/crop_your_image.dart';
import 'package:flutter/foundation.dart' show compute;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

class ImageCropperScreen extends StatefulWidget {
  final Uint8List imageData;

  const ImageCropperScreen({super.key, required this.imageData});

  @override
  State<ImageCropperScreen> createState() => _ImageCropperScreenState();
}

/// A rotate / flip request, sent to a background isolate.
class _Xform {
  final Uint8List bytes;
  final String op; // rotL rotR flipH flipV
  const _Xform(this.bytes, this.op);
}

Uint8List _applyXform(_Xform r) {
  final im = img.decodeImage(r.bytes);
  if (im == null) return r.bytes;
  final img.Image out = switch (r.op) {
    'rotL' => img.copyRotate(im, angle: -90),
    'rotR' => img.copyRotate(im, angle: 90),
    'flipH' => img.flipHorizontal(im),
    'flipV' => img.flipVertical(im),
    _ => im,
  };
  return Uint8List.fromList(img.encodeJpg(out, quality: 92));
}

class _ImageCropperScreenState extends State<ImageCropperScreen> {
  final CropController controller = CropController();
  bool _isCropping = false;
  bool _busy = false;

  /// The image the cropper works on — replaced by each rotate / flip.
  late Uint8List _working = widget.imageData;

  void _onCropped(Uint8List croppedData) {
    Navigator.of(context).pop(croppedData);
  }

  Future<void> _transform(String op) async {
    if (_busy || _isCropping) return;
    setState(() => _busy = true);
    try {
      final result = await compute(_applyXform, _Xform(_working, op));
      if (!mounted) return;
      setState(() => _working = result);
    } catch (_) {
      // keep the current image on failure
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        elevation: 0,
        title:
            const Text('Adjust Border', style: TextStyle(color: Colors.white)),
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          TextButton(
            onPressed: (_isCropping || _busy)
                ? null
                : () {
                    setState(() => _isCropping = true);
                    controller.crop();
                  },
            child: const Text('Apply',
                style: TextStyle(color: Colors.white, fontSize: 16)),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Stack(
                children: [
                  Crop(
                    key: ValueKey(_working),
                    image: _working,
                    controller: controller,
                    onCropped: _onCropped,
                    withCircleUi: false,
                    baseColor: Colors.black,
                    maskColor: Colors.black.withValues(alpha: 0.4),
                    cornerDotBuilder: (size, edgeAlignment) => Container(
                      width: size,
                      height: size,
                      decoration: const BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                  if (_isCropping || _busy)
                    const Positioned.fill(
                      child: ColoredBox(
                        color: Colors.black54,
                        child: Center(
                            child:
                                CircularProgressIndicator(color: Colors.white)),
                      ),
                    ),
                ],
              ),
            ),
            _toolbar(),
          ],
        ),
      ),
    );
  }

  Widget _toolbar() {
    Widget btn(IconData icon, String label, String op) => Expanded(
          child: TextButton(
            onPressed: (_busy || _isCropping) ? null : () => _transform(op),
            style: TextButton.styleFrom(foregroundColor: Colors.white),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 22),
                const SizedBox(height: 2),
                Text(label, style: const TextStyle(fontSize: 11)),
              ],
            ),
          ),
        );

    return Container(
      color: const Color(0xFF111111),
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          btn(Icons.rotate_90_degrees_ccw, 'Rotate L', 'rotL'),
          btn(Icons.rotate_90_degrees_cw, 'Rotate R', 'rotR'),
          btn(Icons.flip, 'Flip H', 'flipH'),
          btn(Icons.flip_camera_android, 'Flip V', 'flipV'),
        ],
      ),
    );
  }
}
