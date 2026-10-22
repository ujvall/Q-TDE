"""
Unit Tests for Camera and Perspective-Correction Processing Layer (app/camera.py).

Tests pure/helper logic without requiring a physical webcam:
1. QR detection & perspective correction on standard synthetic QR
2. Perspective correction on tilted/warped QR (data/raw/tilted_qr.png)
3. Non-QR / blank frame handling
4. Overlay drawing helper
5. Camera failure state handling (invalid device ID)
6. End-to-end inspect_frame pipeline
"""

import unittest
import os
import sys
import numpy as np
import cv2
import qrcode

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.camera import QRCameraProcessor


class TestQRCameraProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = QRCameraProcessor(target_size=370)

    @staticmethod
    def _create_synthetic_qr(payload: str = "https://example.com/camera_test") -> np.ndarray:
        qr = qrcode.QRCode(
            version=3,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4
        )
        qr.add_data(payload)
        qr.make(fit=False)
        img = qr.make_image(fill_color="black", back_color="white")
        return np.array(img.convert("RGB"))

    def test_1_detect_and_correct_standard_qr(self):
        """Test localization and perspective warp on a standard synthetic frame."""
        frame = self._create_synthetic_qr()
        success, corrected, decoded, pts, msg = self.processor.detect_and_correct(frame)

        self.assertTrue(success)
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.shape, (370, 370))
        self.assertEqual(decoded, "https://example.com/camera_test")
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 4)

    def test_2_detect_and_correct_tilted_qr(self):
        """Test perspective warp on real distorted sample (data/raw/tilted_qr.png)."""
        tilted_path = "data/raw/tilted_qr.png"
        self.assertTrue(os.path.exists(tilted_path), f"{tilted_path} must exist.")

        frame = cv2.imread(tilted_path)
        success, corrected, decoded, pts, msg = self.processor.detect_and_correct(frame)

        self.assertTrue(success)
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.shape, (370, 370))
        self.assertEqual(decoded, "https://example.com")

    def test_3_detect_and_correct_blank_frame(self):
        """Test localization failure handling on empty frame."""
        blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        success, corrected, decoded, pts, msg = self.processor.detect_and_correct(blank_frame)

        self.assertFalse(success)
        self.assertIsNone(corrected)
        self.assertIsNone(decoded)
        self.assertIn("no qr code localized", msg.lower())

    def test_4_draw_overlay(self):
        """Test bounding box drawing on frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = np.float32([[10, 10], [100, 10], [100, 100], [10, 100]])
        annotated = self.processor.draw_overlay(frame, pts)

        self.assertEqual(annotated.shape, frame.shape)
        # Verify drawing occurred (some pixels should now be non-zero green)
        self.assertGreater(np.sum(annotated), 0)

    def test_5_capture_frame_invalid_device(self):
        """Test graceful failure when camera device is invalid (does not crash)."""
        success, frame, msg = self.processor.capture_frame(device_id=-999)
        self.assertFalse(success)
        self.assertIsNone(frame)
        self.assertIn("unavailable", msg.lower())

    def test_6_inspect_frame_pipeline_success(self):
        """Test end-to-end inspect_frame pipeline with a valid QR code."""
        frame = self._create_synthetic_qr("https://example.com/inspect_pipeline")
        res, crop, msg = self.processor.inspect_frame(frame)

        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (370, 370))
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(res["url"]["risk_level"], "none")
        self.assertEqual(res["overall"]["tampering_status"], "no_tampering_evidence_detected")

    def test_7_inspect_frame_pipeline_no_qr(self):
        """Test end-to-end inspect_frame pipeline with a blank frame."""
        blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        res, crop, msg = self.processor.inspect_frame(blank_frame)

        self.assertIsNone(crop)
        self.assertEqual(res["tampering"]["status"], "detection_failed")
        self.assertEqual(res["url"]["risk_level"], "unavailable")
        self.assertEqual(res["overall"]["tampering_status"], "detection_failed")


if __name__ == "__main__":
    unittest.main()
