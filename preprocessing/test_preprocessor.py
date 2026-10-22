"""
Comprehensive Unit Tests for Shared QR Preprocessing and Rectification Layer.

Covers all 9 mandated test cases:
1. Clean Version 3-M QR from dataset
2. ECC-tampered Version 3-M sample from dataset
3. Visual-anomaly Version 3-M sample from dataset
4. Valid QR rotated by 90 degrees
5. Valid QR surrounded by additional wide margins
6. Perspective-distorted QR (synthetic homography)
7. Invalid / blank image
8. Unsupported QR version (Version 1 / Version 2)
9. QR detected with localized corners but decoding failed
"""

import unittest
import os
import sys
import cv2
import numpy as np
import qrcode

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.qr_preprocessor import (
    QRPreprocessor,
    preprocess_qr_image,
    PreprocessingResult
)
from app.inspector import inspect_image


class TestQRPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = QRPreprocessor(target_size=370)

    @staticmethod
    def _create_v3_qr(payload: str = "https://example.com/test_v3") -> np.ndarray:
        qr = qrcode.QRCode(
            version=3,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4
        )
        qr.add_data(payload)
        qr.make(fit=False)
        img = qr.make_image(fill_color="black", back_color="white")
        return np.array(img.convert("L"))

    def test_1_clean_v3_qr_dataset(self):
        """1. Existing clean Version 3-M QR from dataset retains identity and clean verdict."""
        img_path = "data/val/genuine/genuine_001.png"
        self.assertTrue(os.path.exists(img_path), f"Sample {img_path} must exist.")

        prep = self.preprocessor.preprocess(image_path=img_path)
        self.assertTrue(prep.success)
        self.assertEqual(prep.status, "SUCCESS")
        self.assertEqual(prep.output_dimensions, (370, 370))
        self.assertTrue(prep.is_version_3)
        self.assertEqual(prep.detected_version, 3)
        self.assertIsNotNone(prep.corners)
        self.assertEqual(len(prep.corners), 4)

        # End-to-end inspection check
        res = inspect_image(preprocessed=prep)
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)
        self.assertFalse(res["tampering"]["peak_anomaly"]["triggered"])
        self.assertEqual(res["url"]["risk_level"], "none")

    def test_2_ecc_tampered_v3_dataset(self):
        """2. Existing ECC-tampered Version 3-M sample triggers ECC integrity violation."""
        img_path = "data/val/tampered/tampered_028.png"
        self.assertTrue(os.path.exists(img_path), f"Sample {img_path} must exist.")

        prep = self.preprocessor.preprocess(image_path=img_path)
        self.assertTrue(prep.success)
        self.assertEqual(prep.status, "SUCCESS")
        self.assertTrue(prep.is_version_3)

        res = inspect_image(preprocessed=prep)
        self.assertEqual(res["tampering"]["status"], "tampering_detected")
        self.assertIn("ecc_integrity_violation", res["tampering"]["evidence"])
        self.assertGreater(res["tampering"]["ecc"]["residual"], 0)

    def test_3_visual_anomaly_v3_dataset(self):
        """3. Existing visual-anomaly Version 3-M sample triggers peak edge gradient."""
        img_path = "data/val/tampered/tampered_065.png"
        self.assertTrue(os.path.exists(img_path), f"Sample {img_path} must exist.")

        prep = self.preprocessor.preprocess(image_path=img_path)
        self.assertTrue(prep.success)
        self.assertEqual(prep.status, "SUCCESS")
        self.assertTrue(prep.is_version_3)

        res = inspect_image(preprocessed=prep)
        self.assertEqual(res["tampering"]["status"], "tampering_detected")
        self.assertIn("local_visual_anomaly", res["tampering"]["evidence"])
        self.assertTrue(res["tampering"]["peak_anomaly"]["triggered"])
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)

    def test_4_rotated_qr_90_degrees(self):
        """4. Valid QR rotated by 90 degrees is rectified to upright orientation without false tampering."""
        base = self._create_v3_qr("https://example.com/rot_90")
        rot90 = cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)

        prep = self.preprocessor.preprocess(image=rot90)
        self.assertTrue(prep.success)
        self.assertEqual(prep.status, "SUCCESS")
        self.assertEqual(prep.output_dimensions, (370, 370))
        self.assertEqual(prep.decoded_payload, "https://example.com/rot_90")

        res = inspect_image(preprocessed=prep)
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)
        self.assertFalse(res["tampering"]["peak_anomaly"]["triggered"])

    def test_5_wide_margins_normalization(self):
        """5. Valid QR surrounded by additional wide margins (border=10) has quiet zone normalized."""
        qr_wide = qrcode.QRCode(
            version=3,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=10
        )
        qr_wide.add_data("https://example.com/margins_test")
        qr_wide.make(fit=False)
        wide_img = np.array(qr_wide.make_image(fill_color="black", back_color="white").convert("L"))
        self.assertEqual(wide_img.shape, (490, 490))

        prep = self.preprocessor.preprocess(image=wide_img)
        self.assertTrue(prep.success)
        self.assertEqual(prep.status, "SUCCESS")
        self.assertEqual(prep.output_dimensions, (370, 370))
        self.assertTrue(prep.has_normalized_quiet_zone)

        res = inspect_image(preprocessed=prep)
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)

    def test_6_perspective_distorted_qr(self):
        """6. Perspective-distorted QR code is rectified via homography to clean Version 3-M."""
        clean = self._create_v3_qr("https://example.com/distorted_v3")
        h, w = clean.shape
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[25, 30], [w - 15, 15], [w - 40, h - 25], [15, h - 35]])
        M = cv2.getPerspectiveTransform(src, dst)
        tilted = cv2.warpPerspective(clean, M, (w, h), borderValue=255)

        prep = self.preprocessor.preprocess(image=tilted)
        self.assertTrue(prep.success)
        self.assertEqual(prep.status, "SUCCESS")
        self.assertEqual(prep.output_dimensions, (370, 370))
        self.assertEqual(prep.decoded_payload, "https://example.com/distorted_v3")

        res = inspect_image(preprocessed=prep)
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)

    def test_7_invalid_or_blank_image(self):
        """7. Blank image and empty arrays return explicit failure states without exceptions."""
        blank_img = np.ones((400, 400), dtype=np.uint8) * 255
        prep_blank = self.preprocessor.preprocess(image=blank_img)
        self.assertFalse(prep_blank.success)
        self.assertEqual(prep_blank.status, "NO_QR_DETECTED")
        self.assertIsNone(prep_blank.rectified_image)

        res_blank = inspect_image(preprocessed=prep_blank)
        self.assertEqual(res_blank["tampering"]["status"], "detection_failed")
        self.assertEqual(res_blank["url"]["risk_level"], "unavailable")

        # Empty array
        empty_img = np.array([], dtype=np.uint8)
        prep_empty = self.preprocessor.preprocess(image=empty_img)
        self.assertFalse(prep_empty.success)
        self.assertEqual(prep_empty.status, "INVALID_IMAGE")

    def test_8_unsupported_qr_version(self):
        """8. Unsupported QR version (Version 1 / Version 2) is flagged as unsupported_geometry."""
        # Version 1 QR code
        qr1 = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4
        )
        qr1.add_data("https://v1.org")
        qr1.make(fit=False)
        img1 = np.array(qr1.make_image(fill_color="black", back_color="white").convert("L"))

        prep1 = self.preprocessor.preprocess(image=img1)
        self.assertTrue(prep1.success)
        self.assertEqual(prep1.status, "UNSUPPORTED_VERSION")
        self.assertFalse(prep1.is_version_3)
        self.assertEqual(prep1.detected_version, 1)

        res1 = inspect_image(preprocessed=prep1)
        self.assertEqual(res1["tampering"]["status"], "unsupported_geometry")
        self.assertEqual(res1["tampering"]["ecc"]["status"], "unsupported_version")
        self.assertEqual(res1["tampering"]["ecc"]["residual"], 0)
        # URL analysis remains independently operational
        self.assertEqual(res1["url"]["url"], "https://v1.org")
        self.assertEqual(res1["url"]["risk_level"], "none")

        # Real tilted Version 2 QR from data/raw/tilted_qr.png
        tilted_path = "data/raw/tilted_qr.png"
        self.assertTrue(os.path.exists(tilted_path))
        prep2 = self.preprocessor.preprocess(image_path=tilted_path)
        self.assertTrue(prep2.success)
        self.assertEqual(prep2.status, "UNSUPPORTED_VERSION")
        self.assertEqual(prep2.detected_version, 2)
        res2 = inspect_image(preprocessed=prep2)
        self.assertEqual(res2["tampering"]["status"], "unsupported_geometry")
        self.assertEqual(res2["url"]["url"], "https://example.com")

    def test_9_qr_detected_decode_failed(self):
        """9. Scrambled payload center with intact finder patterns produces DECODE_FAILED state."""
        base = self._create_v3_qr("https://example.com/scramble_test")
        corrupted = base.copy()
        # Corrupt center data area (110x110 px) while leaving all three 70x70 finder patterns intact
        corrupted[130:240, 130:240] = np.random.randint(0, 2, (110, 110), dtype=np.uint8) * 255

        prep = self.preprocessor.preprocess(image=corrupted)
        self.assertTrue(prep.success)
        self.assertEqual(prep.status, "DECODE_FAILED")
        self.assertIsNotNone(prep.corners)
        self.assertIsNotNone(prep.rectified_image)
        self.assertIsNone(prep.decoded_payload)

        res = inspect_image(preprocessed=prep)
        self.assertIn(res["tampering"]["status"], ["decode_failed", "tampering_detected"])
        self.assertEqual(res["url"]["risk_level"], "unavailable")


if __name__ == "__main__":
    unittest.main()
