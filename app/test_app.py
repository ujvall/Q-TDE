"""
Unit and Integration Tests for Application Layer (app/inspector.py).

Covers:
1. Valid genuine QR (from dataset)
2. ECC-detected tampered QR (from dataset)
3. Visual-anomaly tampered QR (from dataset)
4. Risky URL payload (via in-memory image inspection)
5. Invalid/missing image path
6. Blank / non-QR image
7. QRInspector class instantiation and functional inspect_image
"""

import unittest
import os
import sys
import numpy as np
import qrcode

# Ensure project root is present in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.inspector import inspect_image, QRInspector


class TestAppInspector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inspector = QRInspector()

    @staticmethod
    def _create_synthetic_qr(payload: str) -> np.ndarray:
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

    def test_1_valid_genuine_qr(self):
        """Test inspection of a valid genuine QR code from dataset."""
        img_path = "data/val/genuine/genuine_001.png"
        self.assertTrue(os.path.exists(img_path), f"Sample {img_path} must exist.")

        res = inspect_image(image_path=img_path)

        # Structure checks
        self.assertIn("tampering", res)
        self.assertIn("url", res)
        self.assertIn("overall", res)

        # Integrity checks
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(len(res["tampering"]["evidence"]), 0)
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)
        self.assertFalse(res["tampering"]["peak_anomaly"]["triggered"])

        # URL checks
        self.assertEqual(res["url"]["risk_level"], "none")
        self.assertEqual(res["overall"]["tampering_status"], "no_tampering_evidence_detected")
        self.assertEqual(res["overall"]["url_risk_level"], "none")

    def test_2_ecc_detected_tampered_qr(self):
        """Test inspection of an ECC integrity violation from dataset (tampered_028.png)."""
        img_path = "data/val/tampered/tampered_028.png"
        self.assertTrue(os.path.exists(img_path), f"Sample {img_path} must exist.")

        res = inspect_image(image_path=img_path)

        self.assertEqual(res["tampering"]["status"], "tampering_detected")
        self.assertIn("ecc_integrity_violation", res["tampering"]["evidence"])
        self.assertGreater(res["tampering"]["ecc"]["residual"], 0)
        self.assertEqual(res["overall"]["tampering_status"], "tampering_detected")

    def test_3_visual_anomaly_tampered_qr(self):
        """Test inspection of a visual anomaly from dataset (tampered_065.png)."""
        img_path = "data/val/tampered/tampered_065.png"
        self.assertTrue(os.path.exists(img_path), f"Sample {img_path} must exist.")

        res = inspect_image(image_path=img_path)

        self.assertEqual(res["tampering"]["status"], "tampering_detected")
        self.assertIn("local_visual_anomaly", res["tampering"]["evidence"])
        self.assertTrue(res["tampering"]["peak_anomaly"]["triggered"])
        self.assertGreater(res["tampering"]["peak_anomaly"]["score"], 0.525490)
        # Verify ECC was clean
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)

    def test_4_risky_url_destination(self):
        """Test inspection of a clean QR with an independently risky destination URL."""
        img = self._create_synthetic_qr("http://192.168.1.100/admin")
        res = self.inspector.inspect_image(image=img)

        # Tampering status remains independent and clean
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(len(res["tampering"]["evidence"]), 0)

        # URL risk is independently elevated
        self.assertEqual(res["url"]["risk_level"], "high")
        indicators = [i["indicator"] for i in res["url"]["risk_indicators"]]
        self.assertIn("IP_ADDRESS_HOSTNAME", indicators)
        self.assertIn("INSECURE_HTTP_SCHEME", indicators)
        self.assertEqual(res["overall"]["url_risk_level"], "high")

    def test_5_invalid_image_path(self):
        """Test inspection when given a non-existent image path."""
        res = inspect_image(image_path="non_existent_image_path_12345.png")

        self.assertEqual(res["tampering"]["status"], "unavailable")
        self.assertEqual(res["url"]["risk_level"], "unavailable")
        self.assertEqual(res["overall"]["tampering_status"], "unavailable")
        self.assertIn("not found", res["overall"]["inspection_summary"].lower())

    def test_6_blank_non_qr_image(self):
        """Test inspection when given a blank image without any QR code."""
        blank_img = np.ones((370, 370), dtype=np.uint8) * 255
        res = self.inspector.inspect_image(image=blank_img)

        self.assertEqual(res["tampering"]["status"], "detection_failed")
        self.assertEqual(res["url"]["risk_level"], "unavailable")
        self.assertEqual(res["overall"]["tampering_status"], "detection_failed")


if __name__ == "__main__":
    unittest.main()
