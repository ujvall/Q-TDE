"""
Unit & Integration Tests for Hierarchical QR Decision Engine.

Covers all 7 mandated deterministic cases:
CASE 1: Clean QR + benign HTTPS URL
CASE 2: QR with ECC structural corruption
CASE 3: Visually anomalous QR that does not trigger ECC
CASE 4: Clean-looking QR with suspicious static URL
CASE 5: Tampering + suspicious URL
CASE 6: Invalid/malformed URL
CASE 7: QR decode failure / blank image
"""

import unittest
import numpy as np
import cv2
import qrcode
from fusion.decision_engine import HierarchicalQRDecisionEngine


class TestHierarchicalQRDecisionEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = HierarchicalQRDecisionEngine()

    @staticmethod
    def _create_clean_qr(payload: str) -> np.ndarray:
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

    def test_case_1_clean_qr_benign_url(self):
        """CASE 1: Clean QR + benign HTTPS URL"""
        img = self._create_clean_qr("https://example.com/page/10")
        res = self.engine.inspect(image=img)

        # Tampering checks
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(len(res["tampering"]["evidence"]), 0)
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)
        self.assertEqual(res["tampering"]["ecc"]["nonzero_syndromes"], 0)
        self.assertFalse(res["tampering"]["peak_anomaly"]["triggered"])
        self.assertLessEqual(res["tampering"]["peak_anomaly"]["score"], 0.525490)

        # URL checks
        self.assertEqual(res["url"]["risk_level"], "none")
        self.assertEqual(res["url"]["risk_score"], 0)
        self.assertEqual(len(res["url"]["risk_indicators"]), 0)

        # Overall checks
        self.assertEqual(res["overall"]["tampering_status"], "no_tampering_evidence_detected")
        self.assertEqual(res["overall"]["url_risk_level"], "none")

    def test_case_2_ecc_structural_corruption(self):
        """CASE 2: QR with ECC structural corruption (Pair 028 from validation set)"""
        res = self.engine.inspect(image_path="data/val/tampered/tampered_028.png")

        self.assertEqual(res["tampering"]["status"], "tampering_detected")
        self.assertIn("ecc_integrity_violation", res["tampering"]["evidence"])
        self.assertGreater(res["tampering"]["ecc"]["residual"], 0)
        self.assertEqual(res["tampering"]["ecc"]["nonzero_syndromes"], 26)
        self.assertEqual(res["tampering"]["ecc"]["status"], "integrity_violation_detected")

    def test_case_3_visual_anomaly_without_ecc(self):
        """CASE 3: Visually anomalous QR that does not trigger ECC (Pair 065 from validation set)"""
        res = self.engine.inspect(image_path="data/val/tampered/tampered_065.png")

        self.assertEqual(res["tampering"]["status"], "tampering_detected")
        self.assertIn("local_visual_anomaly", res["tampering"]["evidence"])
        self.assertTrue(res["tampering"]["peak_anomaly"]["triggered"])
        self.assertGreater(res["tampering"]["peak_anomaly"]["score"], 0.525490)
        # Verify ECC was clean
        self.assertEqual(res["tampering"]["ecc"]["residual"], 0)

    def test_case_4_clean_qr_suspicious_url(self):
        """CASE 4: Clean-looking QR with suspicious static URL (IP address + HTTP)"""
        img = self._create_clean_qr("http://192.168.1.50/login")
        res = self.engine.inspect(image=img)

        # Tampering status remains independent (clean)
        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(len(res["tampering"]["evidence"]), 0)

        # URL risk is independently elevated
        self.assertEqual(res["url"]["risk_level"], "high")
        self.assertGreaterEqual(res["url"]["risk_score"], 50)
        indicators = [i["indicator"] for i in res["url"]["risk_indicators"]]
        self.assertIn("IP_ADDRESS_HOSTNAME", indicators)
        self.assertIn("INSECURE_HTTP_SCHEME", indicators)

    def test_case_5_tampering_plus_suspicious_url(self):
        """CASE 5: Tampering + suspicious URL"""
        img = self._create_clean_qr("https://bit.ly/malicious-link")
        # Add visual tampering: spike noise to exceed peak threshold
        h, w = img.shape
        img[150:154, 150:154] = 255 - img[150:154, 150:154]

        res = self.engine.inspect(image=img)

        # Both dimensions independently triggered
        self.assertEqual(res["tampering"]["status"], "tampering_detected")
        self.assertIn(res["url"]["risk_level"], ["moderate", "high"])
        indicators = [i["indicator"] for i in res["url"]["risk_indicators"]]
        self.assertIn("URL_SHORTENER_SERVICE", indicators)

    def test_case_6_malformed_url(self):
        """CASE 6: Invalid/malformed URL"""
        img = self._create_clean_qr("javascript:alert(document.cookie)")
        res = self.engine.inspect(image=img)

        self.assertEqual(res["url"]["scheme"], "javascript")
        self.assertEqual(res["url"]["risk_level"], "high")
        indicators = [i["indicator"] for i in res["url"]["risk_indicators"]]
        self.assertIn("DANGEROUS_JAVASCRIPT_SCHEME", indicators)

    def test_case_7_qr_decode_failure(self):
        """CASE 7: Blank image / QR detection failure"""
        blank_img = np.ones((370, 370), dtype=np.uint8) * 255
        res = self.engine.inspect(image=blank_img)

        self.assertEqual(res["tampering"]["status"], "detection_failed")
        self.assertEqual(res["url"]["risk_level"], "unavailable")
        self.assertIn("detection failed", res["overall"]["inspection_summary"].lower())

    def test_case_8_non_url_plain_text_payload(self):
        """CASE 8: Plain text non-URL payload"""
        img = self._create_clean_qr("WIFI:S:OfficeGuest;T:WPA;P:SecretPass;;")
        res = self.engine.inspect(image=img)

        self.assertEqual(res["tampering"]["status"], "no_tampering_evidence_detected")
        self.assertEqual(res["url"]["risk_level"], "non_url_payload")
        self.assertIn("non-URL", res["url"]["summary"])

    def test_case_9_file_not_found(self):
        """CASE 9: Non-existent image path"""
        res = self.engine.inspect(image_path="non_existent_file_path_xyz.png")
        self.assertEqual(res["tampering"]["status"], "unavailable")
        self.assertEqual(res["url"]["risk_level"], "unavailable")


if __name__ == "__main__":
    unittest.main()

