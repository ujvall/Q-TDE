"""
Application Service Layer for QR Code Inspection.

Provides a unified interface for inspecting QR codes from file paths or memory buffers,
delegating QR localization, perspective correction, and quiet-zone normalization to the
shared preprocessing pipeline before delegating to the HierarchicalQRDecisionEngine.
"""

import os
import sys
from typing import Dict, Any, Optional
import numpy as np

# Ensure project root is present in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fusion.decision_engine import (
    HierarchicalQRDecisionEngine,
    calculate_peak_high_frequency_anomaly,
    PEAK_ANOMALY_THRESHOLD
)
from preprocessing.qr_preprocessor import (
    preprocess_qr_image,
    PreprocessingResult
)


class QRInspector:
    """
    Service wrapper integrating shared preprocessing with HierarchicalQRDecisionEngine.
    Provides decoupled inspection for CLI, GUI, and API entry points.
    """

    def __init__(self, engine: Optional[HierarchicalQRDecisionEngine] = None):
        self.engine = engine or HierarchicalQRDecisionEngine()

    def inspect_image(
        self,
        image_path: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        manual_payload: Optional[str] = None,
        preprocessed: Optional[PreprocessingResult] = None
    ) -> Dict[str, Any]:
        """
        Inspect a QR code image and return the complete structured inspection result.

        Parameters:
            image_path: Filesystem path to the image file.
            image: Optional raw numpy array (grayscale or BGR).
            manual_payload: Optional pre-decoded payload if available.
            preprocessed: Optional PreprocessingResult if already rectified upstream.

        Returns:
            Structured dictionary matching HierarchicalQRDecisionEngine schema:
            - tampering: {status, evidence, ecc, peak_anomaly, summary}
            - url: {url, risk_level, risk_score, risk_indicators, summary}
            - overall: {tampering_status, url_risk_level, inspection_summary}
            - preprocessing: {success, status, message, corners, detected_version, ...}
        """
        # Step 1: Execute shared preprocessing if not already performed
        if preprocessed is not None:
            prep = preprocessed
        else:
            prep = preprocess_qr_image(
                image=image,
                image_path=image_path,
                target_size=370
            )

        # Step 2: Handle preprocessing failure states
        if not prep.success:
            if prep.status == "INVALID_IMAGE":
                error_type = "FILE_NOT_FOUND" if (image_path and not os.path.exists(image_path)) else "INVALID_IMAGE"
                res = self.engine._build_failure_result(error_type, prep.message)
            else:
                # NO_QR_DETECTED or PERSPECTIVE_WARP_FAILED
                res = self.engine._build_failure_result("NO_QR_DETECTED", prep.message)
                res["tampering"]["status"] = "detection_failed"
                res["overall"]["tampering_status"] = "detection_failed"
                res["overall"]["inspection_summary"] = f"Physical Integrity: {prep.message} | URL Security: Unavailable."

            res["preprocessing"] = prep.to_dict()
            return res

        # Step 3: Handle unsupported QR versions / geometries (not Version 3-M)
        if prep.status == "UNSUPPORTED_VERSION":
            payload = manual_payload or prep.decoded_payload or ""

            # Independent URL analysis for the decoded payload
            if payload and self.engine._is_likely_url(payload):
                url_result = self.engine.url_analyzer.analyze(payload)
            elif payload:
                url_result = {
                    "url": payload,
                    "scheme": "",
                    "hostname": "",
                    "is_https": False,
                    "is_ip_address": False,
                    "url_length": len(payload),
                    "domain_length": 0,
                    "subdomain_count": 0,
                    "query_parameter_count": 0,
                    "uses_punycode": False,
                    "uses_shortener": False,
                    "risk_indicators": [],
                    "risk_score": 0,
                    "risk_level": "non_url_payload",
                    "summary": "Decoded payload is plain text or non-URL data; destination risk analysis not applicable."
                }
            else:
                url_result = {
                    "url": "",
                    "risk_level": "unavailable",
                    "risk_score": 0,
                    "risk_indicators": [],
                    "summary": "QR code decode failed; no URL available."
                }

            # Peak anomaly analysis on the rectified crop
            peak_score = 0.0
            peak_triggered = False
            peak_status = "unavailable"
            if prep.rectified_image is not None:
                try:
                    peak_score = calculate_peak_high_frequency_anomaly(prep.rectified_image)
                    peak_triggered = (peak_score > self.engine.peak_threshold)
                    peak_status = "anomaly_detected" if peak_triggered else "within_normal_limits"
                except Exception:
                    peak_status = "analysis_failed"

            tampering_evidence = []
            if peak_triggered:
                tampering_evidence.append("local_visual_anomaly")
                tampering_status = "tampering_detected"
                tampering_summary = f"Visual anomaly detected on unsupported QR geometry (Peak: {round(peak_score, 4)})."
            else:
                tampering_status = "unsupported_geometry"
                tampering_summary = (
                    f"Physical integrity analysis unsupported: QR Version {prep.detected_version} detected; "
                    f"current ECC analysis requires Version 3-M."
                )

            overall_summary = (
                f"Physical Integrity: {tampering_summary} | "
                f"URL Security: {url_result['summary']}"
            )

            return {
                "tampering": {
                    "status": tampering_status,
                    "evidence": tampering_evidence,
                    "ecc": {
                        "module_diff": 0,
                        "nonzero_syndromes": 0,
                        "residual": 0,
                        "status": "unsupported_version"
                    },
                    "peak_anomaly": {
                        "score": round(peak_score, 6),
                        "threshold": self.engine.peak_threshold,
                        "triggered": peak_triggered,
                        "status": peak_status
                    },
                    "summary": tampering_summary
                },
                "url": url_result,
                "overall": {
                    "tampering_status": tampering_status,
                    "url_risk_level": url_result["risk_level"],
                    "inspection_summary": overall_summary
                },
                "preprocessing": prep.to_dict()
            }

        # Step 4: Handle decode failure with localized corners
        if prep.status == "DECODE_FAILED":
            res = self.engine.inspect(
                image=prep.rectified_image,
                manual_payload=manual_payload
            )
            res["preprocessing"] = prep.to_dict()
            return res

        # Step 5: Standard Version 3-M execution on rectified image
        effective_payload = manual_payload or prep.decoded_payload
        res = self.engine.inspect(
            image=prep.rectified_image,
            manual_payload=effective_payload
        )
        res["preprocessing"] = prep.to_dict()
        return res


# Global singleton instance for functional usage
_global_inspector: Optional[QRInspector] = None


def get_inspector() -> QRInspector:
    """Returns the shared global QRInspector instance."""
    global _global_inspector
    if _global_inspector is None:
        _global_inspector = QRInspector()
    return _global_inspector


def inspect_image(
    image_path: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    manual_payload: Optional[str] = None,
    preprocessed: Optional[PreprocessingResult] = None
) -> Dict[str, Any]:
    """
    Main application entry point for QR inspection.

    Usage:
        result = inspect_image("data/val/genuine/genuine_001.png")
    """
    return get_inspector().inspect_image(
        image_path=image_path,
        image=image,
        manual_payload=manual_payload,
        preprocessed=preprocessed
    )
