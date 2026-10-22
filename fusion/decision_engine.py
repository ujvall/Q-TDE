"""
Hierarchical Decision Engine for QR Code Detection & Threat Analysis.

Maintains two independent evidence dimensions:
1. Physical / Structural Tampering Integrity (Reed-Solomon ECC + Classical Peak Anomaly)
2. Cyber / Destination URL Risk (StaticURLAnalyzer)

Deep Learning models (ViT, Autoencoder) are explicitly excluded from the active decision path
based on rigorous experimental characterization showing lack of reference-free discriminative capacity.
"""

import os
import cv2
import numpy as np
import qrcode
from typing import Dict, Any, Optional, Tuple, List
from url_analysis.url_analyzer import StaticURLAnalyzer


# ============================================================
# CONSTANTS & CONFIGURATION (QR Version 3-M Standards)
# ============================================================

VERSION = 3
MODULE_SIZE = 10
SIZE = 37
SYMBOL_SIZE = 29

TOTAL_CODEWORDS = 70
DATA_CODEWORDS = 44
ECC_CODEWORDS = 26
RS_BLOCKS = 1
BLOCK_TOTAL = 70
BLOCK_DATA = 44
BLOCK_ECC = 26

# Experimentally calibrated peak high-frequency edge gradient threshold
# Note: Calibrated specifically for 370x370 / 128x128 binary QR code generation setup.
PEAK_ANOMALY_THRESHOLD = 0.525490


# ============================================================
# GF(256) TABLES & GALOIS ARITHMETIC
# ============================================================

EXP_TABLE = [0] * 512
LOG_TABLE = [0] * 256

def _initialize_gf():
    value = 1
    for i in range(255):
        EXP_TABLE[i] = value
        LOG_TABLE[value] = i
        value <<= 1
        if value & 0x100:
            value ^= 0x11D
    for i in range(255, 512):
        EXP_TABLE[i] = EXP_TABLE[i - 255]

_initialize_gf()

def gf_multiply(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return EXP_TABLE[LOG_TABLE[a] + LOG_TABLE[b]]


# ============================================================
# QR CODEWORD EXTRACTION & ECC ANALYSIS
# ============================================================

def mask_condition(mask: int, row: int, col: int) -> bool:
    if mask == 0: return (row + col) % 2 == 0
    if mask == 1: return row % 2 == 0
    if mask == 2: return col % 3 == 0
    if mask == 3: return (row + col) % 3 == 0
    if mask == 4: return (row // 2 + col // 3) % 2 == 0
    if mask == 5: return (((row * col) % 2) + ((row * col) % 3)) == 0
    if mask == 6: return ((((row * col) % 2) + ((row * col) % 3)) % 2) == 0
    if mask == 7: return ((((row + col) % 2) + ((row * col) % 3)) % 2) == 0
    raise ValueError("Invalid QR mask")

def sample_modules(image: np.ndarray) -> np.ndarray:
    image_resized = cv2.resize(
        image,
        (SIZE * MODULE_SIZE, SIZE * MODULE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    modules = np.zeros((SIZE, SIZE), dtype=bool)
    for r in range(SIZE):
        for c in range(SIZE):
            y = r * MODULE_SIZE + MODULE_SIZE // 2
            x = c * MODULE_SIZE + MODULE_SIZE // 2
            modules[r, c] = (image_resized[y, x] < 128)
    return modules

def remove_quiet_zone(matrix: np.ndarray) -> np.ndarray:
    return matrix[4:4 + SYMBOL_SIZE, 4:4 + SYMBOL_SIZE]

def format_bits_from_matrix(matrix: np.ndarray) -> int:
    coordinates = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
        (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8)
    ]
    bits = 0
    for r, c in coordinates:
        bits <<= 1
        if matrix[r, c]:
            bits |= 1
    return bits

def decode_mask_pattern(matrix: np.ndarray) -> int:
    format_value = format_bits_from_matrix(matrix)
    format_data = format_value ^ 0x5412
    return (format_data >> 10) & 0x07

def build_functional_map() -> np.ndarray:
    functional = np.zeros((SYMBOL_SIZE, SYMBOL_SIZE), dtype=bool)
    finder_origins = [(0, 0), (0, SYMBOL_SIZE - 7), (SYMBOL_SIZE - 7, 0)]
    for origin_r, origin_c in finder_origins:
        for r in range(origin_r - 1, origin_r + 8):
            for c in range(origin_c - 1, origin_c + 8):
                if 0 <= r < SYMBOL_SIZE and 0 <= c < SYMBOL_SIZE:
                    functional[r, c] = True
    for c in range(8, SYMBOL_SIZE - 8): functional[6, c] = True
    for r in range(8, SYMBOL_SIZE - 8): functional[r, 6] = True
    for r in range(20, 25):
        for c in range(20, 25): functional[r, c] = True
    for c in range(9): functional[8, c] = True
    for r in range(9): functional[r, 8] = True
    for c in range(SYMBOL_SIZE - 8, SYMBOL_SIZE): functional[8, c] = True
    for r in range(SYMBOL_SIZE - 7, SYMBOL_SIZE): functional[r, 8] = True
    functional[SYMBOL_SIZE - 8, 8] = True
    return functional

def get_data_coordinates() -> List[Tuple[int, int]]:
    functional = build_functional_map()
    coordinates = []
    upward = True
    col = SYMBOL_SIZE - 1
    while col > 0:
        if col == 6: col -= 1
        rows = range(SYMBOL_SIZE - 1, -1, -1) if upward else range(SYMBOL_SIZE)
        for row in rows:
            for current_col in (col, col - 1):
                if not functional[row, current_col]:
                    coordinates.append((row, current_col))
        upward = not upward
        col -= 2
    return coordinates

def extract_codewords(matrix: np.ndarray) -> List[int]:
    symbol = remove_quiet_zone(matrix)
    mask = decode_mask_pattern(symbol)
    coordinates = get_data_coordinates()
    bits = []
    for row, col in coordinates:
        bit = symbol[row, col]
        if mask_condition(mask, row, col):
            bit = not bit
        bits.append(1 if bit else 0)
    codewords = []
    for i in range(TOTAL_CODEWORDS):
        value = 0
        for j in range(8):
            value <<= 1
            value |= bits[i * 8 + j]
        codewords.append(value)
    return codewords

def evaluate_reed_solomon_syndromes(codewords: List[int]) -> Tuple[int, int]:
    """
    Calculate 26 Reed-Solomon syndromes for single-block Version 3-M.
    Evaluates whether extracted codewords satisfy generator polynomial roots over GF(256).
    Returns (nonzero_syndromes_count, residual_sum).
    A residual > 0 indicates a Reed-Solomon integrity violation in the extracted codewords.
    """
    syndromes = []
    for syndrome_index in range(BLOCK_ECC):
        result = 0
        for position, codeword in enumerate(codewords):
            if codeword == 0:
                continue
            exponent = (syndrome_index * (BLOCK_TOTAL - 1 - position)) % 255
            coefficient = EXP_TABLE[exponent]
            result ^= gf_multiply(codeword, coefficient)
        syndromes.append(result)
    residual = sum(syndromes)
    nonzero = sum(val != 0 for val in syndromes)
    return nonzero, residual

def synthesize_reference_matrix(payload: str) -> np.ndarray:
    qr = qrcode.QRCode(
        version=VERSION,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=MODULE_SIZE,
        border=4
    )
    qr.add_data(payload)
    qr.make(fit=False)
    return np.array(qr.get_matrix(), dtype=bool)


# ============================================================
# PEAK HIGH-FREQUENCY ANOMALY
# ============================================================

def calculate_peak_high_frequency_anomaly(image: np.ndarray) -> float:
    """
    Computes maximum high-frequency edge residual:
    detail = abs(image - GaussianBlur(image, (5,5), 0)) / 255.0

    Measures local high-frequency edge gradients against calibrated threshold tau = 0.525490.
    Detects local high-frequency anomalies exceeding this calibrated threshold;
    does not constitute a universal detector for all visual tampering categories.
    """
    img_res = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img_res, (5, 5), 0)
    detail = cv2.absdiff(img_res, blur).astype(np.float32) / 255.0
    return float(np.max(detail))


# ============================================================
# HIERARCHICAL DECISION ENGINE
# ============================================================

class HierarchicalQRDecisionEngine:
    """
    Unified Hierarchical QR Inspection Engine.

    Maintains two strictly independent evidence dimensions:
    1. Physical / Structural Tampering Integrity:
       - Reed-Solomon ECC: evaluates whether extracted Version 3-M codewords produce
         non-zero syndromes, indicating a Reed-Solomon integrity violation.
       - Peak High-Frequency Anomaly: measures local edge gradient residual against
         calibrated threshold tau = 0.525490, detecting local high-frequency anomalies
         exceeding this threshold.
    2. Cyber / Destination URL Risk:
       - StaticURLAnalyzer: evaluates 21 structural risk indicator rules
         (emitting up to 23 distinct indicator identifiers) across scheme, authority,
         hostname, path, and query components.

    Independence Invariant:
    Physical tampering status and destination URL risk level are computed orthogonally.
    URL risk never alters tampering status, and tampering evidence never alters URL risk.
    """

    def __init__(self, peak_anomaly_threshold: float = PEAK_ANOMALY_THRESHOLD):
        self.peak_threshold = peak_anomaly_threshold
        self.url_analyzer = StaticURLAnalyzer()
        self.qr_detector = cv2.QRCodeDetector()
        self.aruco_detector = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else None

    @staticmethod
    def _is_likely_url(payload: str) -> bool:
        """Determines if a decoded string resembles a web URL or dangerous scheme."""
        p = payload.strip().lower()
        schemes = ("http://", "https://", "ftp://", "ftps://", "javascript:", "data:", "file:", "mailto:", "tel:")
        if any(p.startswith(s) for s in schemes):
            return True
        if " " not in p and "." in p:
            host_candidate = p.split("/")[0].split(":")[0]
            if "." in host_candidate and not host_candidate.startswith(".") and not host_candidate.endswith("."):
                return True
        return False

    def inspect(
        self,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        manual_payload: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main inspection entry point.
        Accepts raw grayscale/BGR image or image path.
        """
        # Load image if path provided
        if image is None and image_path is not None:
            if not os.path.exists(image_path):
                return self._build_failure_result("FILE_NOT_FOUND", f"Image file not found: {image_path}")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return self._build_failure_result("INVALID_IMAGE", "No valid image data provided for inspection.")

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------
        # STEP 1: QR DETECTION & DECODING
        # ------------------------------------------------------------
        decoded_payload = manual_payload
        qr_detected = True
        qr_decoded = True
        detection_points = None

        if decoded_payload is None:
            try:
                data, points, _ = self.qr_detector.detectAndDecode(image)
                detection_points = points
                qr_detected = (points is not None)
                qr_decoded = bool(data)
                decoded_payload = data if qr_decoded else None

                # If legacy detector failed to decode, try Aruco-based detector if available
                if not qr_decoded and self.aruco_detector is not None:
                    data_ar, points_ar, _ = self.aruco_detector.detectAndDecode(image)
                    if data_ar:
                        decoded_payload = data_ar
                        detection_points = points_ar
                        qr_detected = True
                        qr_decoded = True
                    elif points_ar is not None and not qr_detected:
                        detection_points = points_ar
                        qr_detected = True
            except Exception as e:
                qr_detected = False
                qr_decoded = False
                decoded_payload = None

        # ------------------------------------------------------------
        # STEP 2: PEAK HIGH-FREQUENCY ANOMALY ANALYSIS (Always runs if image valid)
        # ------------------------------------------------------------
        peak_score = 0.0
        peak_triggered = False
        peak_status = "unavailable"

        try:
            peak_score = calculate_peak_high_frequency_anomaly(image)
            peak_triggered = (peak_score > self.peak_threshold)
            peak_status = "anomaly_detected" if peak_triggered else "within_normal_limits"
        except Exception as e:
            peak_status = "analysis_failed"

        # ------------------------------------------------------------
        # STEP 3: REED-SOLOMON ECC STRUCTURAL ANALYSIS
        # ------------------------------------------------------------
        ecc_module_diff = 0
        ecc_nonzero = 0
        ecc_residual = 0
        ecc_triggered = False
        ecc_status = "unavailable"

        if decoded_payload:
            try:
                sampled = sample_modules(image)
                reference = synthesize_reference_matrix(decoded_payload)
                ecc_module_diff = int(np.sum(sampled != reference))
                codewords = extract_codewords(sampled)
                ecc_nonzero, ecc_residual = evaluate_reed_solomon_syndromes(codewords)
                
                ecc_triggered = (ecc_residual > 0 or ecc_nonzero > 0)
                ecc_status = "integrity_violation_detected" if ecc_triggered else "valid_codewords"
            except Exception as e:
                ecc_status = "extraction_failed"
        elif not qr_detected:
            ecc_status = "detection_failed"
        else:
            ecc_status = "decode_failed"

        # ------------------------------------------------------------
        # STEP 4: HIERARCHICAL TAMPERING VERDICT
        # ------------------------------------------------------------
        tampering_evidence: List[str] = []
        
        if ecc_triggered:
            tampering_evidence.append("ecc_integrity_violation")
        if peak_triggered:
            tampering_evidence.append("local_visual_anomaly")

        if not qr_detected:
            tampering_status = "detection_failed"
            tampering_summary = "QR code localization failed."
        elif not qr_decoded and not tampering_evidence:
            tampering_status = "decode_failed"
            tampering_summary = "QR code detected but could not be decoded; no visual anomaly above threshold."
        elif len(tampering_evidence) > 0:
            tampering_status = "tampering_detected"
            tampering_summary = f"Tampering detected via: {', '.join(tampering_evidence)}."
        else:
            tampering_status = "no_tampering_evidence_detected"
            tampering_summary = "No visual or structural tampering evidence detected."

        # ------------------------------------------------------------
        # STEP 5: URL DESTINATION RISK ANALYSIS (Independent)
        # ------------------------------------------------------------
        if decoded_payload:
            if not self._is_likely_url(decoded_payload):
                url_result = {
                    "url": decoded_payload,
                    "scheme": "",
                    "hostname": "",
                    "is_https": False,
                    "is_ip_address": False,
                    "url_length": len(decoded_payload),
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
                url_result = self.url_analyzer.analyze(decoded_payload)
        elif not qr_detected:
            url_result = {
                "url": "",
                "risk_level": "unavailable",
                "risk_score": 0,
                "risk_indicators": [],
                "summary": "QR code detection failed; no URL available."
            }
        else:
            url_result = {
                "url": "",
                "risk_level": "unavailable",
                "risk_score": 0,
                "risk_indicators": [],
                "summary": "QR code decode failed; no URL available."
            }

        # ------------------------------------------------------------
        # STEP 6: UNIFIED ORTHOGONAL RESULT
        # ------------------------------------------------------------
        overall_summary = (
            f"Physical Integrity: {tampering_summary} | "
            f"URL Security: {url_result['summary']}"
        )

        return {
            "tampering": {
                "status": tampering_status,
                "evidence": tampering_evidence,
                "ecc": {
                    "module_diff": ecc_module_diff,
                    "nonzero_syndromes": ecc_nonzero,
                    "residual": ecc_residual,
                    "status": ecc_status
                },
                "peak_anomaly": {
                    "score": round(peak_score, 6),
                    "threshold": self.peak_threshold,
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
            }
        }

    def _build_failure_result(self, error_code: str, error_msg: str) -> Dict[str, Any]:
        return {
            "tampering": {
                "status": "unavailable",
                "evidence": [],
                "ecc": {"module_diff": 0, "nonzero_syndromes": 0, "residual": 0, "status": "unavailable"},
                "peak_anomaly": {"score": 0.0, "threshold": self.peak_threshold, "triggered": False, "status": "unavailable"},
                "summary": error_msg
            },
            "url": {
                "url": "",
                "risk_level": "unavailable",
                "risk_score": 0,
                "risk_indicators": [],
                "summary": error_msg
            },
            "overall": {
                "tampering_status": "unavailable",
                "url_risk_level": "unavailable",
                "inspection_summary": f"Inspection aborted: {error_msg}"
            }
        }
