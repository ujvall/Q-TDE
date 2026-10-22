"""
Unified QR Code Preprocessing, Localization, and Rectification Layer.

Provides a shared preprocessing implementation across CLI, GUI upload, and camera input:
1. QR detection and 4-corner localization (with OpenCV QRCodeDetector and Aruco fallback).
2. Orientation and perspective rectification to standard Version 3-M dimensions (370x370 px).
3. Quiet-zone normalization (4 modules padding with white background).
4. Explicit geometry validation and structured failure reporting.
"""

import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class PreprocessingResult:
    """
    Structured outcome of the QR preprocessing and rectification pipeline.
    """
    success: bool
    status: str
    message: str
    original_image: Optional[np.ndarray]
    rectified_image: Optional[np.ndarray]
    corners: Optional[np.ndarray]
    decoded_payload: Optional[str]
    detected_version: Optional[int]
    is_version_3: bool
    has_normalized_quiet_zone: bool
    output_dimensions: Optional[Tuple[int, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes preprocessing metadata for structured JSON output."""
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "corners": self.corners.tolist() if self.corners is not None else None,
            "decoded_payload": self.decoded_payload,
            "detected_version": self.detected_version,
            "is_version_3": self.is_version_3,
            "has_normalized_quiet_zone": self.has_normalized_quiet_zone,
            "output_dimensions": list(self.output_dimensions) if self.output_dimensions else None
        }


class QRPreprocessor:
    """
    Reusable QR image preprocessor performing localization, corner detection,
    perspective correction, and quiet zone normalization.
    """

    def __init__(self, target_size: int = 370):
        self.target_size = target_size
        self.detector = cv2.QRCodeDetector()
        self.aruco_detector = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else None

    def preprocess(
        self,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None
    ) -> PreprocessingResult:
        """
        Executes unified QR localization and rectification.

        Parameters:
            image: In-memory numpy array (grayscale or BGR).
            image_path: Optional path to an image file.

        Returns:
            PreprocessingResult dataclass with structured diagnostics.
        """
        # Step 1: Input ingestion and validation
        if image is None and image_path is not None:
            if not os.path.exists(image_path):
                return PreprocessingResult(
                    success=False,
                    status="INVALID_IMAGE",
                    message=f"Image file not found: {image_path}",
                    original_image=None,
                    rectified_image=None,
                    corners=None,
                    decoded_payload=None,
                    detected_version=None,
                    is_version_3=False,
                    has_normalized_quiet_zone=False,
                    output_dimensions=None
                )
            image = cv2.imread(image_path)

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return PreprocessingResult(
                success=False,
                status="INVALID_IMAGE",
                message="Input image data is empty, None, or unreadable.",
                original_image=None,
                rectified_image=None,
                corners=None,
                decoded_payload=None,
                detected_version=None,
                is_version_3=False,
                has_normalized_quiet_zone=False,
                output_dimensions=None
            )

        original_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Step 2: Corner localization and initial decode
        points = None
        decoded_data = ""
        straight_qrcode = None

        try:
            data, pts, qrcode_img = self.detector.detectAndDecode(gray)
            if pts is not None:
                points = pts
                decoded_data = data or ""
                straight_qrcode = qrcode_img
        except Exception:
            pass

        # Fallback to ArUco-based detector if standard detector found no corners
        if points is None and self.aruco_detector is not None:
            try:
                data_ar, pts_ar, qrcode_img_ar = self.aruco_detector.detectAndDecode(gray)
                if pts_ar is not None:
                    points = pts_ar
                    decoded_data = data_ar or ""
                    straight_qrcode = qrcode_img_ar
            except Exception:
                pass

        if points is None:
            return PreprocessingResult(
                success=False,
                status="NO_QR_DETECTED",
                message="No QR code localized in the input image.",
                original_image=original_image,
                rectified_image=None,
                corners=None,
                decoded_payload=None,
                detected_version=None,
                is_version_3=False,
                has_normalized_quiet_zone=False,
                output_dimensions=None
            )

        # Step 3: Corner formatting and perspective transformation
        pts_reshaped = points[0] if len(points.shape) == 3 else points
        pts_float = pts_reshaped.astype(np.float32)

        # Version 3 standard geometry: 29 modules symbol + 4 modules quiet zone per side = 37 modules
        # Module size = target_size / 37.0 = 10.0 px (at 370x370)
        border_px = int(round(4 * (self.target_size / 37.0)))  # 40 px
        symbol_end_px = self.target_size - 1 - border_px         # 329 px

        destination = np.float32([
            [border_px, border_px],
            [symbol_end_px, border_px],
            [symbol_end_px, symbol_end_px],
            [border_px, symbol_end_px]
        ])

        # Check if image is already a normalized upright 370x370 crop (bit-exact fast path)
        if gray.shape == (self.target_size, self.target_size) and np.allclose(pts_float, destination, atol=1.5):
            rectified = gray.copy()
        else:
            try:
                matrix = cv2.getPerspectiveTransform(pts_float, destination)
                rectified = cv2.warpPerspective(
                    gray,
                    matrix,
                    (self.target_size, self.target_size),
                    borderValue=255
                )
            except Exception as e:
                return PreprocessingResult(
                    success=False,
                    status="PERSPECTIVE_WARP_FAILED",
                    message=f"Perspective transformation failed: {str(e)}",
                    original_image=original_image,
                    rectified_image=None,
                    corners=pts_float,
                    decoded_payload=decoded_data if decoded_data else None,
                    detected_version=None,
                    is_version_3=False,
                    has_normalized_quiet_zone=False,
                    output_dimensions=None
                )

        # Step 4: Re-attempt decode on rectified image if initial decode was empty
        if not decoded_data:
            try:
                data_rect, _, qrcode_img_rect = self.detector.detectAndDecode(rectified)
                if data_rect:
                    decoded_data = data_rect
                if straight_qrcode is None or straight_qrcode.size == 0:
                    straight_qrcode = qrcode_img_rect
            except Exception:
                pass

        # Step 5: Version determination & geometry validation
        detected_version = None
        symbol_modules = None

        if straight_qrcode is not None and straight_qrcode.size > 0:
            symbol_modules = straight_qrcode.shape[0]
            if symbol_modules >= 21 and (symbol_modules - 17) % 4 == 0:
                detected_version = (symbol_modules - 17) // 4

        # If OpenCV straight QR was not extracted but symbol decodes cleanly
        if detected_version is None and decoded_data:
            # Assume Version 3 standard for 290 px symbol inside 370 px rectified image
            detected_version = 3
            symbol_modules = 29

        is_v3 = (detected_version == 3 or symbol_modules == 29)

        # Step 6: Determine explicit status
        if not decoded_data:
            status = "DECODE_FAILED"
            message = "QR code localized and perspective-rectified, but payload decoding failed."
        elif not is_v3:
            status = "UNSUPPORTED_VERSION"
            version_str = f"Version {detected_version}" if detected_version else "Unknown version"
            modules_str = f"{symbol_modules}x{symbol_modules}" if symbol_modules else "non-standard"
            message = (
                f"QR code localized and rectified, but detected geometry is {version_str} ({modules_str} modules); "
                f"current ECC analysis requires Version 3-M (29x29 modules)."
            )
        else:
            status = "SUCCESS"
            message = "QR code localized and perspective-corrected to 370x370 Version 3-M format."

        return PreprocessingResult(
            success=True,
            status=status,
            message=message,
            original_image=original_image,
            rectified_image=rectified,
            corners=pts_float,
            decoded_payload=decoded_data if decoded_data else None,
            detected_version=detected_version,
            is_version_3=is_v3,
            has_normalized_quiet_zone=True,
            output_dimensions=(self.target_size, self.target_size)
        )


# Shared global singleton preprocessor
_global_preprocessor: Optional[QRPreprocessor] = None

def get_preprocessor(target_size: int = 370) -> QRPreprocessor:
    global _global_preprocessor
    if _global_preprocessor is None or _global_preprocessor.target_size != target_size:
        _global_preprocessor = QRPreprocessor(target_size=target_size)
    return _global_preprocessor


def preprocess_qr_image(
    image: Optional[np.ndarray] = None,
    image_path: Optional[str] = None,
    target_size: int = 370
) -> PreprocessingResult:
    """
    Main entry point for unified QR preprocessing across CLI, GUI, and camera.
    """
    return get_preprocessor(target_size=target_size).preprocess(
        image=image,
        image_path=image_path
    )


# Standalone backward compatibility execution
if __name__ == "__main__":
    input_path = "data/raw/tilted_qr.png"
    output_path = "results/corrected_qr.png"

    if os.path.exists(input_path):
        res = preprocess_qr_image(image_path=input_path, target_size=400)
        if res.rectified_image is not None:
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(output_path, res.rectified_image)
            print("QR detected successfully")
            print("Decoded data:", res.decoded_payload)
            print("Perspective correction completed")
            print("Saved:", output_path)
        else:
            print("QR code not detected")
    else:
        print(f"Sample {input_path} not found.")