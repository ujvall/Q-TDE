"""
Camera and Live QR Input Processing Layer.

Provides frame capture, visual feedback overlays, and integration with the
shared preprocessing pipeline (preprocessing/qr_preprocessor.py) and
the application inspection service (app/inspector.py).
"""

import os
import sys
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.inspector import inspect_image
from preprocessing.qr_preprocessor import preprocess_qr_image, PreprocessingResult


class QRCameraProcessor:
    """
    Handles camera capture, delegating QR localization and perspective correction
    to the unified preprocessing layer, and visual feedback overlay rendering.
    """

    def __init__(self, target_size: int = 370):
        self.target_size = target_size

    def capture_frame(self, device_id: int = 0) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Captures a single frame from the specified local webcam device.
        Opens, reads, and immediately releases the camera device.
        """
        try:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                return False, None, f"Camera device {device_id} is unavailable or in use by another application."

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return False, None, "Camera opened, but failed to capture a valid frame."

            return True, frame, "Frame captured successfully."
        except Exception as e:
            return False, None, f"Camera access error: {str(e)}"

    def detect_and_correct(
        self,
        frame: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[str], Optional[np.ndarray], str]:
        """
        Delegates detection and perspective correction to shared preprocessor.

        Returns:
            (success, corrected_crop, decoded_data, corner_points, status_message)
        """
        prep = preprocess_qr_image(image=frame, target_size=self.target_size)
        return prep.success, prep.rectified_image, prep.decoded_payload, prep.corners, prep.message

    def draw_overlay(self, frame: np.ndarray, points: Optional[np.ndarray]) -> np.ndarray:
        """
        Draws green bounding box around detected QR corners on the RGB frame for visual feedback.
        """
        if frame is None:
            return frame

        annotated = frame.copy()
        if points is not None:
            try:
                pts = points.astype(np.int32)
                if len(pts.shape) == 3:
                    pts = pts[0]
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 230, 115), thickness=2)
                for pt in pts:
                    cv2.circle(annotated, tuple(pt), radius=4, color=(0, 230, 115), thickness=-1)
            except Exception:
                pass

        return annotated

    def inspect_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Optional[np.ndarray], str]:
        """
        End-to-end camera pipeline using shared preprocessing:
        Frame -> preprocess_qr_image() -> inspect_image(preprocessed=prep) -> Result.

        Returns:
            (structured_result, corrected_crop, status_message)
        """
        prep = preprocess_qr_image(image=frame, target_size=self.target_size)

        if not prep.success or prep.rectified_image is None:
            failure_result = inspect_image(preprocessed=prep)
            return failure_result, None, prep.message

        result = inspect_image(preprocessed=prep)
        return result, prep.rectified_image, prep.message
