"""
Preprocessing package for QR code localization, perspective correction, and normalization.
"""

from preprocessing.qr_preprocessor import (
    QRPreprocessor,
    PreprocessingResult,
    preprocess_qr_image
)

__all__ = [
    "QRPreprocessor",
    "PreprocessingResult",
    "preprocess_qr_image"
]
