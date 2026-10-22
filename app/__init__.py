"""
Application Layer for QR Code Inspection.
Exposes inspector service and high-level inspect_image interface.
"""

from app.inspector import QRInspector, inspect_image, get_inspector

__all__ = ["QRInspector", "inspect_image", "get_inspector"]
