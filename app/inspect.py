"""
Command-Line Interface (CLI) for QR Code Inspection.

Usage:
    python app/inspect.py <image_path>
"""

import sys
import os

# Prevent shadowing Python standard library 'inspect' module when running `python app/inspect.py`
current_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == current_dir:
    sys.path.pop(0)

project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now stdlib and project imports proceed without circular shadow
import json
import argparse
from app.inspector import inspect_image


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a QR code image for physical/structural tampering and destination URL risk."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the QR code image file to inspect."
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for pretty printing (default: 2)."
    )

    args = parser.parse_args()

    result = inspect_image(image_path=args.image_path)
    print(json.dumps(result, indent=args.indent))


if __name__ == "__main__":
    main()
