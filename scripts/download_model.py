"""Download the MediaPipe Face Landmarker model used for eye tracking.

    python scripts/download_model.py
"""

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
TARGET = Path(__file__).resolve().parent.parent / "face_landmarker.task"


def main() -> int:
    if TARGET.exists():
        print(f"Model already present: {TARGET}")
        return 0
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, TARGET)
    print(f"Saved to {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
