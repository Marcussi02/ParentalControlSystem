import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Import modules exactly as main.py does (from the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.drowsiness.ear_calculator import LEFT_EYE_INDICES, RIGHT_EYE_INDICES  # noqa: E402

NUM_LANDMARKS = 478  # MediaPipe FaceMesh with iris refinement
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def eye_points(ear: float, x0: float = 0.2):
    """Six normalised eye points [P1..P6] whose eye aspect ratio equals `ear`.

    Eye width (P1-P4) is 0.2, so both vertical gaps are 0.2 * ear.
    """
    w, h = 0.2, 0.2 * ear
    return [
        (x0, 0.5),                      # P1 corner
        (x0 + w / 3, 0.5 - h / 2),      # P2 upper
        (x0 + 2 * w / 3, 0.5 - h / 2),  # P3 upper
        (x0 + w, 0.5),                  # P4 corner
        (x0 + 2 * w / 3, 0.5 + h / 2),  # P5 lower
        (x0 + w / 3, 0.5 + h / 2),      # P6 lower
    ]


def face_with_ear(ear: float):
    """Fake MediaPipe landmark list where both eyes have the given EAR."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(NUM_LANDMARKS)]
    for indices, x0 in ((LEFT_EYE_INDICES, 0.2), (RIGHT_EYE_INDICES, 0.6)):
        for idx, (x, y) in zip(indices, eye_points(ear, x0), strict=True):
            lm[idx] = SimpleNamespace(x=x, y=y)
    return lm


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    from modules.drowsiness import drowsiness_detector

    fake = FakeClock()
    monkeypatch.setattr(drowsiness_detector.time, "monotonic", fake)
    return fake
