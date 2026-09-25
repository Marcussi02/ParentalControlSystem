import numpy as np
import pytest

from conftest import FRAME, eye_points, face_with_ear
from modules.drowsiness.drowsiness_detector import DrowsinessDetector
from modules.drowsiness.ear_calculator import EARCalculator

CONFIG = {
    "ear_threshold": 0.21,
    "sleepy_ear_threshold": 0.26,
    "sleepy_seconds": 15,
    "continuous_close_seconds": 3,
    "perclos_window_seconds": 60,
    "perclos_alert_threshold": 0.80,
}


# ---------- EAR maths ----------

@pytest.mark.parametrize("ear", [0.1, 0.21, 0.3])
def test_ear_formula_matches_constructed_eye(ear):
    pts = np.array(eye_points(ear)) * 100  # pixel space
    assert EARCalculator.ear(pts) == pytest.approx(ear)


def test_ear_degenerate_eye_returns_zero():
    pts = np.zeros((6, 2))
    assert EARCalculator.ear(pts) == 0.0


def test_average_ear():
    assert EARCalculator.average_ear(0.2, 0.3) == pytest.approx(0.25)


# ---------- detector behaviour ----------

def run(detector, clock, ear, seconds, step=0.5):
    """Feed frames with a constant EAR for `seconds`; return the last status."""
    status = None
    for _ in range(int(seconds / step) + 1):
        status = detector.on_frame(FRAME, face_with_ear(ear))
        clock.advance(step)
    return status


def test_open_eyes_never_alert(clock):
    alerts = []
    d = DrowsinessDetector(lambda *a: alerts.append(a), CONFIG)
    status = run(d, clock, ear=0.30, seconds=30)
    assert alerts == []
    assert status["eyes_closed"] is False and status["drowsy"] is False


def test_eyes_closed_for_three_seconds_raises_one_alert(clock):
    alerts = []
    d = DrowsinessDetector(lambda *a: alerts.append(a), CONFIG)
    status = run(d, clock, ear=0.10, seconds=5)
    assert status["drowsy"] is True
    assert status["continuous_close_duration"] >= 3
    # Cooldown (10 s) means one alert, not one per frame
    assert [a[0] for a in alerts] == ["DROWSINESS"]


def test_brief_blink_does_not_alert(clock):
    alerts = []
    d = DrowsinessDetector(lambda *a: alerts.append(a), CONFIG)
    run(d, clock, ear=0.10, seconds=1)   # blink / short close
    run(d, clock, ear=0.30, seconds=5)   # eyes open again
    assert alerts == []


def test_sustained_drooping_eyes_raise_sleepy_alert(clock):
    alerts = []
    d = DrowsinessDetector(lambda *a: alerts.append(a), CONFIG)
    status = run(d, clock, ear=0.23, seconds=16)  # between 0.21 and 0.26
    assert status["eyes_sleepy"] is True and status["eyes_closed"] is False
    assert "SLEEPY" in [a[0] for a in alerts]
    assert "DROWSINESS" not in [a[0] for a in alerts]


def test_perclos_is_share_of_closed_frames_in_window(clock):
    d = DrowsinessDetector(lambda *a: None, {**CONFIG, "continuous_close_seconds": 999})
    for i in range(10):  # alternate open/closed: 50% closed
        d.on_frame(FRAME, face_with_ear(0.10 if i % 2 else 0.30))
        clock.advance(1)
    assert d.status["perclos"] == pytest.approx(0.5)


def test_perclos_forgets_frames_outside_window(clock):
    d = DrowsinessDetector(lambda *a: None, {**CONFIG, "continuous_close_seconds": 999,
                                             "perclos_window_seconds": 5})
    run(d, clock, ear=0.10, seconds=4, step=1)   # closed
    run(d, clock, ear=0.30, seconds=10, step=1)  # open long enough to flush window
    assert d.status["perclos"] == 0.0


def test_perclos_alerts_only_after_a_full_window(clock):
    alerts = []
    cfg = {**CONFIG, "continuous_close_seconds": 999, "perclos_window_seconds": 10}
    d = DrowsinessDetector(lambda *a: alerts.append(a), cfg)
    # 90% closed frames (1 open in 10) from the very first frame
    for i in range(8):
        d.on_frame(FRAME, face_with_ear(0.30 if i % 10 == 0 else 0.10))
        clock.advance(1)
    assert alerts == []  # window (10 s) not yet full
    for i in range(8, 20):
        d.on_frame(FRAME, face_with_ear(0.30 if i % 10 == 0 else 0.10))
        clock.advance(1)
    assert [a[0] for a in alerts] == ["DROWSINESS"]
    assert "PERCLOS" in alerts[0][1]


def test_no_face_resets_to_not_drowsy(clock):
    d = DrowsinessDetector(lambda *a: None, CONFIG)
    run(d, clock, ear=0.10, seconds=5)
    status = d.on_frame(FRAME, None)
    assert status["drowsy"] is False and status["ear"] is None


def test_glasses_offset_raises_closed_threshold(clock):
    d = DrowsinessDetector(lambda *a: None, {**CONFIG, "ear_threshold_offset": 0.02})
    status = d.on_frame(FRAME, face_with_ear(0.22))  # closed only with the +0.02 offset
    assert status["eyes_closed"] is True
