import time
import threading
from .gaze_detector import GazeDetector


class AttentionMonitor:
    """
    Registered as a FacePipeline callback.
    Fires alerts when the child looks away or no face is detected for
    longer than look_away_threshold seconds.
    """

    _ALERT_COOLDOWN = 15.0  # minimum seconds between repeated alerts

    def __init__(self, on_alert, look_away_threshold: float = 10.0,
                 config: dict = None):
        cfg = config or {}
        self._gaze = GazeDetector(
            center_tolerance=cfg.get("gaze_center_tolerance", 0.15)
        )
        self._threshold = look_away_threshold
        self._on_alert = on_alert
        self._looking_away_since: float | None = None
        self._no_face_since: float | None = None
        self._last_alert_time: float = 0.0
        self._lock = threading.Lock()
        self._status = {
            "gaze": {"direction": "unknown", "looking_at_screen": False},
            "looking_away_duration": 0.0,
            "no_face_duration": 0.0,
            "attention_ok": True,
        }

    @property
    def status(self) -> dict:
        return self._status

    def on_frame(self, frame, landmarks) -> dict:
        now = time.monotonic()
        frame_h, frame_w = frame.shape[:2]

        gaze = self._gaze.detect(landmarks, frame_w, frame_h)

        with self._lock:
            if landmarks is None:
                # No face detected
                if self._no_face_since is None:
                    self._no_face_since = now
                self._looking_away_since = None
                no_face_duration = now - self._no_face_since
                looking_away_duration = 0.0
            else:
                self._no_face_since = None
                no_face_duration = 0.0
                if not gaze["looking_at_screen"]:
                    if self._looking_away_since is None:
                        self._looking_away_since = now
                    looking_away_duration = now - self._looking_away_since
                else:
                    self._looking_away_since = None
                    looking_away_duration = 0.0

            attention_ok = (
                looking_away_duration < self._threshold
                and no_face_duration < self._threshold
            )

            # Fire alert once per cooldown period
            if not attention_ok and (now - self._last_alert_time) > self._ALERT_COOLDOWN:
                self._last_alert_time = now
                if no_face_duration >= self._threshold:
                    msg = f"No face detected for {no_face_duration:.0f}s — child may have left"
                    meta = {"no_face_seconds": no_face_duration}
                else:
                    msg = f"Child looking away for {looking_away_duration:.0f}s"
                    meta = {
                        "gaze_direction": gaze["direction"],
                        "away_seconds": looking_away_duration,
                    }
                try:
                    self._on_alert("ATTENTION", msg, meta)
                except Exception:
                    pass

            self._status = {
                "gaze": gaze,
                "looking_away_duration": looking_away_duration,
                "no_face_duration": no_face_duration,
                "attention_ok": attention_ok,
            }

        return self._status
