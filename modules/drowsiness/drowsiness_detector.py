import time
import threading
import collections
from .ear_calculator import EARCalculator, LEFT_EYE_INDICES, RIGHT_EYE_INDICES


class DrowsinessDetector:
    """
    Registered as a FacePipeline callback.
    Tracks three drowsiness signals:
      1. Sleepy zone: EAR persistently between ear_threshold and sleepy_threshold
         (heavy/drooping eyes) for > sleepy_seconds → SLEEPY alert
      2. Continuous close: eyes fully shut for > continuous_close_seconds → DROWSINESS alert
      3. PERCLOS: % of closed frames over a rolling window > threshold → DROWSINESS alert
    """

    _DROWSY_COOLDOWN = 10.0
    _SLEEPY_COOLDOWN = 20.0

    def __init__(self, on_alert, config: dict = None):
        cfg = config or {}
        self._ear_threshold = cfg.get("ear_threshold", 0.21)
        self._ear_threshold += cfg.get("ear_threshold_offset", 0.0)
        self._sleepy_threshold = cfg.get("sleepy_ear_threshold", 0.26)
        self._sleepy_limit = cfg.get("sleepy_seconds", 5.0)
        self._continuous_limit = cfg.get("continuous_close_seconds", 3.0)
        self._perclos_window = cfg.get("perclos_window_seconds", 60.0)
        self._perclos_alert = cfg.get("perclos_alert_threshold", 0.80)
        self._on_alert = on_alert

        self._eyes_closed_since: float | None = None
        self._sleepy_since: float | None = None
        self._perclos_buffer: collections.deque = collections.deque()
        self._perclos_started: float | None = None  # first frame seen
        self._last_drowsy_alert: float = 0.0
        self._last_sleepy_alert: float = 0.0
        self._lock = threading.Lock()
        self._status = {
            "ear": None,
            "eyes_closed": False,
            "eyes_sleepy": False,
            "continuous_close_duration": 0.0,
            "sleepy_duration": 0.0,
            "perclos": 0.0,
            "drowsy": False,
            "alert_reason": None,
        }

    @property
    def status(self) -> dict:
        return self._status

    def on_frame(self, frame, landmarks) -> dict:
        now = time.monotonic()
        frame_h, frame_w = frame.shape[:2]

        if landmarks is None:
            with self._lock:
                self._status = {**self._status, "ear": None, "drowsy": False,
                                "eyes_sleepy": False, "alert_reason": None}
            return self._status

        try:
            left_pts = EARCalculator.landmarks_to_points(
                landmarks, LEFT_EYE_INDICES, frame_w, frame_h
            )
            right_pts = EARCalculator.landmarks_to_points(
                landmarks, RIGHT_EYE_INDICES, frame_w, frame_h
            )
            left_ear = EARCalculator.ear(left_pts)
            right_ear = EARCalculator.ear(right_pts)
            ear = EARCalculator.average_ear(left_ear, right_ear)
        except Exception:
            return self._status

        is_closed = ear < self._ear_threshold
        # Sleepy zone: above closed threshold but below normal open threshold
        is_sleepy = self._ear_threshold <= ear < self._sleepy_threshold

        with self._lock:
            # --- Continuous close timer ---
            if is_closed:
                if self._eyes_closed_since is None:
                    self._eyes_closed_since = now
                continuous_duration = now - self._eyes_closed_since
            else:
                self._eyes_closed_since = None
                continuous_duration = 0.0

            # --- Sleepy zone timer (only counts when NOT fully closed) ---
            if is_sleepy and not is_closed:
                if self._sleepy_since is None:
                    self._sleepy_since = now
                sleepy_duration = now - self._sleepy_since
            else:
                self._sleepy_since = None
                sleepy_duration = 0.0

            # --- PERCLOS rolling window ---
            perclos = self._update_perclos(now, is_closed)

            # --- Determine alerts ---
            alert_reason = None

            # Full drowsiness (closed or PERCLOS)
            if continuous_duration >= self._continuous_limit:
                alert_reason = f"eyes closed {continuous_duration:.1f}s"
            elif perclos >= self._perclos_alert and self._perclos_window_full(now):
                # Only trust PERCLOS once a full window has been observed;
                # otherwise a single blink at start-up reads as 100% closed.
                alert_reason = f"PERCLOS {perclos:.0%} over {self._perclos_window:.0f}s"

            drowsy = alert_reason is not None
            if drowsy and (now - self._last_drowsy_alert) > self._DROWSY_COOLDOWN:
                self._last_drowsy_alert = now
                msg = f"Drowsiness detected: {alert_reason}"
                meta = {"ear": ear, "perclos": perclos,
                        "continuous_seconds": continuous_duration}
                try:
                    self._on_alert("DROWSINESS", msg, meta)
                except Exception:
                    pass

            # Sleepy eyes (sustained heavy/drooping without full close)
            if (sleepy_duration >= self._sleepy_limit
                    and (now - self._last_sleepy_alert) > self._SLEEPY_COOLDOWN):
                self._last_sleepy_alert = now
                msg = (f"Sleepy eyes detected: EAR {ear:.3f} "
                       f"(drooping for {sleepy_duration:.0f}s)")
                meta = {"ear": ear, "sleepy_seconds": sleepy_duration}
                try:
                    self._on_alert("SLEEPY", msg, meta)
                except Exception:
                    pass

            self._status = {
                "ear": ear,
                "eyes_closed": is_closed,
                "eyes_sleepy": is_sleepy,
                "continuous_close_duration": continuous_duration,
                "sleepy_duration": sleepy_duration,
                "perclos": perclos,
                "drowsy": drowsy,
                "alert_reason": alert_reason,
            }

        return self._status

    def _perclos_window_full(self, now: float) -> bool:
        return (self._perclos_started is not None
                and now - self._perclos_started >= self._perclos_window)

    def _update_perclos(self, now: float, is_closed: bool) -> float:
        if self._perclos_started is None:
            self._perclos_started = now
        cutoff = now - self._perclos_window
        while self._perclos_buffer and self._perclos_buffer[0][0] < cutoff:
            self._perclos_buffer.popleft()
        self._perclos_buffer.append((now, is_closed))
        if not self._perclos_buffer:
            return 0.0
        closed_count = sum(1 for _, c in self._perclos_buffer if c)
        return closed_count / len(self._perclos_buffer)
