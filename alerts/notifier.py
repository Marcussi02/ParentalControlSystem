import threading
import json
from datetime import datetime
from pathlib import Path

try:
    from plyer import notification as plyer_notification
    _PLYER_AVAILABLE = True
except Exception:
    _PLYER_AVAILABLE = False


class AlertNotifier:
    """Thread-safe desktop notification dispatcher and JSON event logger."""

    def __init__(self, log_path: str, notification_title: str, timeout: int = 5):
        self._lock = threading.Lock()
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._title = notification_title
        self._timeout = timeout

    def send(self, alert_type: str, message: str, metadata: dict = None) -> None:
        """Send desktop notification and append event to log file."""
        if metadata is None:
            metadata = {}
        self._log_event(alert_type, message, metadata)
        self._toast(alert_type, message)

    def _toast(self, alert_type: str, message: str) -> None:
        if not _PLYER_AVAILABLE:
            return
        try:
            plyer_notification.notify(
                title=f"{self._title} — {alert_type}",
                message=message,
                timeout=self._timeout,
            )
        except Exception:
            pass

    def _log_event(self, alert_type: str, message: str, metadata: dict) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "message": message,
            "metadata": metadata,
        }
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def get_recent_events(self, n: int = 50) -> list:
        """Return the last n events from the log file."""
        if not self._log_path.exists():
            return []
        with self._lock:
            try:
                lines = self._log_path.read_text(encoding="utf-8").splitlines()
                records = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return records[-n:]
            except Exception:
                return []
