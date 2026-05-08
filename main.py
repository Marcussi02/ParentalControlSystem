import sys
import signal
from pathlib import Path

import yaml
import tkinter as tk

from alerts.notifier import AlertNotifier
from modules.eye_tracking.face_pipeline import FacePipeline
from modules.eye_tracking.attention_monitor import AttentionMonitor
from modules.drowsiness.drowsiness_detector import DrowsinessDetector
from modules.clickbait.scorer import ClickbaitScorer
from ui.dashboard import Dashboard
from ui.browser import BrowserWindow


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))

    notifier = AlertNotifier(
        log_path=config["alerts"]["log_path"],
        notification_title=config["alerts"]["notification_title"],
        timeout=config["alerts"].get("notification_timeout", 5),
    )

    pipeline = FacePipeline(
        device_index=config["camera"]["device_index"],
        width=config["camera"]["frame_width"],
        height=config["camera"]["frame_height"],
        model_path=config.get("face_landmarker_model", "face_landmarker.task"),
    )

    def on_alert(alert_type: str, message: str, metadata: dict = None) -> None:
        notifier.send(alert_type, message, metadata or {})

    attention = AttentionMonitor(
        on_alert=on_alert,
        look_away_threshold=config["eye_tracking"]["look_away_threshold_seconds"],
        config=config["eye_tracking"],
    )
    drowsiness = DrowsinessDetector(
        on_alert=on_alert,
        config=config["drowsiness"],
    )

    # Register callbacks before starting the pipeline
    pipeline.register_callback(attention.on_frame)
    pipeline.register_callback(drowsiness.on_frame)

    scorer = ClickbaitScorer(
        threshold=config["clickbait"]["score_threshold"],
        kids_mode=config["clickbait"].get("kids_mode", True),
    )

    pipeline.start()

    root = tk.Tk()
    root.title("Parental Control System")
    root.geometry(
        f"{config['gui']['dashboard_width']}x{config['gui']['dashboard_height']}"
    )

    app = Dashboard(root, pipeline, attention, drowsiness, scorer, notifier, config)

    # Browser window — started on demand when user clicks "Open Browser"
    def on_browser_url(url: str) -> None:
        root.after(0, lambda: app.on_browser_url(url))

    browser = BrowserWindow(on_url_changed=on_browser_url)
    app.set_browser(browser)

    def on_close() -> None:
        app.stop()
        pipeline.stop()
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    signal.signal(signal.SIGINT, lambda *_: on_close())

    app.run()


if __name__ == "__main__":
    main()
