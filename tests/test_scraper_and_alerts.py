import json

from alerts import notifier as notifier_module
from alerts.notifier import AlertNotifier
from modules.clickbait.youtube_scraper import YouTubeScraper


def test_bare_video_id_becomes_watch_url():
    s = YouTubeScraper()
    assert s._normalize_input(" dQw4w9WgXcQ ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_full_url_is_left_alone():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10"
    assert YouTubeScraper()._normalize_input(url) == url


def test_best_thumbnail_prefers_highest_resolution():
    info = {"thumbnails": [{"url": "small.jpg"}, {"url": ""}, {"url": "large.jpg"}]}
    assert YouTubeScraper()._best_thumbnail(info) == "large.jpg"
    assert YouTubeScraper()._best_thumbnail({"thumbnail": "only.jpg"}) == "only.jpg"


def test_alerts_are_logged_as_json_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(notifier_module, "_PLYER_AVAILABLE", False)  # no desktop popups
    log = tmp_path / "logs" / "events.json"
    n = AlertNotifier(str(log), "Parental Control Alert")

    n.send("DROWSINESS", "Drowsiness detected", {"ear": 0.12})
    n.send("LOOK_AWAY", "Looked away for 20s")

    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["alert_type"] for line in lines] == ["DROWSINESS", "LOOK_AWAY"]
    recent = n.get_recent_events(1)
    assert recent[0]["message"] == "Looked away for 20s"
    assert recent[0]["metadata"] == {}
