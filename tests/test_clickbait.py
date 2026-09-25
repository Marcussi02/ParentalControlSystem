import pytest

from modules.clickbait.content_filter import ContentFilter
from modules.clickbait.title_analyzer import TitleAnalyzer
from modules.eye_tracking.gaze_detector import GazeDetector

# ---------- title analyser ----------

def test_plain_title_scores_zero():
    result = TitleAnalyzer().analyze("How volcanoes form - science for kids")
    assert result["sub_score"] == 0
    assert result["flags"] == []


def test_clickbait_title_is_flagged():
    title = "OMG!!! 24 HOURS IN A HAUNTED HOUSE CHALLENGE (GONE WRONG) 😱😱😱😱"
    result = TitleAnalyzer().analyze(title)
    assert result["sub_score"] >= 30
    assert {"omg", "24 hours", "haunted", "challenge", "gone wrong"} <= set(
        result["matched_keywords"]
    )
    assert result["exclamation_count"] == 3
    assert result["caps_ratio"] > 0.5
    assert any("emoji" in f for f in result["flags"])


def test_keyword_points_are_capped():
    many = "challenge amazing shocking secret banned free insane scary"
    result = TitleAnalyzer().analyze(many)
    assert result["sub_score"] == 20  # 8 keywords x 5 pts, capped at 20


def test_kids_mode_multiplier_never_exceeds_cap():
    title = "OMG!!! SECRET BANNED CHALLENGE?? YOU WON'T BELIEVE 😱😱😱😱"
    assert TitleAnalyzer().analyze(title, kids_mode_multiplier=1.2)["sub_score"] <= 40


# ---------- content filter ----------

def test_safe_video_is_not_flagged():
    result = ContentFilter().analyze("Counting to 10 with animals", "A gentle song")
    assert result["safety_score"] == 0
    assert result["is_inappropriate"] is False


def test_age_restricted_video_is_always_inappropriate():
    result = ContentFilter().analyze("Some video", age_limit=18)
    assert result["is_inappropriate"] is True
    assert "AGE_RESTRICTED" in result["categories_flagged"]


def test_title_match_scores_higher_than_description_match():
    cf = ContentFilter()
    in_title = cf.analyze("Real exorcism caught on tape", "")
    in_desc = cf.analyze("Caught on tape", "a real exorcism")
    assert in_title["safety_score"] > in_desc["safety_score"] > 0


# ---------- gaze classification ----------

@pytest.mark.parametrize(
    ("h", "v", "expected"),
    [
        (0.5, 0.5, "center"),
        (0.2, 0.5, "left"),
        (0.8, 0.5, "right"),
        (0.5, 0.2, "up"),
        (0.5, 0.8, "down"),
        (0.62, 0.40, "center"),  # within the 0.15 tolerance
    ],
)
def test_gaze_classification(h, v, expected):
    assert GazeDetector(center_tolerance=0.15)._classify(h, v) == expected
