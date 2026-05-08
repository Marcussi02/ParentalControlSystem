from .youtube_scraper import YouTubeScraper, ScraperError
from .title_analyzer import TitleAnalyzer
from .thumbnail_analyzer import ThumbnailAnalyzer
from .content_filter import ContentFilter


class ClickbaitScorer:
    """
    Full pipeline: scrape YouTube → analyze title → analyze thumbnail →
    run content safety filter → combined scores.

    Returns both a clickbait score (0–100) and a content safety score (0–100).
    """

    def __init__(self, threshold: int = 60, kids_mode: bool = True,
                 safety_threshold: int = 30):
        self._scraper = YouTubeScraper()
        self._title_analyzer = TitleAnalyzer()
        self._thumbnail_analyzer = ThumbnailAnalyzer()
        self._content_filter = ContentFilter()
        self._threshold = threshold
        self._safety_threshold = safety_threshold
        self._kids_multiplier = 1.2 if kids_mode else 1.0

    def score_url(self, url_or_id: str) -> dict:
        """
        Raises ScraperError if YouTube metadata cannot be retrieved.

        Returns dict with keys:
            video_id, title, channel, thumbnail_url
            final_score         int 0–100  (clickbait)
            is_clickbait        bool
            safety_score        int 0–100  (content safety)
            is_inappropriate    bool
            categories_flagged  list[str]
            flags               list[str]  (clickbait flags)
            safety_flags        list[str]  (content flags)
            title_analysis, thumbnail_analysis, content_analysis  (raw detail)
        """
        metadata = self._scraper.extract_metadata(url_or_id)

        # Download thumbnail once; share the BGR array with both analyzers
        thumb_bgr = self._thumbnail_analyzer._download_image(
            metadata["thumbnail_url"]
        ) if metadata["thumbnail_url"] else None

        # --- Clickbait analysis ---
        title_result = self._title_analyzer.analyze(
            metadata["title"], kids_mode_multiplier=self._kids_multiplier
        )
        thumbnail_result = (
            self._thumbnail_analyzer.analyze_from_array(thumb_bgr)
            if thumb_bgr is not None
            else self._thumbnail_analyzer._empty_result()
        )

        bonus = self._compute_bonus(title_result, thumbnail_result)
        final_score = min(
            100,
            title_result["sub_score"] + thumbnail_result["sub_score"] + bonus,
        )
        clickbait_flags = title_result["flags"] + thumbnail_result["flags"]
        if bonus > 0:
            clickbait_flags.append(f"Coordinated clickbait bonus (+{bonus})")

        # --- Content safety analysis ---
        content_result = self._content_filter.analyze(
            title=metadata["title"],
            description=metadata.get("description", ""),
            thumbnail_img_bgr=thumb_bgr,
            age_limit=metadata.get("age_limit", 0),
        )

        return {
            "video_id": metadata["video_id"],
            "title": metadata["title"],
            "channel": metadata["channel"],
            "thumbnail_url": metadata["thumbnail_url"],
            # Clickbait
            "final_score": final_score,
            "is_clickbait": final_score >= self._threshold,
            "title_sub_score": title_result["sub_score"],
            "thumbnail_sub_score": thumbnail_result["sub_score"],
            "bonus_points": bonus,
            "flags": clickbait_flags,
            # Content safety
            "safety_score": content_result["safety_score"],
            "is_inappropriate": content_result["is_inappropriate"],
            "categories_flagged": content_result["categories_flagged"],
            "safety_flags": content_result["flags"],
            # Raw detail
            "title_analysis": title_result,
            "thumbnail_analysis": thumbnail_result,
            "content_analysis": content_result,
        }

    def _compute_bonus(self, title_result: dict, thumb_result: dict) -> int:
        bonus = 0
        kw_count = len(title_result.get("matched_keywords", []))
        has_shock_colors = thumb_result.get("high_sat_pixel_ratio", 0) > 0.35
        has_text_overlay = thumb_result.get("has_text_overlay", False)
        has_excited_face = thumb_result.get("has_excited_face", False)
        is_all_caps = title_result.get("caps_ratio", 0) > 0.5

        if kw_count >= 2 and has_shock_colors:
            bonus += 10
        if is_all_caps and has_text_overlay:
            bonus += 8
        if has_excited_face and kw_count >= 1:
            bonus += 7

        return min(25, bonus)
