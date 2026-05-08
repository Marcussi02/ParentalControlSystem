import re


class ScraperError(Exception):
    pass


class YouTubeScraper:
    """Scrapes YouTube video metadata via yt-dlp — no API key required."""

    _YDL_OPTIONS = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }

    _BARE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{11}$")

    def extract_metadata(self, url_or_id: str) -> dict:
        """
        Returns dict with keys: video_id, title, thumbnail_url, duration,
        view_count, channel, description.
        Raises ScraperError on failure.
        """
        import yt_dlp  # lazy — only loaded when first URL is checked
        url = self._normalize_input(url_or_id)
        try:
            with yt_dlp.YoutubeDL(self._YDL_OPTIONS) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise ScraperError(f"yt-dlp failed: {exc}") from exc
        except Exception as exc:
            raise ScraperError(f"Unexpected error: {exc}") from exc

        thumbnail_url = self._best_thumbnail(info)
        return {
            "video_id": info.get("id", ""),
            "title": info.get("title", ""),
            "thumbnail_url": thumbnail_url,
            "duration": info.get("duration", 0),
            "view_count": info.get("view_count"),
            "channel": info.get("uploader") or info.get("channel", ""),
            "description": (info.get("description") or "")[:500],
            "age_limit": info.get("age_limit") or 0,
        }

    def _normalize_input(self, url_or_id: str) -> str:
        if self._BARE_ID_RE.match(url_or_id.strip()):
            return f"https://www.youtube.com/watch?v={url_or_id.strip()}"
        return url_or_id.strip()

    def _best_thumbnail(self, info: dict) -> str:
        thumbnails = info.get("thumbnails") or []
        if thumbnails:
            # yt-dlp returns thumbnails sorted ascending by resolution; take last
            for t in reversed(thumbnails):
                url = t.get("url", "")
                if url:
                    return url
        return info.get("thumbnail", "")
