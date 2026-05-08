import io
import numpy as np
import requests
import cv2
from PIL import Image

# Saturation thresholds (HSV S channel, 0-255)
_HIGH_SAT_VALUE = 180
_HIGH_SAT_RATIO_THRESHOLD = 0.35

# Canny edge density proxy for text overlay
_EDGE_DENSITY_THRESHOLD = 0.08

_REQUEST_TIMEOUT = 10  # seconds


class ThumbnailAnalyzer:
    """Analyzes a YouTube thumbnail for visual clickbait signals."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def analyze_from_url(self, thumbnail_url: str) -> dict:
        """Download thumbnail then analyze it."""
        try:
            img_bgr = self._download_image(thumbnail_url)
        except Exception as exc:
            return self._empty_result(error=str(exc))
        return self.analyze_from_array(img_bgr)

    def analyze_from_array(self, img_bgr: np.ndarray) -> dict:
        """
        Returns sub_score (0–35) and detail signals.
        """
        if img_bgr is None or img_bgr.size == 0:
            return self._empty_result(error="Empty image")

        mean_sat, high_sat_ratio = self._check_saturation(img_bgr)
        has_text_overlay = self._detect_text_overlay(img_bgr)
        has_excited_face, face_count = self._detect_excited_face(img_bgr)

        points = 0
        flags = []

        if high_sat_ratio > _HIGH_SAT_RATIO_THRESHOLD:
            points += 12
            flags.append(f"Shock colors ({high_sat_ratio:.0%} highly saturated)")

        if has_text_overlay:
            points += 10
            flags.append("Bold text overlay detected")

        if has_excited_face:
            points += 8
            flags.append("Excited/open-mouth face detected")

        if face_count > 2:
            points += 5
            flags.append(f"{face_count} faces (multiple shocked faces)")

        sub_score = min(35, points)
        return {
            "sub_score": sub_score,
            "mean_saturation": float(mean_sat),
            "high_sat_pixel_ratio": float(high_sat_ratio),
            "has_text_overlay": has_text_overlay,
            "has_excited_face": has_excited_face,
            "face_count": face_count,
            "flags": flags,
        }

    def _check_saturation(self, img_bgr: np.ndarray) -> tuple:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        mean_sat = float(np.mean(s_channel))
        high_sat_pixels = np.sum(s_channel > _HIGH_SAT_VALUE)
        total_pixels = s_channel.size
        high_sat_ratio = high_sat_pixels / total_pixels if total_pixels > 0 else 0.0
        return mean_sat, high_sat_ratio

    def _detect_text_overlay(self, img_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density > _EDGE_DENSITY_THRESHOLD

    def _detect_excited_face(self, img_bgr: np.ndarray) -> tuple:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        face_count = len(faces) if faces is not None and len(faces) > 0 else 0
        has_excited = False
        if face_count > 0:
            for (x, y, w, h) in faces:
                # Tall bounding box aspect ratio suggests open mouth (excited expression)
                aspect = h / w if w > 0 else 1.0
                if aspect > 1.25:
                    has_excited = True
                    break
        return has_excited, face_count

    def _download_image(self, url: str) -> np.ndarray:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _empty_result(self, error: str = "") -> dict:
        return {
            "sub_score": 0,
            "mean_saturation": 0.0,
            "high_sat_pixel_ratio": 0.0,
            "has_text_overlay": False,
            "has_excited_face": False,
            "face_count": 0,
            "flags": [],
            "error": error,
        }
