import numpy as np

# Iris center landmarks (requires refine_landmarks=True)
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

# Eye corner landmarks
_LEFT_EYE_LEFT_CORNER = 33
_LEFT_EYE_RIGHT_CORNER = 133
_RIGHT_EYE_LEFT_CORNER = 362
_RIGHT_EYE_RIGHT_CORNER = 263

# Vertical reference landmarks (upper/lower eyelid midpoints)
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374


class GazeDetector:
    """
    Estimates gaze direction from MediaPipe iris + eye-corner landmarks.
    Returns normalized horizontal and vertical ratios:
      horizontal ~0 = far left, ~0.5 = center, ~1 = far right
      vertical   ~0 = far up,   ~0.5 = center, ~1 = far down
    """

    def __init__(self, center_tolerance: float = 0.15):
        self._tol = center_tolerance

    def detect(self, face_landmarks, frame_w: int, frame_h: int) -> dict:
        if face_landmarks is None:
            return {
                "direction": "unknown",
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "looking_at_screen": False,
            }

        lm = face_landmarks  # Tasks API: plain list of NormalizedLandmark
        try:
            left_h, left_v = self._iris_ratios(
                lm, _LEFT_IRIS_CENTER,
                _LEFT_EYE_LEFT_CORNER, _LEFT_EYE_RIGHT_CORNER,
                _LEFT_EYE_TOP, _LEFT_EYE_BOTTOM,
                frame_w, frame_h,
            )
            right_h, right_v = self._iris_ratios(
                lm, _RIGHT_IRIS_CENTER,
                _RIGHT_EYE_LEFT_CORNER, _RIGHT_EYE_RIGHT_CORNER,
                _RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM,
                frame_w, frame_h,
            )
        except (IndexError, ZeroDivisionError):
            return {
                "direction": "unknown",
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "looking_at_screen": False,
            }

        h_ratio = (left_h + right_h) / 2.0
        v_ratio = (left_v + right_v) / 2.0

        direction = self._classify(h_ratio, v_ratio)
        looking_at_screen = direction == "center"

        return {
            "direction": direction,
            "horizontal_ratio": h_ratio,
            "vertical_ratio": v_ratio,
            "looking_at_screen": looking_at_screen,
        }

    def _iris_ratios(self, lm, iris_idx: int,
                     left_corner_idx: int, right_corner_idx: int,
                     top_idx: int, bottom_idx: int,
                     frame_w: int, frame_h: int) -> tuple:
        def px(idx):
            p = lm[idx]
            return np.array([p.x * frame_w, p.y * frame_h])

        iris = px(iris_idx)
        left_c = px(left_corner_idx)
        right_c = px(right_corner_idx)
        top = px(top_idx)
        bottom = px(bottom_idx)

        eye_width = np.linalg.norm(right_c - left_c)
        eye_height = np.linalg.norm(bottom - top)

        h_ratio = (iris[0] - left_c[0]) / eye_width if eye_width > 0 else 0.5
        v_ratio = (iris[1] - top[1]) / eye_height if eye_height > 0 else 0.5

        return h_ratio, v_ratio

    def _classify(self, h_ratio: float, v_ratio: float) -> str:
        tol = self._tol
        center_h = abs(h_ratio - 0.5) <= tol
        center_v = abs(v_ratio - 0.5) <= tol

        if center_h and center_v:
            return "center"
        if not center_h:
            return "left" if h_ratio < 0.5 else "right"
        return "up" if v_ratio < 0.5 else "down"
