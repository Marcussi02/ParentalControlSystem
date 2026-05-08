import numpy as np
from scipy.spatial import distance as dist

# Canonical 6-point EAR indices mapped to MediaPipe FaceMesh landmarks.
# Order: P1(left corner), P2(upper outer), P3(upper inner),
#        P4(right corner), P5(lower inner), P6(lower outer)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 380, 373]


class EARCalculator:
    """Computes Eye Aspect Ratio from MediaPipe face landmarks."""

    @staticmethod
    def landmarks_to_points(face_landmarks, indices: list,
                            frame_w: int, frame_h: int) -> np.ndarray:
        """Extract pixel (x, y) coordinates for the given landmark indices."""
        pts = []
        for idx in indices:
            lm = face_landmarks[idx]   # Tasks API: plain list, not proto
            pts.append([lm.x * frame_w, lm.y * frame_h])
        return np.array(pts, dtype=np.float64)

    @staticmethod
    def ear(eye_points: np.ndarray) -> float:
        """
        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
        eye_points: shape (6, 2), ordered [P1, P2, P3, P4, P5, P6]
        Returns ~0.25-0.35 for open eyes; < 0.21 for closed eyes.
        """
        A = dist.euclidean(eye_points[1], eye_points[5])  # P2-P6
        B = dist.euclidean(eye_points[2], eye_points[4])  # P3-P5
        C = dist.euclidean(eye_points[0], eye_points[3])  # P1-P4
        if C < 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    @staticmethod
    def average_ear(left_ear: float, right_ear: float) -> float:
        return (left_ear + right_ear) / 2.0
