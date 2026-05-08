import threading
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from typing import Callable

FrameCallback = Callable


class FacePipeline:
    """
    Owns the single webcam capture and MediaPipe FaceLandmarker (Tasks API).
    Runs in a background daemon thread and fans out to registered callbacks.

    Callbacks receive (frame_bgr: np.ndarray, landmarks: list | None) on each
    processed frame.  'landmarks' is a plain list of NormalizedLandmark objects
    (478 points: 468 face mesh + 10 iris) or None when no face is detected.

    The GUI reads the latest frame via get_latest_frame() without blocking.
    """

    def __init__(self, device_index: int = 0,
                 width: int = 640, height: int = 480,
                 model_path: str = "face_landmarker.task"):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._model_path = model_path
        self._callbacks: list = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_landmarks = None
        self._pending_frame = None       # frame awaiting async result
        self._camera_error: str | None = None

    def register_callback(self, cb: FrameCallback) -> None:
        self._callbacks.append(cb)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="FacePipeline"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_latest_frame(self) -> tuple:
        """Non-blocking read. Returns (frame_bgr, landmarks) or (None, None)."""
        with self._lock:
            return self._latest_frame, self._latest_landmarks

    def get_camera_error(self) -> str | None:
        return self._camera_error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        """MediaPipe Tasks result callback — fires on the pipeline thread."""
        landmarks = result.face_landmarks[0] if result.face_landmarks else None

        with self._lock:
            frame = self._pending_frame
            # _latest_frame is already kept current by the capture loop;
            # only update landmarks here so the GUI always has the freshest image.
            self._latest_landmarks = landmarks

        if frame is None:
            return

        for cb in self._callbacks:
            try:
                cb(frame, landmarks)
            except Exception:
                pass  # never let a callback crash the pipeline

    def _capture_loop(self) -> None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=VisionTaskRunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            result_callback=self._on_result,
        )

        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            self._camera_error = f"Cannot open camera device {self._device_index}"
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        with FaceLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_idx += 1

                # Always make the latest raw frame available for smooth GUI display
                with self._lock:
                    self._latest_frame = frame
                    self._pending_frame = frame

                # Run face analysis on every other frame — halves CPU load while
                # keeping alert responsiveness well within human perception time.
                if frame_idx % 2 != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )
                timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, timestamp_ms)

        cap.release()
