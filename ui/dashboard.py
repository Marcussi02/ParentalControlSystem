import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk


# Score band colors
def _score_color(score: int) -> str:
    if score < 30:
        return "#27ae60"  # green
    if score < 60:
        return "#f39c12"  # orange
    return "#e74c3c"      # red


def _status_color(ok: bool) -> str:
    return "#27ae60" if ok else "#e74c3c"


def _safety_color(score: int) -> str:
    if score < 30:
        return "#27ae60"   # green — safe
    if score < 60:
        return "#f39c12"   # orange — concerning
    return "#e74c3c"       # red — inappropriate


class Dashboard:
    """
    Main tkinter GUI. Integrates webcam feed, attention/drowsiness status,
    and YouTube clickbait checker in a single window.

    Threading rules:
    - Widget updates happen ONLY from the main thread via root.after().
    - Clickbait scoring (network I/O) runs in a worker thread; result is
      posted back via root.after(0, callback).
    - get_latest_frame() reads a lock-guarded slot — safe from main thread.
    """

    def __init__(self, root: tk.Tk, face_pipeline,
                 attention_monitor, drowsiness_detector,
                 clickbait_scorer, alert_notifier, config: dict):
        self._root = root
        self._pipeline = face_pipeline
        self._attention = attention_monitor
        self._drowsiness = drowsiness_detector
        self._scorer = clickbait_scorer
        self._notifier = alert_notifier
        self._cfg = config
        self._running = True
        self._photo_ref = None  # prevent GC of PhotoImage
        self._browser = None    # set later via set_browser()
        self._warning_open = False

        self._build_ui()
        self._load_alert_history()

    def set_browser(self, browser) -> None:
        """Wire up the BrowserWindow after construction."""
        self._browser = browser

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._root.configure(bg="#1e1e2e")
        self._root.resizable(False, False)

        main = tk.Frame(self._root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # ---- Left panel: webcam feed ----
        left = tk.Frame(main, bg="#1e1e2e")
        left.pack(side="left", padx=(0, 10))

        tk.Label(
            left, text="Live Feed", bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 4))

        self._video_label = tk.Label(left, bg="#000000", width=640, height=480)
        self._video_label.pack()

        cam_err = self._pipeline.get_camera_error()
        if cam_err:
            tk.Label(
                left, text=f"Camera error: {cam_err}",
                bg="#1e1e2e", fg="#e74c3c", font=("Segoe UI", 9)
            ).pack(pady=4)

        # ---- Right panel ----
        right = tk.Frame(main, bg="#1e1e2e", width=420)
        right.pack(side="left", fill="both", expand=True)
        right.pack_propagate(False)

        self._build_status_panel(right)
        self._build_browser_panel(right)
        self._build_clickbait_panel(right)
        self._build_log_panel(right)

    def _build_status_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="Monitoring Status", bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 10, "bold"), bd=1, relief="groove"
        )
        frame.pack(fill="x", pady=(0, 8))

        # Attention
        row = tk.Frame(frame, bg="#1e1e2e")
        row.pack(fill="x", padx=8, pady=4)
        tk.Label(row, text="Attention:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        self._attention_label = tk.Label(
            row, text="STARTING…", bg="#1e1e2e", fg="#f5c2e7",
            font=("Segoe UI", 9, "bold"), anchor="w"
        )
        self._attention_label.pack(side="left")

        # Gaze direction
        row2 = tk.Frame(frame, bg="#1e1e2e")
        row2.pack(fill="x", padx=8, pady=2)
        tk.Label(row2, text="Gaze:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        self._gaze_label = tk.Label(
            row2, text="—", bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 9)
        )
        self._gaze_label.pack(side="left")

        # Drowsiness
        row3 = tk.Frame(frame, bg="#1e1e2e")
        row3.pack(fill="x", padx=8, pady=4)
        tk.Label(row3, text="Drowsiness:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        self._drowsy_label = tk.Label(
            row3, text="STARTING…", bg="#1e1e2e", fg="#f5c2e7",
            font=("Segoe UI", 9, "bold"), anchor="w"
        )
        self._drowsy_label.pack(side="left")

        # EAR value
        row4 = tk.Frame(frame, bg="#1e1e2e")
        row4.pack(fill="x", padx=8, pady=2)
        tk.Label(row4, text="EAR:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        self._ear_label = tk.Label(
            row4, text="—", bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 9)
        )
        self._ear_label.pack(side="left")

        # PERCLOS bar
        row5 = tk.Frame(frame, bg="#1e1e2e")
        row5.pack(fill="x", padx=8, pady=(2, 6))
        tk.Label(row5, text="PERCLOS:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        self._perclos_bar = ttk.Progressbar(
            row5, maximum=100, length=180, mode="determinate"
        )
        self._perclos_bar.pack(side="left", padx=(0, 6))
        self._perclos_pct_label = tk.Label(
            row5, text="0%", bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 9)
        )
        self._perclos_pct_label.pack(side="left")

    def _build_browser_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="Browser Control", bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 10, "bold"), bd=1, relief="groove"
        )
        frame.pack(fill="x", pady=(0, 8))

        # Open browser button
        btn_row = tk.Frame(frame, bg="#1e1e2e")
        btn_row.pack(fill="x", padx=8, pady=(6, 2))

        self._open_browser_btn = tk.Button(
            btn_row, text="Open Browser",
            command=self._open_browser,
            bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=10, cursor="hand2"
        )
        self._open_browser_btn.pack(side="left", padx=(0, 8))

        self._browser_status_label = tk.Label(
            btn_row, text="Not started", bg="#1e1e2e",
            fg="#6c7086", font=("Segoe UI", 8)
        )
        self._browser_status_label.pack(side="left")

        # Current URL display
        url_row = tk.Frame(frame, bg="#1e1e2e")
        url_row.pack(fill="x", padx=8, pady=2)
        tk.Label(url_row, text="URL:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=5, anchor="w").pack(side="left")
        self._browser_url_label = tk.Label(
            url_row, text="—", bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 8), anchor="w", wraplength=320, justify="left"
        )
        self._browser_url_label.pack(side="left", fill="x", expand=True)

        # Navigate-to box
        nav_row = tk.Frame(frame, bg="#1e1e2e")
        nav_row.pack(fill="x", padx=8, pady=(2, 6))
        tk.Label(nav_row, text="Go to:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=5, anchor="w").pack(side="left")
        self._nav_entry = tk.Entry(
            nav_row, width=30, bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief="flat", font=("Segoe UI", 8)
        )
        self._nav_entry.pack(side="left", padx=(0, 6))
        self._nav_entry.bind("<Return>", lambda _: self._browser_navigate())
        tk.Button(
            nav_row, text="Go", command=self._browser_navigate,
            bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 8, "bold"),
            relief="flat", padx=6, cursor="hand2"
        ).pack(side="left")

        # Auto-detect score row
        score_row = tk.Frame(frame, bg="#1e1e2e")
        score_row.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(score_row, text="Auto-score:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9), width=10, anchor="w").pack(side="left")
        self._auto_score_label = tk.Label(
            score_row, text="—", bg="#1e1e2e", fg="#6c7086",
            font=("Segoe UI", 9, "bold")
        )
        self._auto_score_label.pack(side="left")

    # ------------------------------------------------------------------
    # Browser actions
    # ------------------------------------------------------------------

    def _open_browser(self) -> None:
        if self._browser is None:
            return
        if self._browser.is_open():
            self._browser_status_label.configure(text="Already open", fg="#a6e3a1")
            return
        self._browser.start()
        self._browser_status_label.configure(text="Opening…", fg="#f9e2af")
        self._open_browser_btn.configure(state="disabled")
        # Re-enable after a moment
        self._root.after(3000, lambda: self._open_browser_btn.configure(state="normal"))

    def _browser_navigate(self) -> None:
        if self._browser is None or not self._browser.is_open():
            self._browser_status_label.configure(text="Browser not open", fg="#e74c3c")
            return
        url = self._nav_entry.get().strip()
        if not url:
            return
        if not url.startswith("http"):
            url = "https://" + url
        self._browser.navigate(url)
        self._nav_entry.delete(0, "end")

    def on_browser_url(self, url: str) -> None:
        """
        Called from main.py (via root.after) when the browser navigates.
        Always runs on the main thread — safe to update widgets.
        """
        # Truncate long URLs for display
        display = url if len(url) <= 60 else url[:57] + "…"
        self._browser_url_label.configure(text=display)
        self._browser_status_label.configure(text="Active", fg="#a6e3a1")

        # Auto-check clickbait if the user landed on a YouTube video
        from ui.browser import BrowserWindow
        video_id = BrowserWindow.extract_video_id(url)
        if video_id:
            self._auto_score_label.configure(text="Checking…", fg="#f9e2af")
            captured_url = url
            def worker():
                try:
                    result = self._scorer.score_url(captured_url)
                    result["_url"] = captured_url
                except Exception as exc:
                    result = {"error": str(exc)}
                self._root.after(0, lambda: self._show_auto_score(result))
            threading.Thread(target=worker, daemon=True).start()
        else:
            self._auto_score_label.configure(text="—", fg="#6c7086")

    def _show_auto_score(self, result: dict) -> None:
        if "error" in result:
            self._auto_score_label.configure(
                text=f"Error: {result['error'][:40]}", fg="#e74c3c"
            )
            return

        score = result["final_score"]
        safety = result.get("safety_score", 0)
        is_clickbait = result["is_clickbait"]
        is_inappropriate = result.get("is_inappropriate", False)
        title = result.get("title", "")

        # Worst of the two determines the display color
        worst_color = _safety_color(safety) if is_inappropriate else _score_color(score)
        parts = []
        if is_clickbait:
            parts.append(f"CB:{score}")
        if is_inappropriate:
            cats = result.get("categories_flagged", [])
            parts.append(f"UNSAFE:{safety} ({', '.join(cats[:2])})")
        if not parts:
            parts.append(f"CB:{score} OK | Safe")

        self._auto_score_label.configure(text="  ".join(parts), fg=worst_color)

        if is_clickbait:
            self._notifier.send(
                "CLICKBAIT",
                f"Clickbait while browsing (score {score}): {title}",
                {"score": score, "video_id": result.get("video_id"), "flags": result.get("flags", [])},
            )
            self._append_alert_log({
                "timestamp": datetime.now().isoformat(),
                "alert_type": "CLICKBAIT",
                "message": f"Browser CB:{score} — {title[:50]}",
            })

        if is_inappropriate:
            cats = result.get("categories_flagged", [])
            self._notifier.send(
                "INAPPROPRIATE",
                f"Inappropriate content while browsing (score {safety}): {title}",
                {"safety_score": safety, "categories": cats,
                 "flags": result.get("safety_flags", [])},
            )
            self._append_alert_log({
                "timestamp": datetime.now().isoformat(),
                "alert_type": "INAPPROPRIATE",
                "message": f"Browser Safety:{safety} [{', '.join(cats[:2])}] — {title[:40]}",
            })
            self._show_unsafe_warning(result)

    # ------------------------------------------------------------------
    # Unsafe content warning (shown inside the browser window)
    # ------------------------------------------------------------------

    def _show_unsafe_warning(self, result: dict) -> None:
        """
        Injects a blocking overlay directly into the browser page.
        Runs the blocking request_warning() call on a worker thread so tkinter
        stays responsive. The browser thread owns the page interaction and
        handles navigation back to YouTube if the user clicks Block.
        """
        if self._warning_open or self._browser is None or not self._browser.is_open():
            return
        self._warning_open = True

        def worker():
            self._browser.request_warning(result)  # blocks until user clicks
            self._root.after(0, lambda: setattr(self, "_warning_open", False))

        threading.Thread(target=worker, daemon=True, name="WarningWorker").start()

    def _build_clickbait_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="YouTube Clickbait Radar", bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 10, "bold"), bd=1, relief="groove"
        )
        frame.pack(fill="x", pady=(0, 8))

        url_row = tk.Frame(frame, bg="#1e1e2e")
        url_row.pack(fill="x", padx=8, pady=6)
        tk.Label(url_row, text="YouTube URL:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        self._url_entry = tk.Entry(
            url_row, width=28, bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief="flat", font=("Segoe UI", 9)
        )
        self._url_entry.pack(side="left", padx=(0, 6))
        self._url_entry.bind("<Return>", lambda _: self._check_youtube_url())

        self._check_btn = tk.Button(
            url_row, text="Check", command=self._check_youtube_url,
            bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, cursor="hand2"
        )
        self._check_btn.pack(side="left")

        # Result area
        result_row = tk.Frame(frame, bg="#1e1e2e")
        result_row.pack(fill="x", padx=8, pady=(0, 4))

        # Clickbait score bar
        tk.Label(result_row, text="Clickbait", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 7)).pack(anchor="w")
        self._score_canvas = tk.Canvas(
            result_row, height=18, bg="#313244", highlightthickness=0, width=390
        )
        self._score_canvas.pack(fill="x", pady=(0, 2))

        self._score_label = tk.Label(
            result_row, text="Enter a YouTube URL above to check",
            bg="#1e1e2e", fg="#6c7086", font=("Segoe UI", 8), wraplength=390,
            justify="left"
        )
        self._score_label.pack(anchor="w")

        self._flags_label = tk.Label(
            result_row, text="", bg="#1e1e2e", fg="#fab387",
            font=("Segoe UI", 8), wraplength=390, justify="left"
        )
        self._flags_label.pack(anchor="w")

        # Content safety score bar
        tk.Label(result_row, text="Content Safety", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 7)).pack(anchor="w", pady=(4, 0))
        self._safety_canvas = tk.Canvas(
            result_row, height=18, bg="#313244", highlightthickness=0, width=390
        )
        self._safety_canvas.pack(fill="x", pady=(0, 2))

        self._safety_label = tk.Label(
            result_row, text="", bg="#1e1e2e", fg="#6c7086",
            font=("Segoe UI", 8), wraplength=390, justify="left"
        )
        self._safety_label.pack(anchor="w")

        self._safety_flags_label = tk.Label(
            result_row, text="", bg="#1e1e2e", fg="#f38ba8",
            font=("Segoe UI", 8), wraplength=390, justify="left"
        )
        self._safety_flags_label.pack(anchor="w")

    def _build_log_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="Alert History", bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 10, "bold"), bd=1, relief="groove"
        )
        frame.pack(fill="both", expand=True)

        self._log_text = scrolledtext.ScrolledText(
            frame, height=10, bg="#181825", fg="#cdd6f4",
            font=("Consolas", 8), state="disabled", relief="flat",
            insertbackground="#cdd6f4"
        )
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Frame update loop (main thread, called every ~33ms)
    # ------------------------------------------------------------------

    def _update_frame(self) -> None:
        if not self._running:
            return

        frame, landmarks = self._pipeline.get_latest_frame()
        if frame is not None:
            annotated = self._draw_overlay(frame.copy(), landmarks)
            img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).resize((640, 480))
            photo = ImageTk.PhotoImage(pil_img)
            self._video_label.configure(image=photo)
            self._photo_ref = photo  # prevent GC

        self._update_status_indicators()
        self._root.after(self._cfg["gui"]["webcam_refresh_ms"], self._update_frame)

    def _draw_overlay(self, frame: np.ndarray, landmarks) -> np.ndarray:
        if landmarks:
            h, w = frame.shape[:2]
            self._draw_face_landmarks(frame, landmarks, w, h)

        # Overlay EAR value
        d_status = self._drowsiness.status
        ear = d_status.get("ear")
        if ear is not None:
            color = (0, 255, 0) if not d_status.get("drowsy") else (0, 0, 255)
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Overlay gaze direction
        a_status = self._attention.status
        direction = a_status.get("gaze", {}).get("direction", "unknown")
        cv2.putText(frame, f"Gaze: {direction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        return frame

    # Face oval and eye contour landmark index groups
    _FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
        288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
        150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    ]
    _LEFT_EYE_CONTOUR  = [33, 160, 158, 133, 153, 144, 33]
    _RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 380, 373, 362]
    _LEFT_IRIS  = 468
    _RIGHT_IRIS = 473

    def _draw_face_landmarks(self, frame: np.ndarray, landmarks: list,
                              w: int, h: int) -> None:
        def pt(idx):
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)

        # Face oval
        oval_pts = [pt(i) for i in self._FACE_OVAL]
        for i in range(len(oval_pts) - 1):
            cv2.line(frame, oval_pts[i], oval_pts[i + 1], (80, 200, 80), 1, cv2.LINE_AA)

        # Eye contours
        for eye_indices in (self._LEFT_EYE_CONTOUR, self._RIGHT_EYE_CONTOUR):
            eye_pts = [pt(i) for i in eye_indices]
            for i in range(len(eye_pts) - 1):
                cv2.line(frame, eye_pts[i], eye_pts[i + 1], (0, 220, 255), 1, cv2.LINE_AA)

        # Iris centers (only present when model includes iris — 478 landmarks)
        if len(landmarks) > self._LEFT_IRIS:
            cv2.circle(frame, pt(self._LEFT_IRIS),  3, (0, 140, 255), -1, cv2.LINE_AA)
        if len(landmarks) > self._RIGHT_IRIS:
            cv2.circle(frame, pt(self._RIGHT_IRIS), 3, (0, 140, 255), -1, cv2.LINE_AA)

    def _update_status_indicators(self) -> None:
        # Attention
        a = self._attention.status
        away_sec = a.get("looking_away_duration", 0)
        no_face_sec = a.get("no_face_duration", 0)
        attention_ok = a.get("attention_ok", True)
        direction = a.get("gaze", {}).get("direction", "unknown")

        if no_face_sec > 1:
            att_text = f"NO FACE ({no_face_sec:.0f}s)"
            att_color = "#e74c3c"
        elif away_sec > 0:
            att_text = f"LOOKING AWAY ({away_sec:.0f}s)"
            att_color = "#f39c12"
        else:
            att_text = "ATTENTIVE"
            att_color = "#27ae60"

        self._attention_label.configure(text=att_text, fg=att_color)
        self._gaze_label.configure(text=direction.upper())

        # Drowsiness
        d = self._drowsiness.status
        ear = d.get("ear")
        perclos = d.get("perclos", 0.0)
        drowsy = d.get("drowsy", False)
        eyes_sleepy = d.get("eyes_sleepy", False)
        sleepy_dur = d.get("sleepy_duration", 0.0)

        if drowsy:
            dr_text = f"DROWSY ({d.get('alert_reason', '')})"
            dr_color = "#e74c3c"
        elif eyes_sleepy:
            dr_text = f"SLEEPY EYES ({sleepy_dur:.0f}s)"
            dr_color = "#f39c12"
        else:
            dr_text = "ALERT"
            dr_color = "#27ae60"

        self._drowsy_label.configure(text=dr_text, fg=dr_color)
        ear_text = f"{ear:.3f}" if ear is not None else "—"
        self._ear_label.configure(text=ear_text)
        self._perclos_bar["value"] = perclos * 100
        self._perclos_pct_label.configure(text=f"{perclos:.0%}")

    # ------------------------------------------------------------------
    # Clickbait checker
    # ------------------------------------------------------------------

    def _check_youtube_url(self) -> None:
        url = self._url_entry.get().strip()
        if not url:
            return
        self._check_btn.configure(state="disabled", text="Checking…")
        self._score_label.configure(text="Fetching video info…", fg="#6c7086")
        self._flags_label.configure(text="")
        self._score_canvas.delete("all")
        self._safety_canvas.delete("all")
        self._safety_label.configure(text="")
        self._safety_flags_label.configure(text="")

        def worker():
            try:
                result = self._scorer.score_url(url)
            except Exception as exc:
                result = {"error": str(exc)}
            self._root.after(0, lambda: self._display_score_result(result))

        threading.Thread(target=worker, daemon=True).start()

    def _display_score_result(self, result: dict) -> None:
        self._check_btn.configure(state="normal", text="Check")
        if "error" in result:
            self._score_label.configure(
                text=f"Error: {result['error']}", fg="#e74c3c"
            )
            return

        title = result.get("title", "")
        channel = result.get("channel", "")

        # --- Clickbait score bar ---
        score = result["final_score"]
        is_clickbait = result["is_clickbait"]
        flags = result.get("flags", [])
        color = _score_color(score)

        self._score_canvas.delete("all")
        bar_w = int(390 * score / 100)
        self._score_canvas.create_rectangle(0, 0, bar_w, 18, fill=color, outline="")
        self._score_canvas.create_text(
            195, 9, text=f"{score}/100",
            fill="white", font=("Segoe UI", 7, "bold")
        )
        verdict = "CLICKBAIT DETECTED" if is_clickbait else "OK"
        self._score_label.configure(
            text=f"{verdict}  |  {title[:55]}  [{channel}]", fg=color
        )
        self._flags_label.configure(text=" · ".join(flags[:5]))

        # --- Content safety score bar ---
        safety = result.get("safety_score", 0)
        is_inappropriate = result.get("is_inappropriate", False)
        safety_flags = result.get("safety_flags", [])
        s_color = _safety_color(safety)

        self._safety_canvas.delete("all")
        s_bar_w = int(390 * safety / 100)
        self._safety_canvas.create_rectangle(0, 0, s_bar_w, 18, fill=s_color, outline="")
        self._safety_canvas.create_text(
            195, 9, text=f"{safety}/100",
            fill="white", font=("Segoe UI", 7, "bold")
        )
        s_verdict = "INAPPROPRIATE" if is_inappropriate else "Safe"
        cats = result.get("categories_flagged", [])
        cat_text = f"  ({', '.join(cats[:3])})" if cats else ""
        self._safety_label.configure(
            text=f"{s_verdict}{cat_text}", fg=s_color
        )
        self._safety_flags_label.configure(text=" · ".join(safety_flags[:4]))

        # --- Alerts ---
        if is_clickbait:
            self._notifier.send(
                "CLICKBAIT",
                f"Clickbait detected (score {score}): {title}",
                {"video_id": result.get("video_id"), "score": score, "flags": flags},
            )
            self._append_alert_log({
                "timestamp": datetime.now().isoformat(),
                "alert_type": "CLICKBAIT",
                "message": f"Score {score}: {title[:50]}",
            })

        if is_inappropriate:
            self._notifier.send(
                "INAPPROPRIATE",
                f"Inappropriate content detected (score {safety}): {title}",
                {"video_id": result.get("video_id"), "safety_score": safety,
                 "categories": cats, "flags": safety_flags},
            )
            self._append_alert_log({
                "timestamp": datetime.now().isoformat(),
                "alert_type": "INAPPROPRIATE",
                "message": f"Safety {safety} [{', '.join(cats[:3])}]: {title[:40]}",
            })

    # ------------------------------------------------------------------
    # Alert log
    # ------------------------------------------------------------------

    def _load_alert_history(self) -> None:
        events = self._notifier.get_recent_events(50)
        for ev in events:
            self._append_alert_log(ev)

    def _append_alert_log(self, event: dict) -> None:
        ts = event.get("timestamp", "")[:19].replace("T", " ")
        kind = event.get("alert_type", "")
        msg = event.get("message", "")
        line = f"[{ts}] {kind}: {msg}\n"
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._root.after(
            self._cfg["gui"]["webcam_refresh_ms"], self._update_frame
        )
        self._root.mainloop()

    def stop(self) -> None:
        self._running = False
