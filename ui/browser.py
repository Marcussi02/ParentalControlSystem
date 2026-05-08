import json
import queue
import threading
from urllib.parse import urlparse, parse_qs

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
YOUTUBE_HOME = "https://www.youtube.com"


class BrowserWindow:
    """
    Launches a real Edge browser window via Playwright for interactive browsing.
    The child sees a full Edge window they can navigate freely.

    on_url_changed(url) fires on the playwright thread whenever the main frame
    navigates. Callers must use root.after(0, ...) before touching tkinter widgets.
    """

    def __init__(self, on_url_changed=None):
        self._on_url_changed = on_url_changed
        self._page = None
        self._thread = None
        self._started = False
        self._warning_queue: queue.Queue = queue.Queue()
        self._warning_done = threading.Event()
        self._warning_result: bool = True  # True = allow, False = block

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="BrowserWindow"
        )
        self._thread.start()

    def navigate(self, url: str) -> None:
        """Ask the browser to navigate (no-op if not running)."""
        if self._page and not self._page.is_closed():
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            except Exception:
                pass

    def is_open(self) -> bool:
        return self._page is not None and not self._page.is_closed()

    def request_warning(self, result: dict) -> bool:
        """
        Thread-safe. Posts a warning request to the browser thread, which
        injects a full-page blocking overlay. Blocks until the user responds.

        Returns True (allow) or False (block). If the user blocks, navigation
        back to YouTube home is handled internally by the browser thread.
        """
        if not self.is_open():
            return True
        self._warning_done.clear()
        self._warning_queue.put(result)
        # Wait up to 5 minutes — if user somehow never clicks, default to allow
        self._warning_done.wait(timeout=300)
        return self._warning_result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    executable_path=EDGE_PATH,
                    headless=False,
                    args=["--window-size=1280,900"],
                )
                context = browser.new_context()
                self._page = context.new_page()
                self._page.on("framenavigated", self._on_navigated)
                self._page.goto(YOUTUBE_HOME)

                while not self._page.is_closed():
                    try:
                        # Drain pending warning before sleeping
                        try:
                            warning_data = self._warning_queue.get_nowait()
                        except queue.Empty:
                            self._page.wait_for_timeout(200)
                            continue

                        # Show overlay — page.evaluate blocks until Promise resolves
                        proceed = self._show_js_warning(warning_data)
                        if not proceed:
                            try:
                                self._page.goto(
                                    YOUTUBE_HOME,
                                    wait_until="domcontentloaded",
                                    timeout=15_000,
                                )
                            except Exception:
                                pass
                        self._warning_result = proceed
                        self._warning_done.set()

                    except Exception:
                        break

                browser.close()
        except Exception as exc:
            print(f"[BrowserWindow] error: {exc}")
        finally:
            self._page = None
            self._started = False
            # Unblock any waiting request_warning calls
            self._warning_result = True
            self._warning_done.set()

    def _show_js_warning(self, result: dict) -> bool:
        """
        Injects a full-page blocking overlay into the browser.
        page.evaluate() waits for the returned Promise to resolve —
        the browser is effectively paused until the user clicks a button.
        Returns True=allow, False=block.
        """
        data = {
            "title": result.get("title", "Unknown video"),
            "safety": result.get("safety_score", 0),
            "cats": ", ".join(result.get("categories_flagged", [])) or "None",
            "flags": "  •  ".join(result.get("safety_flags", [])[:4]),
        }
        js = f"""
        (() => {{
            const d = {json.dumps(data)};
            return new Promise(resolve => {{

                /* Full-viewport dim */
                const ov = document.createElement('div');
                ov.style.cssText = [
                    'position:fixed', 'top:0', 'left:0',
                    'width:100%', 'height:100%',
                    'background:rgba(0,0,0,0.88)',
                    'z-index:2147483647',
                    'display:flex', 'align-items:center', 'justify-content:center',
                    'font-family:"Segoe UI",Arial,sans-serif'
                ].join(';');

                /* Dialog card */
                const box = document.createElement('div');
                box.style.cssText = [
                    'background:#1e1e2e',
                    'border:2px solid #e74c3c',
                    'border-radius:14px',
                    'padding:36px 40px',
                    'max-width:560px', 'width:90%',
                    'text-align:center',
                    'color:#cdd6f4',
                    'box-shadow:0 8px 40px rgba(0,0,0,0.7)'
                ].join(';');

                /* Helper: create element with style + textContent */
                const mk = (tag, css, text) => {{
                    const el = document.createElement(tag);
                    if (css)  el.style.cssText = css;
                    if (text !== undefined) el.textContent = text;
                    return el;
                }};

                box.appendChild(mk('div',
                    'color:#e74c3c;font-size:26px;font-weight:bold;margin-bottom:14px',
                    '⚠️  Unsafe Content Detected'));

                box.appendChild(mk('div',
                    'font-size:14px;margin-bottom:12px;color:#cdd6f4', d.title));

                box.appendChild(mk('div',
                    'color:#e74c3c;font-size:15px;font-weight:bold;margin-bottom:6px',
                    'Safety Score: ' + d.safety + ' / 100'));

                box.appendChild(mk('div',
                    'font-size:13px;color:#fab387;margin-bottom:' + (d.flags ? '6px' : '20px'),
                    'Categories: ' + d.cats));

                if (d.flags) {{
                    box.appendChild(mk('div',
                        'font-size:12px;color:#a6adc8;margin-bottom:20px', d.flags));
                }}

                box.appendChild(mk('div',
                    'font-size:15px;margin-bottom:28px;line-height:1.5',
                    'This video may not be appropriate for children. Do you want to continue watching?'));

                /* Buttons */
                const mkBtn = (text, bg, fg) => {{
                    const b = document.createElement('button');
                    b.textContent = text;
                    b.style.cssText = [
                        'background:' + bg, 'color:' + fg,
                        'border:none', 'padding:12px 28px',
                        'border-radius:8px', 'font-size:15px',
                        'font-weight:bold', 'cursor:pointer', 'margin:0 8px'
                    ].join(';');
                    b.onmouseover = () => b.style.opacity = '0.85';
                    b.onmouseout  = () => b.style.opacity = '1';
                    return b;
                }};

                const blockBtn = mkBtn('Block & Go Back', '#e74c3c', '#ffffff');
                const allowBtn = mkBtn('Allow Once',        '#f39c12', '#1e1e2e');

                blockBtn.onclick = () => {{ ov.remove(); resolve(false); }};
                allowBtn.onclick = () => {{ ov.remove(); resolve(true); }};

                const row = document.createElement('div');
                row.appendChild(blockBtn);
                row.appendChild(allowBtn);
                box.appendChild(row);

                ov.appendChild(box);
                document.body.appendChild(ov);
            }});
        }})()
        """
        try:
            # evaluate() awaits the Promise — browser is paused here until click
            self._page.set_default_timeout(0)  # no timeout while waiting for user
            return bool(self._page.evaluate(js))
        except Exception:
            return True  # allow on error (e.g. browser closed mid-dialog)
        finally:
            try:
                self._page.set_default_timeout(30_000)
            except Exception:
                pass

    def _on_navigated(self, frame) -> None:
        try:
            if frame == self._page.main_frame and self._on_url_changed:
                self._on_url_changed(frame.url)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_video_id(url: str) -> str | None:
        """Return YouTube video ID if the URL is a watch page, else None."""
        try:
            parsed = urlparse(url)
            if "youtube.com" in parsed.netloc and parsed.path == "/watch":
                ids = parse_qs(parsed.query).get("v", [])
                return ids[0] if ids else None
            if "youtu.be" in parsed.netloc:
                vid = parsed.path.lstrip("/")
                return vid if vid else None
        except Exception:
            pass
        return None
