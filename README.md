# Parental Control System

[![CI](https://github.com/Marcussi02/ParentalControlSystem/actions/workflows/ci.yml/badge.svg)](https://github.com/Marcussi02/ParentalControlSystem/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

A desktop app that helps parents keep kids' screen time **healthy and safe**. It uses the webcam to notice when a child is **drowsy or not paying attention**, and it checks YouTube videos for **clickbait and inappropriate content** before they play.

All video processing happens **locally on the device**. No webcam footage leaves the machine.

## Features

| Module | What it does | How |
|---|---|---|
| 😴 **Drowsiness detection** | Alerts when eyes stay closed, droop for a long time, or are closed most of the time | Eye Aspect Ratio (EAR) + PERCLOS over a rolling window |
| 👀 **Attention monitor** | Alerts when the child looks away from the screen or leaves the room | Iris position relative to the eye corners (gaze direction) |
| 🎣 **Clickbait scorer** | Scores a YouTube video 0–100 for kids-targeted clickbait | Title signals (keywords, CAPS, punctuation, emoji) + thumbnail signals (saturation, text overlay, excited faces) + combination bonuses |
| 🛡️ **Content safety filter** | Flags age-restricted or inappropriate videos | YouTube age limit + category keyword patterns in title and description + thumbnail skin/blood heuristics |
| 🌐 **Guarded browser** | A real browser window that warns before an unsafe video plays | Playwright drives Edge and injects a full-page warning overlay |
| 🔔 **Alerts & log** | Desktop notifications plus a JSON-lines event history | plyer + `logs/events.json` |

## Architecture

```
 Webcam ──► FacePipeline (MediaPipe FaceMesh, 478 landmarks, background thread)
                │ callbacks per frame
                ├──► DrowsinessDetector ──┐
                └──► AttentionMonitor ────┤
                                          ├──► AlertNotifier ──► desktop toast + events.json
 Browser (Playwright/Edge) ── URL ──► ClickbaitScorer
                                        ├── YouTubeScraper (yt-dlp metadata, no API key)
                                        ├── TitleAnalyzer + ThumbnailAnalyzer  → clickbait score
                                        └── ContentFilter                      → safety score
                                          │
                                   Tkinter Dashboard (live webcam, status, scores, event log)
```

## How detection works

**Eye Aspect Ratio.** For six landmarks around each eye (P1–P6):

```
EAR = (‖P2 − P6‖ + ‖P3 − P5‖) / (2 · ‖P1 − P4‖)
```

Open eyes sit around 0.25–0.35, and closed eyes drop below 0.21. Three signals come from it:

1. **Eyes closed** (EAR < 0.21) continuously for ≥ 3 s triggers a drowsiness alert.
2. **Sleepy eyes** (0.21 ≤ EAR < 0.26, heavy or drooping) for ≥ 15 s triggers a sleepy alert.
3. **PERCLOS**, the share of frames with eyes closed over a 60 s window, reaching ≥ 80% triggers a drowsiness alert. It only fires once a full window has been observed, so a blink at start-up can't read as "100% closed".

Alerts have cooldowns (10 s drowsy, 20 s sleepy), so parents get one notification rather than one per frame. If the child wears glasses, `ear_threshold_offset` shifts the threshold.

**Attention.** The iris centre's horizontal and vertical position between the eye corners gives a ratio from 0 to 1. Within ±0.15 of centre counts as *looking at the screen*. Looking away, or no face at all, for 20 s triggers an alert.

## Getting started

Requires **Windows** for the full app, because the guarded browser launches Microsoft Edge. The detection logic and tests run on any OS.

```bash
pip install -r requirements.txt
playwright install
python scripts/download_model.py     # fetches face_landmarker.task (MediaPipe)
python main.py                       # or run.bat (Windows) / ./run.sh
```

All thresholds live in [`config.yaml`](config.yaml): camera, gaze tolerance, EAR thresholds, PERCLOS window, clickbait and safety thresholds, and log path.

## Tests

The core logic is unit-tested without a webcam, GUI or model. The tests use synthetic face landmarks and a fake clock:

```bash
pip install -r requirements-dev.txt
pytest -v
```

The tests cover:

- the EAR maths
- drowsiness timing (closed, blink, sleepy, PERCLOS window and cooldown)
- gaze classification
- clickbait title scoring and caps
- content-filter weighting
- YouTube URL parsing
- alert logging

## Project structure

```
main.py                  app wiring
config.yaml              all tunable thresholds
modules/eye_tracking/    face pipeline, gaze detector, attention monitor
modules/drowsiness/      EAR calculator, drowsiness detector
modules/clickbait/       scraper, title/thumbnail analysers, content filter, scorer
alerts/                  desktop notifications + JSON event log
ui/                      Tkinter dashboard, Playwright browser
tests/                   unit tests
```

## Limitations and next steps

- The thumbnail heuristics (skin and blood colour ratios) are deliberately simple, so expect false positives. A small image classifier would be the next step.
- The browser is tied to Edge on Windows. Playwright's bundled Chromium would make it cross-platform.
- EAR thresholds vary between people. A short per-child calibration step would improve accuracy.

## License

[MIT](LICENSE) © 2026 Marcus Mah
