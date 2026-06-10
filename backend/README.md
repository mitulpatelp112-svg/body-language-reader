# Backend services (all optional — the app runs fully on-device without them)

| File | Port | Upgrades | Needs |
|---|---|---|---|
| `server.py` | 8001 | Calibrated FACS action units (py-feat) vs MediaPipe blendshapes | py-feat |
| `voice_adv.py` | 8002 | Real **wav2vec2 A/D/V** voice emotion (MSP-Podcast SOTA) vs interim acoustic heuristic | transformers + torch |
| `explain.py` | 8003 | **Claude** coaching reasoning vs on-device template insight | `ANTHROPIC_API_KEY` |
| `train.py` | — | Trains `app/model.json` from logged data; reports accuracy + macro-F1 | scikit-learn |

Each is a standalone scaffold; turn it on once `eval.html` shows it actually beats the on-device
path. Requirements: `requirements.txt` (FACS), `requirements-voice.txt`, `requirements-explain.txt`,
`requirements-train.txt`.

---

# FACS Backend (py-feat) — calibrated action units

The browser app uses MediaPipe blendshapes (fast, on-device, but *geometric approximations* of
action units). For higher accuracy this backend runs **py-feat**, which outputs **regression-
calibrated AU intensities** and validated emotion probabilities from published FACS models.

## Why add it
- Blendshapes ≈ "how smile-shaped is the mouth"; py-feat AUs ≈ "AU12 intensity, trained on FACS-coded data."
- Gives a second, independent facial estimate → fuse with MediaPipe for robustness.
- Outputs emotion probabilities you can **calibrate** against labeled data (the eval harness).

## Setup
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8001
```
First request downloads the models (a few hundred MB) — slow once, cached after.

## Endpoints
| Path | Input | Returns |
|---|---|---|
| `GET /health` | — | `{ok, model_loaded}` |
| `POST /analyze` | JSON `{image: dataURL}` | live-webcam frame → emotions + AUs + V/A |
| `POST /analyze_image` | multipart `file=<image>` | uploaded photo → emotions + AUs + V/A + `top`/`confidence` + caveats |
| `POST /analyze_video` | multipart `file=<video>` (+ `?fps=2&max_frames=120`) | per-frame timeline + aggregate emotion + caveats |

The browser UI for the upload endpoints lives at **`app/upload.html`**.

## Test
```bash
curl -s localhost:8001/health
curl -s -F "file=@some.jpg"  localhost:8001/analyze_image           | jq .top,.confidence
curl -s -F "file=@some.mp4"  "localhost:8001/analyze_video?fps=2"   | jq .aggregate
```

## Wiring it into the app (next step, not yet enabled)
The app would, every ~500ms (not every frame — this is heavier):
1. grab a frame: `canvas.toDataURL("image/jpeg", 0.7)`
2. `POST` it to `http://localhost:8001/analyze`
3. fuse the returned AUs/emotions with the MediaPipe estimate (e.g., average the two emotion
   vectors, weighted by each one's confidence), then feed the existing fusion/`signals.json` layer.

This is intentionally a **scaffold**: it runs standalone now, but the app still defaults to the
fast on-device path. Flip it on once you've measured (via `eval.html`) that fusion actually beats
MediaPipe-only on your data — don't add latency without proof it helps.

## Performance / deployment notes
- CPU inference is ~0.2–1s/frame → keep it to periodic frames, not the live loop.
- For a product: batch frames, use GPU, or run server-side; never block the UI thread on it.
- Privacy: this sends frames off the browser to your server. Keep it localhost, or get explicit
  consent + encryption for any real deployment (GDPR Art.9 / BIPA — biometric data).
