# Live Demo — Body-Language Reader

A browser app that runs the **measurement → interpretation** pipeline live on your webcam:

- **Measurement (on-device, fast):** MediaPipe `FaceLandmarker` (52 blendshapes ≈ facial action
  units) + `PoseLandmarker` (body keypoints). No data leaves the browser.
- **Interpretation:** features feed `../knowledge-base/signals.json` → probabilistic soft-labels
  with confidence, evidence strength, contributing signals, and caveats.
- **Honesty by design:** personal-baseline calibration, context reweighting, "needs corroboration"
  flag for single-signal states, and an "insufficient evidence" fallback. No lie/intent verdicts.

## Run it
Must be served over http (camera + ES modules + KB fetch won't work from `file://`):

```bash
cd "body language reader"
python3 -m http.server 8000
# open http://localhost:8000/app/  and click "Start camera"
```

First load downloads the MediaPipe models from Google's CDN (a few MB). Allow camera access.
Hold a neutral face for ~3s while it calibrates your baseline.

## What you'll see
- **Left:** webcam with pose overlay.
- **Right top:** Valence / Arousal / Dominance estimate + ranked soft-labels (confidence bars,
  caveats, which signals contributed).
- **Right bottom:** active signals from the measurement layer with activation % and reliability.
- **Context dropdown:** reweights interpretations (e.g., "Cold environment" suppresses the
  crossed-arms→defensive reading — the EMOTIC context principle in miniature).

## Tracking (3 MediaPipe models, on-device)
- **Face** — 478-point mesh + 52 blendshapes (action-unit proxies). Mesh is drawn live.
- **Body** — full 33-point pose skeleton (drawn).
- **Hands** — 21 points per hand, both hands (drawn).

## Signals detected (42 in the knowledge base; live subset shown)
- **Face — single action units (16):** genuine smile (AU6+12), brow raise (AU1/2),
  brow lower (AU4), frown (AU15), eye-widen (AU5), nose-wrinkle (AU9/10), lip-press (AU23/24/20),
  jaw-drop (AU26), lip-pucker (AU18/22), lip-suck/bite (AU28), chin-raise/pout (AU17),
  smirk/asymmetry (AU12+14 unilateral), jaw-jut (AU29), cheek-puff, blink-rate (AU45), gaze aversion.
- **Face — emotion prototypes (7 AU constellations, higher reliability):** happiness (AU6+12),
  sadness (AU1+4+15+17), surprise (AU1+2+5+26), fear (AU1+2+5+20), anger (AU4+5+7+23),
  disgust (AU9+10+16), contempt (AU12+14 unilateral). These are NOT flagged "needs corroboration"
  because the constellation is already multi-cue — the scientifically grounded way to read emotion.
- **Body (10):** forward lean, backward lean, crossed arms, shoulder shrug→uncertainty,
  hands-on-hips→assertiveness, expansive posture→confidence, head tilt→interest,
  hand-to-neck→self-soothing, fidget→restlessness, head nod.
- **Hands (3):** open palm→openness, hands-together→contemplation/anxiety, pointing→emphasis.
- Each maps to `signals.json` by `id`, carrying confidence + evidence + caveats.

## Readability
The panel **smooths values (EMA) and repaints ~4×/sec**, not every frame — bars ease instead of
flickering. The video overlay still runs at full framerate.

## Known limits (intentional, documented)
- **2D webcam** → posture/lean/expansiveness are approximate (no true depth).
- **No audio yet** → speech/prosody signals (openSMILE/Whisper) are the next module.
- **Power-pose / steeple** claims are weakly supported — flagged `weak` in the KB on purpose.
- Accuracy on real (non-acted) behavior is inherently limited — outputs are support, not truth.

## Accuracy features (added)
- **Quality gating** — each frame gets a tracking-quality score (face frontal? lighting? body
  visible?). Low-quality frames are *damped* so garbage doesn't drive interpretation, and they
  don't pollute the personal baseline. Shown as the "tracking %" pill in the header.
- **Heavier pose model** — upgraded lite → full for better landmark accuracy.
- **Audio / prosody** — mic is captured; pitch (autocorrelation) + RMS energy → vocal arousal and
  disfluency, fused into the `prosody_*` signals and the arousal dimension. Voice independently
  disambiguates arousal the face can't. (Only active while you're speaking.)
- **Eval harness** — `eval.html`: guided "perform each emotion" protocol → accuracy % + confusion
  matrix + CSV export. **This is how you measure whether any change helps.** Use it before/after tuning.
- **FACS backend scaffold** — `../backend/` runs py-feat for calibrated AU intensities; wire in once
  the eval shows fusion beats MediaPipe-only.

## Still to do
1. Gesture classifier on hand landmarks (train on IPN Hand) → richer `gesture_emblem`.
2. Log (features → label) during eval to build your own training set for a custom model.
3. Temporal model (onset/apex/offset) instead of per-frame thresholds.
4. Probability calibration so "70%" means 70%.
