# Body-Language Reader

A multimodal, on-device reader that detects body-language / paralinguistic signals from webcam +
mic and surfaces **probabilistic, caveated** interpretations — explicitly decision-support, not a
verdict machine. Built on the principle that single nonverbal cues are weak (DePaulo 2003) and
facial expressions are variable (Barrett 2019), so everything is multi-signal, context-weighted,
confidence-scored, and honest about uncertainty.

## System map
```
                 ┌─────────────── MEASUREMENT (on-device) ───────────────┐
  webcam ──▶ MediaPipe Face (478 mesh + 52 blendshapes ≈ action units)   │
        └──▶ MediaPipe Pose (33 body pts)  ·  Hands (21×2 pts)            │
  mic   ──▶ Web Audio: pitch (autocorrelation) + RMS energy (prosody)    │
            (optional) py-feat backend: calibrated FACS AUs ─────────────┘
                                   │  features
                                   ▼
                 ┌─────────────── INTERPRETATION ────────────────────────┐
  quality gating (reject bad frames) · personal-baseline calibration     │
  signals.json (42 signals) → AU constellations = 7 emotion prototypes   │
  evidence-backed CONSTRUCTS (engagement, anxiety, dominance, rapport,   │
    disengagement) — multiple cues vote → confident, corroborated, cited │
  fusion: heuristic + trained model + FACS backend (noisy-OR, weighted)  │
  context reweighting (EMOTIC) · ≥2-signal corroboration rule            │
                                   ▼
  soft-labels + Valence/Arousal/Dominance + "primary read" w/ abstention │
                 └───────────────────────────────────────────────────────┘
```

## Folders
- **`app/`** — the live browser app (`index.html`), eval harness (`eval.html`), shared emotion
  core, in-browser trained-model inference, and a node test. Serve over http and open `/app/`.
- **`knowledge-base/`** — `signals.json`, the interpretation layer (42 signals, each with
  measurable observable + probabilistic interpretations + confidence + caveats + sources).
- **`backend/`** — optional Python services: `server.py` (py-feat calibrated AUs) and `train.py`
  (turns logged data into `app/model.json`).
- **`resources/`** — rated research papers, datasets access guide, reading notes.
- **`resource-catalog.md`** — every source rated on rigor / build value / inference reliability.

## Run
```bash
cd "body language reader"
python3 -m http.server 8000
# open http://localhost:8000/app/   → Start camera (allow camera + mic)
```

## The accuracy loop (how to actually improve it)
1. **Measure** — `app/eval.html` → accuracy, confusion matrix, per-class precision/recall/F1.
2. **Collect** — same page logs (blendshapes → label) as JSONL; repeat across people/lighting.
3. **Train** — `pip install -r backend/requirements-train.txt && python3 backend/train.py data.jsonl`
   → writes `app/model.json` (prints cross-validated P/R/F1).
4. **Use** — reload the app; it auto-loads the model and fuses it in (replaces hand-tuned priors).
5. **Re-measure** — keep the change only if the number went up.

## Test
```bash
node app/test/emotion_test.mjs     # verifies the emotion classifier (8/8)
```

## Honesty & legal
Outputs are probabilistic and caveated. No deception/lie/intent verdicts — the science doesn't
support them. Emotion/biometric inference is regulated (EU AI Act, GDPR Art.9, Illinois BIPA);
data stays on-device unless you enable the backend, which needs explicit consent for any real use.
