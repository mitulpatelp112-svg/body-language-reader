# Knowledge Base — the product's interpretation layer

`signals.json` is the "brain": it maps **objectively measurable observables** → **probabilistic
interpretations** with confidence, required context, and caveats. It is grounded in the resources
we read (see `../resources/notes/reading-notes.md` and `../resource-catalog.md`).

## Why it's built this way
The honest science (DePaulo 2003, Barrett 2019) says single nonverbal cues are weak. So this KB is
deliberately **probabilistic, multi-signal, and context-weighted** — never a verdict machine.

## Organizing framework — Ekman & Friesen (1969), 5 categories
- **emblem** — direct verbal translation, culture-specific (highest confidence, must localize)
- **illustrator** — accompanies speech
- **affect_display** — emotion expression (real but variable)
- **regulator** — manages conversation flow
- **adaptor** — self/object touch (popularly over-read; weak evidence)

## Each signal entry
| Field | Meaning |
|---|---|
| `measurable_via` | The exact tool/feature that detects it (MediaPipe / OpenFace AU / openSMILE / Whisper) — keeps the measurement layer concrete |
| `observable` | What is physically seen/heard |
| `interpretations[]` | Candidate states, each with `prior_confidence`, `evidence` strength, `requires_context`, and `caveats` |
| `inference_reliability` | 1–5, how much to trust the meaning (matches resource-catalog.md) |
| `sources` | Where the claim comes from (and whether it's evidence or folklore) |

## How the app uses it
1. Measurement layer emits observables per frame/window.
2. Look up matching signals, gather candidate interpretations.
3. Apply `fusion_policy`: weight by reliability, reweight by `context_scene_modifier`, calibrate to the person's baseline.
4. Emit ranked **soft-labels + valence/arousal/dominance + confidence**, listing contributing signals.
5. Obey `hard_constraints` — no lie/intent verdicts, require ≥2 corroborating signals, surface "insufficient evidence."

## Current status
- 13 starter signals across all 7 modality tags (facial, body, gesture, subtle_movement, speech_prosody, speech_content, external_context).
- This is a **seed**. Grows as we (a) OCR-mine the Ekman paper for more adaptor/illustrator detail, (b) add culture lookup tables for emblems, (c) tune priors against real labeled data when datasets arrive.

## Honesty / legal note
Every "weak"/"folklore" entry is tagged as such on purpose. For a market product this transparency is
both ethically right and legally protective (EU AI Act limits emotion/intent inference; GDPR Art.9 / BIPA
cover biometric data).
