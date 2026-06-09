# Reading Notes — What Each Resource Gives the Product

Status: read & understood 2026-06-01. These notes translate each source into "what it
contributes to the build" and "how much to trust its claims."

## Your added files (identified)

| File | Actually is | Rigor | Role |
|---|---|:--:|---|
| `1972-29107-001.pdf` | **Wiener, DeVoe, Rubinow & Geller (1972), "Nonverbal Behavior and Nonverbal Communication," *Psychological Review* 79(3)** | High (peer-reviewed classic) | Theoretical backbone. Its core point: **not all nonverbal behavior is "communication."** It separates behavior that *encodes a shared message* from behavior that's just idiosyncratic/expressive. Critical design principle — our app must not treat every movement as meaningful signal. |
| `HowToReadAPersonLikeABook .pdf` | **Nierenberg & Calero (1971)** popular trade book | **Low** (pop-psych, no empirical validation) | Useful only as a **raw gesture-cluster taxonomy** (e.g., "openness cluster," "defensiveness cluster"). Mine it for *candidate* gesture→attitude hypotheses, but every claim must be flagged unverified. Do NOT cite as evidence. |
| `user_unknown_62pg.pdf` | **Ekman & Friesen (1969), "The Repertoire of Nonverbal Behavior: Categories, Origins, Usage, and Coding," *Semiotica* 1(1):49–98** (scanned; now OCR'd → `notes/ekman_friesen_ocr.txt`) | **Very high** (foundational) | THE organizing framework. Defines 5 categories: **emblems, illustrators, affect displays, regulators, adaptors.** This is the backbone of our knowledge base. |

## Core academic papers (downloaded)

**AffectNet (2017)** — 1M+ in-the-wild faces; ~half labeled for 7 discrete emotions + continuous **valence/arousal**. Key for us: it pioneered the *dimensional* (valence/arousal) model alongside categories. Build lesson: prefer valence/arousal over rigid emotion labels — it's more honest and more granular.

**AffectNet+ (2024)** — adds **soft-labels** (an image carries *multiple* emotions with confidences) + metadata (age/gender/ethnicity/head-pose/landmarks). This is the single most important *design pattern* for us: **multi-label with confidence, not one hard verdict.** Exactly the honest-output model the product should adopt.

**EMOTIC (2020)** — people in natural scenes, 26 emotion categories + Valence/Arousal/**Dominance**. Proves **scene context materially improves emotion recognition** (face+body alone underperforms in the wild). Validates your "external conditions" axis as a real, measurable edge — not just a nice-to-have.

**IEMOCAP (2008)** — 12h dyadic acted emotion, mocap on face/head/hands + speech. The reference multimodal corpus showing tone + facial + posture + gaze must be **jointly** modeled. Note: *acted* emotions (a known limitation — less authentic than spontaneous).

**CMU-MOSEI transformer study (2025)** — modern recipe: text (from ASR) + acoustic + visual features → **early fusion** → transformer classifier. This is essentially our interpretation-layer architecture blueprint.

**IPN Hand (2020)** — 4k+ samples, 800k frames, **continuous** real-time gesture recognition with non-gesture "noise" actions and 3D-CNN baselines. Most directly relevant to a *live* app: it handles the hard part — spotting gestures in a continuous stream, not pre-segmented clips.

**H3WB / Human3.6M (2022)** — 3D whole-body keypoint benchmark. For later: upgrading from 2D to 3D pose.

## Cross-cutting synthesis (what the product should be)

1. **Two-layer architecture is non-negotiable.** Measurement (pose/AU/prosody — high confidence) is separate from Interpretation (emotion/intent — probabilistic).
2. **Output soft-labels + confidence + valence/arousal/dominance**, never a single verdict. (AffectNet+ / EMOTIC pattern.)
3. **Fuse modalities + context.** Face alone is weak in the wild; context adds 5–7%. (EMOTIC.)
4. **Distinguish communicative vs. non-communicative movement.** (Wiener 1972.) Filter noise before interpreting.
5. **Trade-book taxonomies (Nierenberg) are hypotheses, not evidence.** Tag accordingly in the knowledge base.
6. **Acted vs. spontaneous gap.** Most datasets are acted; real-world accuracy will be lower. Set expectations.

## Open items
- [ ] OCR `user_unknown_62pg.pdf` to identify & use it.
- [ ] Datasets (AffectNet/EMOTIC/IEMOCAP/MOSEI) are license-gated — need user to register (see `resources/datasets/README`).
- [ ] Decide build vs. taxonomy-first for next step.

## Newly added trade books (2026-06-08) + what was implemented

| File | Is | Rigor | Role |
|---|---|:--:|---|
| `what-everybody-is-saying.pdf` | **Joe Navarro (2008), "What Every BODY is Saying"** (ex-FBI). Head-to-toe taxonomy organized around the **limbic comfort/discomfort** model. | **Low-moderate** (experiential / field-derived, not peer-reviewed RCTs) | Mined for *measurable* additions, tagged accordingly. |
| `Allan_and_Barbara_Pease_-_Body_Language_The_Definitive_Book.pdf` | **Pease & Pease, "The Definitive Book of Body Language"** popular trade book. | **Low** (pop-psych) | Reference only; not used as evidence. |

**Implemented from Navarro (the measurable, defensible parts):**
- **New signal `posture_ventral_denial`** — torso turning its front (ventral) side *away* from the
  interlocutor. Navarro's central, distinctive, and genuinely measurable cue (shoulder-line rotation
  via pose z-difference). Wired into the constructs: it raises **Discomfort/Anxiety** and
  **Disengagement/Withdrawal**, and lowers **Engagement** and **Rapport** (ventral *fronting* =
  comfort). Rated reliability 3/5, evidence "moderate", with the caveat that it's field-derived and
  also caused by simply shifting or addressing someone else.
- **Pacifying-behaviour framing** added to the Discomfort construct: neck/suprasternal touch
  (`adaptor_hand_to_neck`, weight bumped), face stroke (`adaptor_self_touch_face`), and cheek-puff
  exhale (`face_cheek_puff`) are now cited as Navarro's limbic stress cluster.

**Deliberately NOT implemented:** feet/leg "tells" (usually off-camera), pupil dilation (not
reliably measurable via webcam), and any single-cue "this means lying" claims (against the project's
honesty rules and unsupported by the science).
