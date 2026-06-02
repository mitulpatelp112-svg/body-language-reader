# Nonverbal / Behavioral Reading — Rated Resource Catalog

Compiled 2026-06-01. Covers the modalities you listed: **gesture, tone/speech, external
conditions/context, body pattern & poses, facial reactions, subtle movement.**

## How each source is rated (1–5)

| Axis | Meaning |
|---|---|
| **Rigor** | Scientific credibility / peer-review weight / sample quality |
| **Build value** | Usefulness for the live-video app (fastest-accurate path + later training) |
| **Inference reliability** | How trustworthy the *meaning* it claims actually is. This is the honesty axis — high = the cue→state mapping is well-supported; low = popular but weakly supported |

> ⚠️ Pattern across the whole field: the **measurement** layers (pose, face, voice features)
> score high on rigor + build value. The **mind-reading** layers (cue → emotion → intent)
> score *low* on inference reliability. Everything below reflects that split.

---

## 1. Body pattern & poses (skeleton / posture)

| Resource | What it is | Rigor | Build | Inference | Notes |
|---|---|:--:|:--:|:--:|---|
| **MediaPipe Pose / Holistic** (Google) | 33-pt body + hands + 468-pt face, real-time in-browser | 5 | **5** | n/a | Your day-1 extraction engine. Measurement only. |
| **OpenPose** (CMU) | Multi-person body/hand/face/foot keypoints | 5 | 4 | n/a | Highest multi-person accuracy; heavier than MediaPipe. |
| **MoveNet / BlazePose** | Lightweight real-time pose | 4 | 5 | n/a | MoveNet rated top performer in comparative studies; great for live. |
| **COCO-Keypoints (17pt)** | The standard pose benchmark dataset/labels | 5 | 4 | n/a | Train/evaluate pose models against this. |
| **Human3.6M / H3WB** | 3D whole-body mocap benchmark | 5 | 3 | n/a | For 3D pose later; large, academic-license. |
| **Body Action & Posture (BAP) coding** | Validated scheme mapping posture→affect | 4 | 3 | 3 | One of the few *validated* body-affect schemes. Still context-dependent. |

**Verdict:** Pose extraction is the strongest, most shippable layer. Posture→emotion mapping is real but modest — treat as probabilistic.

---

## 2. Facial reactions

| Resource | What it is | Rigor | Build | Inference | Notes |
|---|---|:--:|:--:|:--:|---|
| **FACS** (Ekman & Friesen) | Action-Unit coding of facial muscles — the field's backbone | 5 | 4 | 3 | AUs are objective; AU→emotion mapping is **contested** (see Barrett 2019). |
| **OpenFace 2.0** | Open-source AU + gaze + head-pose extractor | 5 | **5** | n/a | Best free tool to get AUs/gaze live. Pairs with MediaPipe. |
| **AffectNet** (1M+ in-the-wild faces) | Largest facial expression + valence/arousal DB | 4 | 5 | 2 | Huge & useful, but labels assume universal expressions — weak premise. |
| **AffectNet+** (2024) | Adds soft-labels, age/gender/pose metadata | 4 | 5 | 3 | Soft-labels are an honesty improvement over hard emotion classes. |
| **Barrett et al. 2019**, *"Emotional Expressions Reconsidered"* | Landmark review: facial config ≠ reliable emotion | 5 | 2 | **5** | **Read this first.** Demolishes "smile = happy" universality. Anchors honest design. |

**Verdict:** Extract AUs (objective). Be very cautious mapping them to emotions — the science says expressions are variable across people/cultures/context.

---

## 3. Tone / speech (prosody + content)

| Resource | What it is | Rigor | Build | Inference | Notes |
|---|---|:--:|:--:|:--:|---|
| **IEMOCAP** (USC) | 12h dyadic audio-visual emotional speech, mocap | 5 | 5 | 3 | The standard SER benchmark. SOTA ~69–72% on 4 classes — note the ceiling. |
| **CMU-MOSEI** | 23.5k YouTube clips, aligned text+audio+video, emotion+sentiment | 5 | **5** | 3 | Best **multimodal** benchmark — matches your multi-signal goal. |
| **openSMILE / eGeMAPS** | Standard prosodic/acoustic feature extractor | 5 | 5 | n/a | Pitch, energy, jitter, shimmer — your tone feature layer. |
| **Whisper** (OpenAI) | ASR for the speech-content channel | 5 | 5 | n/a | Gives you words; content often beats prosody for meaning. |
| **SER survey (databases & algorithms)** | Field overview | 4 | 3 | 3 | Good orientation to datasets + methods. |

**Verdict:** Prosody is one of the *more* consistent emotion channels, but accuracy still tops out near ~70% on coarse categories. Combine with content (Whisper).

---

## 4. Gesture & subtle movement

| Resource | What it is | Rigor | Build | Inference | Notes |
|---|---|:--:|:--:|:--:|---|
| **Jester** (20BN) | 27-class dynamic hand-gesture video DB (largest) | 5 | 5 | n/a | Best for training dynamic gesture recognition. |
| **IPN Hand** | 4k+ samples, 800k frames, continuous real-time gestures | 5 | 5 | n/a | Built for *continuous* live recognition — matches your app. |
| **NVGesture / EgoGesture** | Depth/RGB dynamic gesture benchmarks | 4 | 4 | n/a | Good secondary benchmarks. |
| **Movement-dynamics-of-deception studies** | Kinematics of fidget/micro-movement | 3 | 2 | 2 | Interesting but small-sample; don't productize claims. |

**Verdict:** Gesture *recognition* (what the hand is doing) is solid. Gesture *meaning* ("fidget = anxiety/lying") is weak — keep descriptive.

---

## 5. External conditions / context

| Resource | What it is | Rigor | Build | Inference | Notes |
|---|---|:--:|:--:|:--:|---|
| **EMOTIC** | 23.5k images, 26 emotions, person + scene context | 5 | 5 | 3 | The key dataset proving **context** adds +5–7% mAP. Validates your "external conditions" axis. |
| **EmotiCon** (paper) | Context-aware multimodal model (Frege's principle) | 4 | 4 | 3 | Method blueprint for fusing face+body+scene. |
| **Frontiers 2024 — contextual emotion detection** | Recent deep-learning context review | 4 | 3 | 3 | Up-to-date methods survey. |

**Verdict:** Context measurably improves recognition — this is a genuine edge for a market product, and one of the better-supported ideas here.

---

## 6. Mind-reading / intent & deception (the honesty section)

| Resource | What it is | Rigor | Build | Inference | Notes |
|---|---|:--:|:--:|:--:|---|
| **DePaulo et al. 2003 meta-analysis** | Cues to deception across 100s of studies | 5 | 2 | **5** | Nonverbal lie cues are weak; observers ≈ **55%** (barely > chance). |
| **Microexpression training trials** | Tested Ekman-style training | 4 | 2 | **5** | Training accuracy ≈ chance / no better than placebo. |
| **Nonverbal-cluster studies** | Lies as *constellations* not single cues | 3 | 3 | 3 | ~68% when combining many cues — best-case, still far from "expert/court-grade". |
| **Checkpoint-screening intent study (PMC)** | Clusters predicting malicious intent | 3 | 2 | 2 | Promising direction, narrow setting. |

**Verdict:** There is **no scientific basis** for reliable single-cue lie/intent detection. The defensible product claim is *probabilistic decision-support with confidence + caveats*, never a verdict. Also legally constrained (EU AI Act restricts emotion/intent inference; biometric-privacy law applies).

---

## Top picks if you only grab a few

1. **MediaPipe Holistic + OpenFace 2.0 + openSMILE** — the live extraction stack (build now).
2. **CMU-MOSEI** — multimodal benchmark that mirrors your face+voice+body goal.
3. **EMOTIC** — proves the context/external-conditions angle.
4. **Barrett 2019** + **DePaulo 2003** — the two papers that keep the product honest (and legally safer).

## Sources
- Pose: [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco), [OpenPose vs MediaPipe](https://saiwa.ai/blog/openpose-vs-mediapipe/), [Skeleton pose comparative analysis (MDPI)](https://www.mdpi.com/1999-5903/14/12/380), [H3WB](https://arxiv.org/pdf/2211.15692)
- Facial: [AffectNet](https://arxiv.org/abs/1708.03985), [AffectNet+](https://arxiv.org/abs/2410.22506), [FER survey](https://www.researchgate.net/publication/371797577)
- Speech: [IEMOCAP](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf), [Multimodal SER on IEMOCAP](https://arxiv.org/pdf/1804.05788), [SER databases survey](http://www.warse.org/IJATCSE/static/pdf/file/ijatcse23952020.pdf)
- Gesture: [IPN Hand](https://arxiv.org/pdf/2005.02134), [Benchmark datasets eval (MDPI)](https://www.mdpi.com/2076-3417/15/11/6045)
- Multimodal: [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), [MOSEI transformer benchmark](https://arxiv.org/abs/2505.06110)
- Context: [EMOTIC](https://arxiv.org/pdf/2003.13401), [EmotiCon](https://ar5iv.labs.arxiv.org/html/2003.06692), [Contextual emotion detection (Frontiers)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1386753/full)
- Limits/deception: [Body-language myths (Earth.com summary)](https://www.earth.com/news/scientists-set-the-record-straight-by-clarifying-common-myths-about-body-language/), [Checkpoint intent clusters (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9090363/), [Movement dynamics of deception (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3608909/)
