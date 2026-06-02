# Graph Report - .  (2026-06-02)

## Corpus Check
- Corpus is ~35,876 words - fits in a single context window. You may not need a graph.

## Summary
- 282 nodes · 414 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 48 edges (avg confidence: 0.82)
- Token cost: 260,000 input · 30,000 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Live App Engine|Live App Engine]]
- [[_COMMUNITY_Architecture & System Docs|Architecture & System Docs]]
- [[_COMMUNITY_Ekman Nonverbal Taxonomy|Ekman Nonverbal Taxonomy]]
- [[_COMMUNITY_Emotion Classifier & Training|Emotion Classifier & Training]]
- [[_COMMUNITY_Signals Knowledge Base|Signals Knowledge Base]]
- [[_COMMUNITY_Rated Resource Catalog|Rated Resource Catalog]]
- [[_COMMUNITY_Facial & Speech Emotion Datasets|Facial & Speech Emotion Datasets]]
- [[_COMMUNITY_Gesture, Context & Pose Datasets|Gesture, Context & Pose Datasets]]
- [[_COMMUNITY_Nonverbal Communication Theory|Nonverbal Communication Theory]]
- [[_COMMUNITY_FACS Backend Service|FACS Backend Service]]
- [[_COMMUNITY_Unprocessed OCR Stub|Unprocessed OCR Stub]]

## God Nodes (most connected - your core abstractions)
1. `Rated Resource Catalog` - 25 edges
2. `Ekman & Friesen (1969): The Repertoire of Nonverbal Behavior (OCR)` - 13 edges
3. `interpret()` - 11 edges
4. `extractFeatures()` - 10 edges
5. `loop()` - 9 edges
6. `activations()` - 9 edges
7. `scoreEmotions()` - 8 edges
8. `runEval()` - 8 edges
9. `clip()` - 8 edges
10. `Interpretation Layer` - 8 edges

## Surprising Connections (you probably didn't know these)
- `analyze()` --semantically_similar_to--> `scoreEmotions()`  [INFERRED] [semantically similar]
  backend/server.py → app/emotion-core.js
- `Regression-calibrated AU Intensities` --semantically_similar_to--> `MediaPipe Face (478 mesh + 52 blendshapes)`  [INFERRED] [semantically similar]
  backend/README.md → README.md
- `Live Demo (index.html)` --references--> `Valence/Arousal/Dominance soft-labels`  [INFERRED]
  app/index.html → README.md
- `Accuracy Eval (eval.html)` --implements--> `Eval Harness (eval.html)`  [INFERRED]
  app/eval.html → README.md
- `FACS Backend Toggle` --references--> `py-feat FACS Backend (server.py)`  [INFERRED]
  app/index.html → README.md

## Hyperedges (group relationships)
- **Live Extraction Stack** — resource_catalog_mediapipe_holistic, resource_catalog_openface, resource_catalog_opensmile, resource_catalog_whisper [EXTRACTED 0.85]
- **Measurement to Interpretation Pipeline** — readme_measurement_layer, readme_signals_json, readme_fusion_policy, readme_vad_softlabels [EXTRACTED 0.85]
- **Accuracy Improvement Loop** — readme_eval_harness, readme_train_py, readme_model_json, readme_fusion_policy [EXTRACTED 0.85]
- **Five Categories of Nonverbal Behavior** — notes_ekman_friesen_ocr_emblem, notes_ekman_friesen_ocr_illustrator, notes_ekman_friesen_ocr_affect_display, notes_ekman_friesen_ocr_regulator, notes_ekman_friesen_ocr_adaptor [EXTRACTED 1.00]
- **Three Fundamental Aspects: Origin, Usage, Coding** — notes_ekman_friesen_ocr_origin, notes_ekman_friesen_ocr_usage, notes_ekman_friesen_ocr_coding [EXTRACTED 1.00]
- **Informative / Communicative / Interactive Meaning Types** — notes_ekman_friesen_ocr_informative, notes_ekman_friesen_ocr_communicative, notes_ekman_friesen_ocr_interactive [EXTRACTED 1.00]
- **Multimodal Emotion Corpora and Fusion** — papers_iemocap_original_2008_audiovisual_corpus, papers_iemocap_multimodal_2018_speech_text_mocap, papers_iemocap_multimodal_2018_multimodal_fusion [INFERRED 0.85]
- **AffectNet Emotion Model Lineage** — papers_affectnet_2017_affectnet, papers_affectnet_2017_valence_arousal, papers_affectnet_2017_categorical_model, papers_affectnet_plus_2024_soft_labels [INFERRED 0.85]
- **Multimodal Fusion Strategies** — papers_ipn_hand_2020_multi_stream_fusion, papers_cmu_mosei_transformer_2025_early_fusion, papers_emotic_context_2020_scene_context [INFERRED 0.75]
- **Body and Pose Signal Understanding** — papers_ipn_hand_2020_continuous_hand_gesture_recognition, papers_emotic_context_2020_body_pose_emotion, papers_h3wb_wholebody_2022_3d_whole_body_pose [INFERRED 0.75]
- **Communication Triad: Code, Encoder, Decoder** — resources_1972_29107_001_code, resources_1972_29107_001_encoder, resources_1972_29107_001_decoder [EXTRACTED 0.95]
- **Gesture-Cluster-to-Attitude Reading Model** — resources_howtoreadapersonlikeabook_gesture_clusters, resources_howtoreadapersonlikeabook_openness_defensiveness, resources_howtoreadapersonlikeabook_attitude_inference [INFERRED 0.75]
- **Closed loop: eval dataset -> train.py -> model.json -> app fusion** — concept_blendshape_dataset, backend_train_main, concept_model_json, app_model_infer_predict, app_app_interpret [INFERRED 0.85]
- **Measurement -> features -> activations -> fused soft-labels** — concept_measurement_layer, app_app_extractfeatures, app_app_activations, app_app_interpret, knowledge_base_signals_json [INFERRED 0.85]
- **Three independent emotion estimates fused in interpret (heuristic, trained model, FACS backend)** — app_app_activations, app_model_infer_predict, backend_server_analyze, app_app_interpret [INFERRED 0.85]

## Communities (11 total, 1 thin omitted)

### Community 0 - "Live App Engine"
Cohesion: 0.06
Nodes (56): $(), activations(), analyzeAudio(), applyContext(), autocorrelate(), blinkTimes, blinkUpdate(), bsMap() (+48 more)

### Community 1 - "Architecture & System Docs"
Cohesion: 0.05
Nodes (50): Live Demo README, EMA Smoothing / 4x-per-sec repaint, Audio / Prosody Signals (prosody_*), Quality Gating (frame damping), FACS Backend (py-feat), Regression-calibrated AU Intensities, server:app (FastAPI /analyze /health), fastapi (dependency) (+42 more)

### Community 2 - "Ekman Nonverbal Taxonomy"
Cohesion: 0.08
Nodes (33): Adaptors, Affect Displays, Arbitrary (Extrinsic) Code, Awareness (Internal Feedback), Batons (Illustrator subtype), Birdwhistell (kinesics framework), Efron (1941), Mahl (1968) (+25 more)

### Community 3 - "Emotion Classifier & Training"
Cohesion: 0.13
Nodes (25): CANONICAL, EMOTIONS, gm(), scoreEmotions(), topPrediction(), $(), computeResults(), dataset (+17 more)

### Community 4 - "Signals Knowledge Base"
Cohesion: 0.08
Nodes (23): description, adaptor, affect_display, emblem, illustrator, regulator, folklore, moderate (+15 more)

### Community 5 - "Rated Resource Catalog"
Cohesion: 0.12
Nodes (23): Datasets Access Instructions, Signal Entry Schema, AffectNet, AffectNet+ (2024), Body Action & Posture (BAP) coding, Rated Resource Catalog, CMU-MOSEI, COCO-Keypoints (17pt) (+15 more)

### Community 6 - "Facial & Speech Emotion Datasets"
Cohesion: 0.13
Nodes (20): AffectNet Database, Baseline Deep Neural Network FER Classifiers, Categorical Emotion Model (Ekman Basic Emotions), Facial Action Coding System (FACS) Model, In-the-Wild Facial Expression Recognition, Valence and Arousal (Dimensional Model), AffectNet+ Database, Data Complexity Subsets (+12 more)

### Community 7 - "Gesture, Context & Pose Datasets"
Cohesion: 0.13
Nodes (17): BERT-based Modality Encoders, CMU-MOSEI Dataset, Early Fusion, Multimodal Sentiment Analysis, Body Pose for Emotion, EMOTIC Dataset, Discrete Emotion Categories, Scene Context (+9 more)

### Community 8 - "Nonverbal Communication Theory"
Cohesion: 0.21
Nodes (13): Not All Nonverbal Behavior Is Communication, Code (Socially Shared Signal System), Decoder, Critique of Decoding-Only Perspective in the Literature, Encoder, Hand and Arm Movements as Communicative Gestures, Wiener et al. (1972) Nonverbal Behavior and Nonverbal Communication, Sign vs. Communication Distinction (+5 more)

### Community 9 - "FACS Backend Service"
Cohesion: 0.23
Nodes (10): normEmotions(), pollBackend(), analyze(), _decode(), detector(), Frame, Calibrated FACS backend for the Body-Language Reader.  Uses py-feat (https://py-, Decode a base64 data URL to a temp file path py-feat can read. (+2 more)

## Ambiguous Edges - Review These
- `Sign vs. Communication Distinction` → `Inferring Attitudes from Gesture Clusters`  [AMBIGUOUS]
  resources/HowToReadAPersonLikeABook .pdf · relation: conceptually_related_to

## Knowledge Gaps
- **92 isolated node(s):** `video`, `trials`, `dataset`, `video`, `canvas` (+87 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Sign vs. Communication Distinction` and `Inferring Attitudes from Gesture Clusters`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `scoreEmotions()` connect `Emotion Classifier & Training` to `Live App Engine`, `FACS Backend Service`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `activations()` connect `Live App Engine` to `Emotion Classifier & Training`, `Signals Knowledge Base`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `interpret()` (e.g. with `Interpretation / Fusion Layer (signals -> states, noisy-OR)` and `frameQuality()`) actually correct?**
  _`interpret()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `video`, `trials`, `dataset` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Live App Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.061016949152542375 - nodes in this community are weakly interconnected._
- **Should `Architecture & System Docs` be split into smaller, more focused modules?**
  _Cohesion score 0.053877551020408164 - nodes in this community are weakly interconnected._