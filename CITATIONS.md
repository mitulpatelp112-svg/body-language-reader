# Citations & Attributions

This project builds on published research, datasets, and open-source tools. Every source used —
in the knowledge base, the resource catalog, or bundled under `resources/` — is credited below.
All third-party materials remain the property of their respective authors/publishers (see
`NOTICE.md`).

## Foundational theory & limits of the field
- **Ekman, P., & Friesen, W. V. (1969).** The Repertoire of Nonverbal Behavior: Categories,
  Origins, Usage, and Coding. *Semiotica, 1*(1), 49–98. — the 5-category framework (emblems,
  illustrators, affect displays, regulators, adaptors) that organizes the knowledge base.
- **Ekman, P., & Friesen, W. V. (1978).** *Facial Action Coding System (FACS).* Consulting
  Psychologists Press. — the action-unit basis for the facial signals.
- **Wiener, M., Devoe, S., Rubinow, S., & Geller, J. (1972).** Nonverbal behavior and nonverbal
  communication. *Psychological Review, 79*(3), 185–214.
- **Nierenberg, G. I., & Calero, H. H. (1971).** *How to Read a Person Like a Book.* (Popular
  trade book — used only as an unverified gesture-cluster source; flagged low-rigor.)
- **Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019).** Emotional
  Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements.
  *Psychological Science in the Public Interest, 20*(1), 1–68.
- **DePaulo, B. M., Lindsay, J. J., Malone, B. E., Muhlenbruck, L., Charlton, K., & Cooper, H.
  (2003).** Cues to deception. *Psychological Bulletin, 129*(1), 74–118.

## Datasets & benchmark papers
- **Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019).** AffectNet: A Database for Facial
  Expression, Valence, and Arousal Computing in the Wild. *IEEE Transactions on Affective
  Computing.* arXiv:1708.03985.
- **AffectNet+ (2024).** A Database for Enhancing Facial Expression Recognition with Soft-Labels.
  arXiv:2410.22506.
- **Busso, C., et al. (2008).** IEMOCAP: Interactive emotional dyadic motion capture database.
  *Language Resources and Evaluation, 42*(4), 335–359.
- **Tripathi, S., et al. (2018).** Multi-Modal Emotion Recognition on IEMOCAP with Deep Learning.
  arXiv:1804.05788.
- **Benitez-Garcia, G., Olivares-Mercado, J., Sanchez-Perez, G., & Yanai, K. (2021).** IPN Hand:
  A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition. *ICPR.*
  arXiv:2005.02134.
- **Zadeh, A. B., Liang, P. P., Poria, S., Cambria, E., & Morency, L.-P. (2018).** Multimodal
  Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. *ACL.*
  - plus the transformer benchmark study, arXiv:2505.06110.
- **Kosti, R., Alvarez, J. M., Recasens, A., & Lapedriza, A. (2019).** Context Based Emotion
  Recognition using EMOTIC Dataset. *IEEE TPAMI.* arXiv:2003.13401.
- **Mittal, T., et al. (2020).** EmotiCon: Context-Aware Multimodal Emotion Recognition using
  Frege's Principle. *CVPR.* arXiv:2003.06692.
- **Zhu, Y., et al. (2023).** H3WB: Human3.6M 3D WholeBody Dataset and Benchmark. *ICCV.*
  arXiv:2211.15692.
- **Ionescu, C., Papava, D., Olaru, V., & Sminchisescu, C. (2014).** Human3.6M. *IEEE TPAMI.*
- **Materzynska, J., et al. (2019).** The Jester Dataset: A Large-Scale Video Dataset of Human
  Gestures. *ICCV Workshops.*

## Construct-level evidence (the corroboration / confidence layer)
These ground the `constructs` in `knowledge-base/signals.json` — the evidence that cue *constellations*
(not single cues) support confident reads:
- **Ambady, N., & Rosenthal, R. (1992).** Thin slices of expressive behavior as predictors of
  interpersonal consequences: A meta-analysis. *Psychological Bulletin, 111*(2), 256–274. (Overall
  r≈.39 predicting interpersonal outcomes from <5 min of behavior; channel didn't matter.)
- **Grahe, J. E., & Bernieri, F. J. (1999).** The importance of nonverbal cues in judging rapport.
  *Journal of Nonverbal Behavior, 23*(4), 253–269.
- **Müller, P., Huang, M. X., & Bulling, A. (2018).** Detecting Low Rapport During Natural
  Interactions in Small Groups from Non-Verbal Behaviour. *IUI.* arXiv:1801.06055.
- **Witkower, Z., & Tracy, J. L. (2019).** A Facial-Action Imposter: How Head Tilt Influences
  Perceptions of Dominance From a Neutral Face. *Psychological Science.* (Dominance as a multi-cue
  display.)
- **Elkjær, E., Mikkelsen, M. B., Michalak, J., Mennin, D. S., & O'Toole, M. S. (2020).** Expansive
  and Contractive Postures and Movement: A Systematic Review and Meta-Analysis. *Perspectives on
  Psychological Science.*
- **Mehrabian, A. (1969).** Some referents and measures of nonverbal behavior (immediacy /
  engagement cues). *Behavior Research Methods & Instrumentation.*
- **Mohiyeddini / Troisi — Signal value of stress behaviour (2022).** *Evolution and Human
  Behavior.* (Self-directed displacement behaviours reliably read as stress.)
- **Bimodal Learning Engagement Recognition from Videos in the Classroom (2022).** *Sensors / PMC.*
  (Multimodal engagement detection accuracy.)

## Tools & open-source libraries
- **MediaPipe** — Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception
  Pipelines. Google. (FaceLandmarker, PoseLandmarker, HandLandmarker; BlazePose — Bazarevsky et
  al. 2020.)
- **OpenPose** — Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2019). *IEEE TPAMI.*
- **OpenFace 2.0** — Baltrušaitis, T., Zadeh, A., Lim, Y. C., & Morency, L.-P. (2018). *IEEE FG.*
- **openSMILE / eGeMAPS** — Eyben, F., Wöllmer, M., & Schuller, B. (2010), *ACM MM*; Eyben et al.
  (2016), eGeMAPS, *IEEE Transactions on Affective Computing.*
- **Whisper** — Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak
  Supervision. OpenAI.
- **Py-Feat** — Cheong, J. H., et al. (2023). Py-Feat: Python Facial Expression Analysis Toolbox.
  *Behavior Research Methods.*
- **scikit-learn** — Pedregosa, F., et al. (2011). *JMLR, 12*, 2825–2830.
- **FastAPI**, **NumPy**, **Pillow** — respective open-source projects.

## How sources are used in-repo
- `resource-catalog.md` — every source rated on rigor / build value / inference reliability.
- `knowledge-base/signals.json` — each signal carries a `sources` field linking the claim to the
  paper(s) above.
- `resources/notes/` — reading notes and OCR text distilling these works.

_If any attribution here is incomplete or imprecise, please open an issue — corrections welcome._
