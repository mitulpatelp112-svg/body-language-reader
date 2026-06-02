# Datasets — Access Instructions (license-gated, can't auto-download)

These are the training/evaluation datasets. Each requires **you** to accept an academic-use
license or register — they cannot be auto-downloaded (legal + multi-GB size). Once you have
access, drop them in subfolders here.

| Dataset | Size | How to get it | License note |
|---|---|---|---|
| **AffectNet / AffectNet+** | ~120 GB | Request form to Mohammad Mahoor's lab (Univ. of Denver) — http://mohammadmahoor.com/affectnet/ | Academic/research only |
| **EMOTIC** | ~2 GB | GitHub + form: https://github.com/rkosti/emotic | Research only; CVPR dataset |
| **IEMOCAP** | ~12 GB | Request form to USC SAIL: https://sail.usc.edu/iemocap/ | Academic; signed release required |
| **CMU-MOSEI** | ~from SDK | CMU Multimodal SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK | Research use |
| **IPN Hand** | ~ tens GB | https://gibranbenitez.github.io/IPN_Hand/ | Research; agreement form |
| **Jester** | large | Qualcomm/20BN (registration) | Research |
| **Human3.6M / H3WB** | large | http://vision.imar.ro/human3.6m/ (academic acct) | Strict academic license |

## For a commercial product — important
Most of the above are **research-only licenses**. For a *market* product you will eventually need:
- Commercially-licensed datasets, OR
- Your own collected + consented data (the route you mentioned: train your own model later).
- Note biometric-data consent law (GDPR Art.9, Illinois BIPA, EU AI Act emotion-recognition limits).

## What we use immediately (no gating)
The **tools** (MediaPipe, OpenFace, openSMILE, Whisper) are open-source and run without any of the
above datasets — they're how we build the live demo first. Datasets come in only when we train.
