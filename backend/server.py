"""
Calibrated FACS backend for the Body-Language Reader.

Uses py-feat (https://py-feat.org) to extract *regression-calibrated* facial action-unit
intensities + emotion probabilities — more rigorous than MediaPipe's blendshapes, which are
geometric approximations. The browser app can POST frames here to upgrade the facial layer.

Run:
    pip install -r requirements.txt
    uvicorn server:app --reload --port 8001

Endpoint:
    POST /analyze   body: { "image": "data:image/jpeg;base64,..." }
    -> { "ok": true, "aus": {"AU01": .., ...}, "emotions": {"happiness": .., ...}, "valence":..,"arousal":.. }
"""
import base64, io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

app = FastAPI(title="Body-Language Reader — FACS backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Lazy-load the detector once (model download happens on first import).
_detector = None
def detector():
    global _detector
    if _detector is None:
        from feat import Detector
        # these are the default, well-validated py-feat models
        _detector = Detector(
            face_model="retinaface", landmark_model="mobilefacenet",
            au_model="xgb", emotion_model="resmasknet",
        )
    return _detector

class Frame(BaseModel):
    image: str  # data URL or raw base64 JPEG/PNG

def _decode(data_url: str) -> str:
    """Decode a base64 data URL to a temp file path py-feat can read."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")
    path = "/tmp/blr_frame.jpg"
    img.save(path, "JPEG", quality=90)
    return path

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _detector is not None}

@app.post("/analyze")
def analyze(frame: Frame):
    try:
        path = _decode(frame.image)
        pred = detector().detect_image(path)
        if pred is None or len(pred) == 0:
            return {"ok": False, "reason": "no face"}
        aus = {k: float(v) for k, v in pred.aus.iloc[0].to_dict().items()}
        emotions = {k: float(v) for k, v in pred.emotions.iloc[0].to_dict().items()}
        # dimensional estimate from emotions (simple mapping; refine as needed)
        val = emotions.get("happiness", 0) - (emotions.get("sadness", 0)
              + emotions.get("anger", 0) + emotions.get("disgust", 0) + emotions.get("fear", 0)) / 4
        aro = emotions.get("anger", 0) + emotions.get("fear", 0) + emotions.get("surprise", 0)
        return {"ok": True, "aus": aus, "emotions": emotions,
                "valence": float(np.clip(val, -1, 1)), "arousal": float(np.clip(aro, 0, 1))}
    except Exception as e:
        return {"ok": False, "error": str(e)}
