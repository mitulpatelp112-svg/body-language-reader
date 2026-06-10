"""
Calibrated FACS backend for the Body-Language Reader.

Uses py-feat (https://py-feat.org) to extract *regression-calibrated* facial action-unit
intensities + emotion probabilities — more rigorous than MediaPipe's blendshapes, which are
geometric approximations. The browser app can POST frames here to upgrade the facial layer.

Run:
    pip install -r requirements.txt
    uvicorn server:app --reload --port 8001

Endpoints:
    POST /analyze        body: { "image": "data:image/jpeg;base64,..." }
        -> {ok, aus, emotions, valence, arousal}                            (live webcam path)
    POST /analyze_image  multipart file=<image>                              (uploaded photo)
        -> {ok, aus, emotions, valence, arousal, top, confidence, caveats}
    POST /analyze_video  multipart file=<video> [?fps=2&max_frames=120]      (uploaded clip)
        -> {ok, duration_s, fps_sampled, frames:[{t, top, confidence, emotions, valence, arousal}],
            aggregate:{emotions, valence, arousal, top, confidence}, caveats}
    POST /analyze_cnn    multipart file=<image>                              (in-house CNN, faster)
        -> {ok, model, classes, emotions, top, confidence, caveats}
"""
import base64, io, json, os, tempfile, uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import numpy as np

app = FastAPI(title="Body-Language Reader — FACS backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Honest caveats returned with every response — emotion inference is probabilistic, not factual.
# See README + RESEARCH_REPORT.md: Barrett (2019), DePaulo (2003).
CAVEATS = [
    "Probabilistic estimate, not a verdict — facial expression maps to emotion only with substantial variance (Barrett 2019).",
    "Single-frame reads are weak; corroborate across frames, voice, and context.",
    "Model trained on labelled image datasets (AffectNet/FER+) — known cultural & lighting bias.",
]

# Lazy-load the detector once (model download happens on first import).
_detector = None
def detector():
    global _detector
    if _detector is None:
        from feat import Detector
        # default, well-validated py-feat models
        _detector = Detector(
            face_model="retinaface", landmark_model="mobilefacenet",
            au_model="xgb", emotion_model="resmasknet",
        )
    return _detector

class Frame(BaseModel):
    image: str  # data URL or raw base64 JPEG/PNG


def _decode_data_url(data_url: str) -> str:
    """Decode a base64 data URL to a temp file path py-feat can read."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")
    path = os.path.join(tempfile.gettempdir(), f"blr_{uuid.uuid4().hex}.jpg")
    img.save(path, "JPEG", quality=90)
    return path


def _save_upload(upload: UploadFile, suffix: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"blr_{uuid.uuid4().hex}{suffix}")
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


def _vad_from_emotions(emotions: dict) -> tuple[float, float]:
    """Map 7-emotion probabilities to a coarse Valence/Arousal pair.
    Mapping follows Russell's circumplex placements; this is a downstream summary, not a re-prediction."""
    happy = emotions.get("happiness", 0)
    sad = emotions.get("sadness", 0)
    anger = emotions.get("anger", 0)
    disgust = emotions.get("disgust", 0)
    fear = emotions.get("fear", 0)
    surprise = emotions.get("surprise", 0)
    val = happy - (sad + anger + disgust + fear) / 4
    aro = anger + fear + surprise
    return float(np.clip(val, -1, 1)), float(np.clip(aro, 0, 1))


def _summarise(emotions: dict) -> tuple[str, float]:
    """Top label + confidence (probability of the argmax)."""
    if not emotions:
        return ("unknown", 0.0)
    top = max(emotions, key=emotions.get)
    return (top, float(emotions[top]))


def _analyse_path(path: str) -> dict:
    """Run py-feat on one image path; return a normalised dict."""
    pred = detector().detect_image(path)
    if pred is None or len(pred) == 0:
        return {"ok": False, "reason": "no face"}
    aus = {k: float(v) for k, v in pred.aus.iloc[0].to_dict().items()}
    emotions = {k: float(v) for k, v in pred.emotions.iloc[0].to_dict().items()}
    val, aro = _vad_from_emotions(emotions)
    top, conf = _summarise(emotions)
    return {
        "ok": True,
        "aus": aus, "emotions": emotions,
        "valence": val, "arousal": aro,
        "top": top, "confidence": conf,
    }


# ---------- In-house ONNX CNN (MobileNetV3, trained on AffectNet 6-class) ----------
# Lazy-load on first call; runs ~10ms/image on CPU via onnxruntime.

_HERE = Path(__file__).resolve().parent
_APP = _HERE.parent / "app"
_CNN_ONNX = _APP / "emotion_cnn.onnx"
_CNN_META = _APP / "emotion_cnn_meta.json"

_cnn_session = None
_cnn_meta = None

def cnn():
    """Return (onnxruntime InferenceSession, meta dict) or raise if model isn't built yet."""
    global _cnn_session, _cnn_meta
    if _cnn_session is None:
        if not _CNN_ONNX.exists():
            raise FileNotFoundError(
                f"emotion_cnn.onnx not found at {_CNN_ONNX}. "
                "Train it first: python3 backend/train_cnn.py"
            )
        import onnxruntime as ort
        _cnn_session = ort.InferenceSession(str(_CNN_ONNX), providers=["CPUExecutionProvider"])
        _cnn_meta = json.load(open(_CNN_META))
    return _cnn_session, _cnn_meta

def _preprocess_for_cnn(pil_img: Image.Image, size: int, mean, std) -> np.ndarray:
    """Center-crop + resize + ImageNet-normalize. Matches the eval transform from train_cnn.py."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    # resize shorter side to 256 then center-crop to `size`
    w, h = img.size
    short = 256
    scale = short / min(w, h)
    img = img.resize((int(round(w*scale)), int(round(h*scale))), Image.BILINEAR)
    w, h = img.size
    left = (w - size) // 2; top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)[None]  # NCHW
    return arr.astype(np.float32)


@app.get("/health")
def health():
    cnn_ready = _CNN_ONNX.exists()
    return {
        "ok": True,
        "model_loaded": _detector is not None,
        "cnn_built": cnn_ready,
        "cnn_path": str(_CNN_ONNX) if cnn_ready else None,
    }


@app.post("/analyze_cnn")
def analyze_cnn(file: UploadFile = File(...)):
    """AJNA's in-house CNN (EfficientNet-B4) trained on AffectNet 6-class subset (no neutral / no
    contempt). Faster than /analyze_image (no Action-Unit head); returns just emotion probabilities."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "expected an image upload")
    try:
        sess, meta = cnn()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    try:
        img = Image.open(io.BytesIO(file.file.read()))
        x = _preprocess_for_cnn(img, meta["input_size"], meta["mean"], meta["std"])
        logits = sess.run(["logits"], {"image": x})[0][0]
        # numerically-stable softmax
        z = logits - logits.max()
        exps = np.exp(z); probs = exps / exps.sum()
        emotions = {c: float(p) for c, p in zip(meta["classes"], probs)}
        top = max(emotions, key=emotions.get)
        # downstream V/A summary (same mapping as the FACS path)
        val, aro = _vad_from_emotions(emotions)
        return {
            "ok": True,
            "model": meta.get("architecture", "AJNA CNN (AffectNet 6-class)"),
            "classes": meta["classes"],
            "emotions": emotions,
            "top": top, "confidence": float(emotions[top]),
            "valence": val, "arousal": aro,
            "test_accuracy_reported": meta.get("test_acc"),
            "caveats": CAVEATS + [
                "Trained on 6 classes — does NOT predict 'neutral' or 'contempt'; those signals are absent in this output.",
            ],
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "caveats": CAVEATS}


@app.post("/analyze_cnn_live")
def analyze_cnn_live(frame: Frame):
    """Live-webcam variant of /analyze_cnn — accepts a JSON data-URL (what the browser app sends
    every ~500ms) instead of a multipart upload, and returns just the emotion probabilities so the
    app can fuse them into its live read. Lighter response than /analyze_cnn (no caveats/V-A blob)."""
    try:
        sess, meta = cnn()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    try:
        data = frame.image
        if "," in data:
            data = data.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        x = _preprocess_for_cnn(img, meta["input_size"], meta["mean"], meta["std"])
        logits = sess.run(["logits"], {"image": x})[0][0]
        z = logits - logits.max()
        exps = np.exp(z); probs = exps / exps.sum()
        emotions = {c: float(p) for c, p in zip(meta["classes"], probs)}
        top = max(emotions, key=emotions.get)
        return {"ok": True, "emotions": emotions, "top": top, "confidence": float(emotions[top])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/analyze")
def analyze(frame: Frame):
    """Legacy live-frame endpoint (py-feat FACS). The webcam app now uses /analyze_cnn_live."""
    try:
        path = _decode_data_url(frame.image)
        out = _analyse_path(path)
        try: os.remove(path)
        except OSError: pass
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/analyze_image")
def analyze_image(file: UploadFile = File(...)):
    """Uploaded image -> calibrated emotion + AUs + VAD."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "expected an image upload")
    try:
        path = _save_upload(file, os.path.splitext(file.filename or "")[1] or ".jpg")
        out = _analyse_path(path)
        try: os.remove(path)
        except OSError: pass
        out["caveats"] = CAVEATS
        return out
    except Exception as e:
        return {"ok": False, "error": str(e), "caveats": CAVEATS}


@app.post("/analyze_video")
def analyze_video(
    file: UploadFile = File(...),
    fps: float = Query(2.0, gt=0.1, le=10.0, description="frames sampled per second"),
    max_frames: int = Query(120, gt=1, le=600, description="hard cap on frames to analyse"),
):
    """Uploaded video -> per-frame timeline + aggregate. Samples at `fps` up to `max_frames`."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(415, "expected a video upload")
    import cv2  # OpenCV — heavy import, lazy-loaded
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    vpath = _save_upload(file, suffix)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        try: os.remove(vpath)
        except OSError: pass
        return {"ok": False, "error": "could not open video"}

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (n_total / src_fps) if src_fps > 0 else 0.0
    stride = max(1, int(round(src_fps / fps)))

    frames_out = []
    sums = {}  # accumulator for emotion average over frames with detections
    n_detected = 0
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if idx % stride == 0:
            t = idx / src_fps if src_fps > 0 else len(frames_out) / fps
            tmp_path = os.path.join(tempfile.gettempdir(), f"blr_v_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(tmp_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            res = _analyse_path(tmp_path)
            try: os.remove(tmp_path)
            except OSError: pass
            if res.get("ok"):
                frames_out.append({
                    "t": round(t, 3),
                    "top": res["top"], "confidence": res["confidence"],
                    "emotions": res["emotions"],
                    "valence": res["valence"], "arousal": res["arousal"],
                })
                n_detected += 1
                for k, v in res["emotions"].items():
                    sums[k] = sums.get(k, 0.0) + v
            else:
                frames_out.append({"t": round(t, 3), "ok": False, "reason": res.get("reason", "no face")})
            if len(frames_out) >= max_frames: break
        idx += 1
    cap.release()
    try: os.remove(vpath)
    except OSError: pass

    aggregate = None
    if n_detected > 0:
        avg_em = {k: v / n_detected for k, v in sums.items()}
        val, aro = _vad_from_emotions(avg_em)
        top, conf = _summarise(avg_em)
        aggregate = {
            "emotions": avg_em, "valence": val, "arousal": aro,
            "top": top, "confidence": conf, "n_frames_with_face": n_detected,
        }
    return {
        "ok": True, "duration_s": round(duration, 3),
        "fps_sampled": fps, "frames": frames_out,
        "aggregate": aggregate, "caveats": CAVEATS,
    }
