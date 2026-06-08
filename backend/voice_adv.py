"""
Real wav2vec2 dimensional speech-emotion backend (Arousal / Dominance / Valence).

The browser app ships an *interim* acoustic A/D/V estimator (jitter/shimmer/centroid heuristics).
This backend upgrades it to the SOTA approach: audEERING's wav2vec2 model that regresses A/D/V —
the architecture that won the 2024 MSP-Podcast challenge (valence CCC ~0.68), far beyond hand
features. See RESEARCH_REPORT.md §3.

Run:
    pip install -r requirements-voice.txt
    uvicorn voice_adv:app --reload --port 8002

Endpoint:
    POST /voice   body: { "audio_b64": "<base64 16kHz mono PCM wav>" }
    -> { "ok": true, "arousal": .., "dominance": .., "valence": .. }   # each 0..1

The app would POST ~3s audio windows here and fuse the returned A/D/V into estimateDims()
(replacing the interim prosody.v/a/d), exactly like the FACS backend toggle does for faces.
"""
import base64, io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Voice A/D/V backend (wav2vec2)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
_model = _proc = None

def _load():
    global _model, _proc
    if _model is None:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
        import torch.nn as nn

        class RegressionHead(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
                self.dropout = nn.Dropout(cfg.final_dropout)
                self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)
            def forward(self, x):
                x = self.dropout(x); x = torch.tanh(self.dense(x)); x = self.dropout(x)
                return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.wav2vec2 = Wav2Vec2Model(cfg); self.classifier = RegressionHead(cfg)
                self.init_weights()
            def forward(self, x):
                h = self.wav2vec2(x).last_hidden_state.mean(dim=1)
                return self.classifier(h)

        _proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _model = EmotionModel.from_pretrained(MODEL_ID).eval()
    return _model, _proc

class Clip(BaseModel):
    audio_b64: str
    sample_rate: int = 16000

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "loaded": _model is not None}

@app.post("/voice")
def voice(clip: Clip):
    try:
        import numpy as np, soundfile as sf, torch
        raw = base64.b64decode(clip.audio_b64)
        wav, sr = sf.read(io.BytesIO(raw))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        model, proc = _load()
        inp = proc(wav, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            out = model(inp)[0].numpy()           # [arousal, dominance, valence] in 0..1
        return {"ok": True, "arousal": float(out[0]), "dominance": float(out[1]), "valence": float(out[2])}
    except Exception as e:
        return {"ok": False, "error": str(e)}
