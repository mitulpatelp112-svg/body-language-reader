"""
LLM reasoning backend — richer coach insight (item 5: MLLM reasoning for explanations).

The browser app ships an on-device template reasoner (generateInsight) that needs no API key.
This backend is the optional upgrade: it takes the structured readout (constructs, signals, dims)
and asks Claude to produce a contextual, caveated coaching explanation — moving from "what" to
"why + what to do", which the research flags as the next-gen UX (RESEARCH_REPORT.md §3, §8).

Run:
    pip install -r requirements-explain.txt
    export ANTHROPIC_API_KEY=sk-...
    uvicorn explain:app --reload --port 8003

Endpoint:
    POST /explain  body: { "reads": [{"state","p","contributors"}...], "dims": {"v","a","d"}, "hr": .. }
    -> { "ok": true, "insight": "..." }

Privacy: only the abstract readout (labels + numbers) is sent — never video/audio. Still, this
leaves the device, so it's opt-in. Self-coaching framing only; never judges third parties.
"""
import os, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Coach reasoning backend (Claude)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SYSTEM = (
    "You are a supportive communication coach. You receive an ABSTRACT readout of a person's own "
    "body-language signals (they are coaching themselves). Give ONE short, warm, actionable tip "
    "(<=2 sentences). Rules: this is probabilistic decision-support, never a verdict; never claim "
    "to detect lying, intent, or hidden truth; acknowledge uncertainty implicitly; focus on what "
    "THEY can adjust. No diagnosis."
)

class Readout(BaseModel):
    reads: list = []
    dims: dict = {}
    hr: float | None = None

@app.get("/health")
def health():
    return {"ok": True, "has_key": bool(os.getenv("ANTHROPIC_API_KEY"))}

@app.post("/explain")
def explain(r: Readout):
    try:
        from anthropic import Anthropic
        client = Anthropic()  # reads ANTHROPIC_API_KEY
        payload = json.dumps({"reads": r.reads, "dims": r.dims, "hr": r.hr}, ensure_ascii=False)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fast + cheap for live coaching
            max_tokens=120,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Readout: {payload}\nGive the coaching tip."}],
        )
        return {"ok": True, "insight": msg.content[0].text.strip()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
