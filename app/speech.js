// Speech-content analysis for Presence (filler words, word count, pace).
// NOTE: the browser SpeechRecognition API may process audio via the browser maker's cloud
// service. This is the ONE feature that can leave the device, so it is opt-in and disclosed in
// the UI. Everything else in Presence stays on-device.
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

const FILLERS = ["um","uh","er","ah","like","so","actually","basically","literally","right","okay","hmm"];
const FILLER_PHRASES = ["you know","i mean","kind of","sort of"];

export const speech = {
  supported: !!SR,
  active: false,
  words: 0,
  fillerTotal: 0,
  fillerByType: {},
  transcript: "",
  _rec: null,
  _finalText: "",

  start() {
    if (!SR || this.active) return false;
    this.reset();
    const r = new SR();
    r.continuous = true; r.interimResults = true; r.lang = "en-US";
    r.onresult = (e) => {
      let finalChunk = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) finalChunk += e.results[i][0].transcript + " ";
      }
      if (finalChunk) this._ingest(finalChunk);
    };
    r.onerror = () => {};
    r.onend = () => { if (this.active) { try { r.start(); } catch (e) {} } };  // auto-restart while recording
    try { r.start(); } catch (e) { return false; }
    this._rec = r; this.active = true;
    return true;
  },

  stop() {
    this.active = false;
    if (this._rec) { try { this._rec.stop(); } catch (e) {} this._rec = null; }
  },

  reset() {
    this.words = 0; this.fillerTotal = 0; this.fillerByType = {}; this.transcript = ""; this._finalText = "";
  },

  _ingest(text) {
    const lower = " " + text.toLowerCase().replace(/[^\w\s']/g, " ").replace(/\s+/g, " ") + " ";
    const tokens = lower.trim().split(" ").filter(Boolean);
    this.words += tokens.length;
    for (const f of FILLERS) {
      const n = tokens.filter((t) => t === f).length;
      if (n) { this.fillerTotal += n; this.fillerByType[f] = (this.fillerByType[f] || 0) + n; }
    }
    for (const p of FILLER_PHRASES) {
      const n = (lower.match(new RegExp("\\b" + p + "\\b", "g")) || []).length;
      if (n) { this.fillerTotal += n; this.fillerByType[p] = (this.fillerByType[p] || 0) + n; }
    }
    this._finalText += text + " ";
    this.transcript = this._finalText.trim().slice(-600);
  },

  // metrics snapshot; talkSeconds comes from the recorder's on-device VAD (more accurate than ASR)
  metrics(talkSeconds) {
    const mins = Math.max(talkSeconds, 1) / 60;
    return {
      supported: this.supported,
      words: this.words,
      wpm: Math.round(this.words / mins),
      fillerTotal: this.fillerTotal,
      fillerByType: { ...this.fillerByType },
      fillerPerMin: +(this.fillerTotal / mins).toFixed(1),
      transcript: this.transcript,
    };
  },
};
