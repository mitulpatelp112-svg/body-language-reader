// Body-Language Reader — live engine
// Measurement: MediaPipe FaceLandmarker (blendshapes ~= action units) + PoseLandmarker.
// Interpretation: feeds detections into ../knowledge-base/signals.json (probabilistic, caveated).
import { FaceLandmarker, PoseLandmarker, HandLandmarker, FilesetResolver, DrawingUtils }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { loadModel, hasModel, predict as modelPredict, modelInfo } from "./model-infer.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx2d = canvas.getContext("2d");
const statusEl = $("status");
let drawer = null; // DrawingUtils, created after canvas sized

let faceLM, poseLM, handLM, running = false, lastVideoTime = -1;
let KB = null;                 // knowledge base (signals.json)
let baseline = null;           // personal-baseline EMAs
let calibrating = 0;           // frames remaining in calibration
const noseHist = [];           // for nod detection
const wristHist = [];          // for fidget/motion energy
const blinkTimes = [];         // timestamps of recent blinks
let blinkOpen = true;          // blink edge-detector state
let frameQ = 1;                // current frame tracking-quality 0..1 (gates interpretation)
let lastBlendshapes = null;    // raw blendshape dict for the trained model
// noise reduction
const FEAT_EMA = {};           // feature-level smoothing (denoise at the source)
const FEAT_A = 0.35;
const activeSig = new Set();    // hysteresis: which signals are currently latched on
const SIG_ON = 0.20, SIG_OFF = 0.10;   // Schmitt-trigger thresholds (no flicker at one cutoff)
const f0Hist = [];             // recent pitch readings for median filtering
const lumaCanvas = Object.assign(document.createElement("canvas"), {width:32, height:24});
const lumaCtx = lumaCanvas.getContext("2d", { willReadFrequently: true });
// rPPG (contactless heart rate / physiological arousal from forehead skin-color changes)
const roiCanvas = Object.assign(document.createElement("canvas"), {width:30, height:18});
const roiCtx = roiCanvas.getContext("2d", { willReadFrequently: true });
const rppg = { buf: [], hr: 0, hrBase: null, arousal: 0, quality: 0 };  // buf: {t, g}
let lastHrT = 0;
// optional py-feat FACS backend (calibrated AUs/emotions) — off by default
const BACKEND_URL = "http://localhost:8001";
let backendOn = false, backendBusy = false, backendEmotions = null;
const grabCanvas = Object.assign(document.createElement("canvas"), {width:320, height:240});
const grabCtx = grabCanvas.getContext("2d");
// audio / prosody state
let audioCtx, analyser, audioBuf, audioOn = false;
let pitchBase = null, energyBase = null;
const voicedHist = [];         // recent voiced flags for pause/disfluency
const prosody = { voiced: 0, pitchRise: 0, energyRise: 0, arousal: 0, disfluency: 0,
                  v: 0, a: 0, d: 0, jitter: 0, shimmer: 0, centroid: 0 };  // dimensional A/D/V
const ampHist = [];            // recent RMS for shimmer
let freqBuf = null;            // spectral frame for centroid
// temporal smoothing + render throttle (fixes "moving too fast to read")
const SMOOTH = { dims: {v:0,a:0,d:0}, states: new Map(), signals: new Map() };
const SM_A = 0.18;             // EMA factor for displayed values
const RENDER_MS = 280;         // panel refresh interval
let lastRender = 0;

// ---------- Knowledge base ----------
async function loadKB() {
  try {
    const r = await fetch("../knowledge-base/signals.json");
    KB = await r.json();
    console.log("KB loaded:", KB.signals.length, "signals");
  } catch (e) {
    console.warn("KB fetch failed (serve from project root). Using inline fallback.", e);
    KB = FALLBACK_KB;
  }
  if (await loadModel()) {                    // trained model present -> app fuses it automatically
    const mi = modelInfo();
    statusEl.textContent = `trained model loaded (CV acc ${mi.acc ?? "?"})`;
  }
}

// Friendly camera errors
class CameraError extends Error {
  constructor(e) {
    const m = {
      NotAllowedError: "Camera permission denied. Click the camera icon in your browser's address bar → Allow, then reload. (Embedded preview panels usually block the camera — open this page in a real browser tab at http://localhost:8000/app/ instead.)",
      NotFoundError: "No camera found. Connect a webcam and reload.",
      NotReadableError: "Camera is in use by another app (Zoom, FaceTime, etc.). Close it and reload.",
      OverconstrainedError: "Camera doesn't support the requested resolution.",
      SecurityError: "Camera blocked — the page must be served over http://localhost or https://.",
    }[e && e.name] || ("Camera error: " + (e && e.message || e));
    super(m); this.name = "CameraError";
  }
}
function showStageMessage(text, bad) {
  const el = $("coach");
  if (el) { el.textContent = text; el.style.color = bad ? "var(--bad)" : "var(--txt)"; }
  statusEl.textContent = bad ? "camera error" : "running";
}

// ---------- Setup ----------
async function init() {
  statusEl.textContent = "loading models…";
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  faceLM = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU" },
    runningMode: "VIDEO", numFaces: 1,
    outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
  poseLM = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU" },
    runningMode: "VIDEO", numPoses: 1 });
  handLM = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU" },
    runningMode: "VIDEO", numHands: 2 });
  statusEl.textContent = "models ready";
}

async function startCamera() {
  // Camera is required; request it on its own so a mic denial can't block it.
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 } });
  } catch (e) {
    throw new CameraError(e);   // surfaced with a friendly, actionable message
  }
  // Mic is optional — prosody/voice just stays off if denied.
  try {
    const a = await navigator.mediaDevices.getUserMedia({ audio: true });
    a.getAudioTracks().forEach(t => stream.addTrack(t));
  } catch { console.warn("mic unavailable — voice/prosody disabled"); }
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  drawer = new DrawingUtils(ctx2d);
  initAudio(stream);
  setInterval(pollBackend, 500);   // periodic FACS backend calls when toggled on
  running = true; startCalibration();
  $("calib").disabled = false;
  statusEl.textContent = "running";
  requestAnimationFrame(loop);
}

// ---------- FACS backend (optional, calibrated AUs/emotions) ----------
async function pollBackend() {
  if (!backendOn || backendBusy || !running || video.readyState < 2) return;
  backendBusy = true;
  try {
    grabCtx.drawImage(video, 0, 0, 320, 240);
    const img = grabCanvas.toDataURL("image/jpeg", 0.7);
    const r = await fetch(BACKEND_URL + "/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: img }) });
    const j = await r.json();
    backendEmotions = (j.ok && j.emotions) ? normEmotions(j.emotions) : null;
  } catch (e) { /* transient network error; keep last value briefly */ }
  finally { backendBusy = false; }
}
function normEmotions(e) {
  const out = {}; for (const k in e) { const lk = k.toLowerCase(); if (lk !== "neutral") out[lk] = e[k]; }
  return out;
}

// ---------- Audio / prosody ----------
function initAudio(stream) {
  try {
    if (!stream.getAudioTracks().length) return;
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048; analyser.smoothingTimeConstant = 0.3;
    audioBuf = new Float32Array(analyser.fftSize);
    src.connect(analyser);
    audioOn = true;
    setInterval(analyzeAudio, 80);   // ~12 Hz prosody analysis, independent of video
  } catch (e) { console.warn("audio init failed (prosody disabled):", e); audioOn = false; }
}
function analyzeAudio() {
  if (!audioOn || calibrating > 0) return;
  analyser.getFloatTimeDomainData(audioBuf);
  // RMS energy
  let sum = 0; for (let i=0;i<audioBuf.length;i++) sum += audioBuf[i]*audioBuf[i];
  const rms = Math.sqrt(sum / audioBuf.length);
  const voiced = rms > 0.012;                                  // simple VAD
  prosody.voiced = voiced ? 1 : 0;
  voicedHist.push(voiced ? 1 : 0); if (voicedHist.length > 60) voicedHist.shift();
  if (voiced) {
    const f0raw = autocorrelate(audioBuf, audioCtx.sampleRate); // fundamental frequency
    if (f0raw > 0) {
      f0Hist.push(f0raw); if (f0Hist.length > 6) f0Hist.shift();
      const f0 = median(f0Hist);                                // median filter kills octave jumps
      pitchBase = pitchBase == null ? f0 : 0.98*pitchBase + 0.02*f0;
      prosody.pitchRise = clip((f0 - pitchBase) / 60);          // semitone-ish rise vs personal pitch
      // jitter: cycle-to-cycle pitch variation (voice instability ~ tension)
      if (f0Hist.length >= 3) { let j=0; for (let i=1;i<f0Hist.length;i++) j += Math.abs(f0Hist[i]-f0Hist[i-1]);
        prosody.jitter = clip((j/(f0Hist.length-1)) / 15); }
    }
    energyBase = energyBase == null ? rms : 0.97*energyBase + 0.03*rms;
    prosody.energyRise = clip((rms - energyBase) / 0.05);
    // shimmer: amplitude variation
    ampHist.push(rms); if (ampHist.length > 8) ampHist.shift();
    if (ampHist.length >= 3) { let s=0; for (let i=1;i<ampHist.length;i++) s += Math.abs(ampHist[i]-ampHist[i-1]);
      prosody.shimmer = clip((s/(ampHist.length-1)) / 0.03); }
    // spectral centroid (brightness ~ arousal/tension)
    if (!freqBuf) freqBuf = new Float32Array(analyser.frequencyBinCount);
    analyser.getFloatFrequencyData(freqBuf);
    let num=0, den=0; const ny = audioCtx.sampleRate/2;
    for (let i=0;i<freqBuf.length;i++){ const mag=Math.pow(10, freqBuf[i]/20); const hz=i/freqBuf.length*ny; num+=hz*mag; den+=mag; }
    prosody.centroid = den>0 ? clip((num/den)/3000) : 0;        // normalized 0..1 (~3kHz cap)
    let trans = 0; for (let i=1;i<voicedHist.length;i++) if (voicedHist[i]!==voicedHist[i-1]) trans++;
    prosody.disfluency = clip((trans / 20) - 0.2);

    // ---- dimensional A/D/V from acoustic features (interim model; wav2vec2 backend = backend/voice_adv.py) ----
    // Arousal is the most reliably voice-encoded dimension (energy + pitch level/variability + brightness)
    prosody.a = clip(0.45*prosody.energyRise + 0.35*prosody.pitchRise + 0.25*prosody.centroid + 0.2*prosody.jitter);
    // Dominance: loud + low-pitched + steady (low jitter)
    prosody.d = clip(0.5*prosody.energyRise + 0.3*(1-clip(prosody.pitchRise)) - 0.3*prosody.jitter - 0.2*prosody.shimmer);
    // Valence is weakly voice-encoded: smoother voice (low jitter/shimmer) + moderate brightness leans positive
    prosody.v = clip(0.4 - 0.5*prosody.jitter - 0.4*prosody.shimmer + 0.2*prosody.centroid) - 0.2;
    prosody.arousal = prosody.a;                                // back-compat with prosody_pitch_rise signal
  } else {
    prosody.arousal *= 0.9; prosody.pitchRise *= 0.9; prosody.energyRise *= 0.9;
    prosody.a *= 0.9; prosody.d *= 0.9; prosody.v *= 0.9;
  }
}
// autocorrelation pitch detector (returns Hz, 0 if unvoiced/unclear)
function autocorrelate(buf, sr) {
  let best = -1, bestOff = -1, rms = 0;
  for (let i=0;i<buf.length;i++) rms += buf[i]*buf[i];
  if (Math.sqrt(rms/buf.length) < 0.01) return 0;
  const MIN = Math.floor(sr/400), MAX = Math.floor(sr/75);    // 75-400 Hz human voice
  for (let off=MIN; off<=MAX; off++) {
    let c = 0; for (let i=0;i<buf.length-off;i++) c += buf[i]*buf[i+off];
    if (c > best) { best = c; bestOff = off; }
  }
  return bestOff > 0 ? sr / bestOff : 0;
}

// ---------- rPPG: contactless heart rate / physiological arousal ----------
// Samples mean green channel of a forehead ROI over time; blood volume pulses modulate skin color.
function sampleRppg(face, t) {
  if (!face.faceLandmarks || !face.faceLandmarks[0]) return;
  const lm = face.faceLandmarks[0];
  const idx = [10, 67, 297, 109, 338, 151];   // forehead landmarks
  let minX=1, minY=1, maxX=0, maxY=0;
  for (const i of idx) { const p = lm[i]; minX=Math.min(minX,p.x); maxX=Math.max(maxX,p.x); minY=Math.min(minY,p.y); maxY=Math.max(maxY,p.y); }
  const vw = video.videoWidth, vh = video.videoHeight;
  const sx=minX*vw, sy=minY*vh, sw=Math.max(4,(maxX-minX)*vw), sh=Math.max(3,(maxY-minY)*vh);
  try {
    roiCtx.drawImage(video, sx, sy, sw, sh, 0, 0, roiCanvas.width, roiCanvas.height);
    const d = roiCtx.getImageData(0,0,roiCanvas.width,roiCanvas.height).data;
    let g=0, n=0; for (let i=0;i<d.length;i+=4){ g+=d[i+1]; n++; }
    rppg.buf.push({ t, g: g/n });
    while (rppg.buf.length && t - rppg.buf[0].t > 10000) rppg.buf.shift();  // keep 10s window
  } catch {}
}
function computeHR(t) {
  if (t - lastHrT < 1000) return; lastHrT = t;            // update ~1 Hz
  const b = rppg.buf;
  const dur = b.length ? (b[b.length-1].t - b[0].t)/1000 : 0;
  if (b.length < 60 || dur < 5) { rppg.quality = 0; return; }
  const fs = b.length / dur;
  const vals = b.map(x => x.g), mean = vals.reduce((a,c)=>a+c,0)/vals.length;
  const sig = vals.map(v => v - mean);
  const minLag = Math.floor(fs/4), maxLag = Math.floor(fs/0.7);  // 42–240 bpm band
  let best=0, bestLag=-1, energy=0;
  for (let i=0;i<sig.length;i++) energy += sig[i]*sig[i];
  for (let lag=minLag; lag<=maxLag && lag<sig.length; lag++) {
    let c=0; for (let i=0;i<sig.length-lag;i++) c += sig[i]*sig[i+lag];
    if (c > best) { best=c; bestLag=lag; }
  }
  if (bestLag > 0) {
    const hr = 60 * fs / bestLag;
    rppg.hr = rppg.hr ? 0.8*rppg.hr + 0.2*hr : hr;
    rppg.quality = clip(best/(energy||1) * 2);
    rppg.hrBase = rppg.hrBase == null ? rppg.hr : 0.995*rppg.hrBase + 0.005*rppg.hr;
    rppg.arousal = clip((rppg.hr - (rppg.hrBase || rppg.hr))/15) * rppg.quality;
  }
}

function startCalibration() { calibrating = 90; baseline = null; statusEl.textContent = "calibrating baseline… hold neutral"; }

// ---------- Per-person profiles (persist baseline across sessions) ----------
const PROFILE_KEY = "blr_profile_default";
function saveProfile() {
  if (!baseline) { $("backendStat").textContent = "no baseline yet — calibrate first"; return; }
  localStorage.setItem(PROFILE_KEY, JSON.stringify({ baseline, pitchBase, energyBase, ts: Date.now() }));
  statusEl.textContent = "profile saved";
}
function loadProfile() {
  const raw = localStorage.getItem(PROFILE_KEY);
  if (!raw) { statusEl.textContent = "no saved profile"; return; }
  try {
    const p = JSON.parse(raw);
    baseline = p.baseline; pitchBase = p.pitchBase ?? null; energyBase = p.energyBase ?? null;
    calibrating = 0; statusEl.textContent = "profile loaded — skipping calibration";
  } catch { statusEl.textContent = "profile corrupt"; }
}

// ---------- Main loop ----------
function loop() {
  if (!running) return;
  const t = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const face = faceLM.detectForVideo(video, t);
    const pose = poseLM.detectForVideo(video, t);
    const hands = handLM.detectForVideo(video, t);
    const qa = frameQuality(face, pose);
    frameQ = frameQ + 0.2*(qa.q - frameQ);          // smooth the quality signal
    sampleRppg(face, t); computeHR(t);              // contactless heart-rate / physiological arousal
    let feat = extractFeatures(face, pose, hands);
    if (feat) feat = smoothFeatures(feat);          // denoise features before interpretation
    drawOverlay(face, pose, hands);
    updateQualityUI(qa);
    if (feat) {
      if (frameQ > 0.5) updateBaseline(feat);     // don't pollute baseline with bad frames
      if (calibrating > 0) {
        if (frameQ > 0.6) calibrating--;           // only count clean frames toward calibration
        statusEl.textContent = `calibrating baseline… hold neutral (${calibrating})`;
        if (calibrating === 0) statusEl.textContent = "running";
      } else {
        smoothUpdate(interpret(feat));           // EMA every frame (cheap), scaled by frameQ
        if (t - lastRender > RENDER_MS) { renderSmoothed(); lastRender = t; }  // paint ~4x/sec
      }
    }
  }
  requestAnimationFrame(loop);
}

// ---------- Feature extraction ----------
function bsMap(face) {
  const m = {};
  if (face.faceBlendshapes && face.faceBlendshapes[0])
    for (const c of face.faceBlendshapes[0].categories) m[c.categoryName] = c.score;
  return m;
}
function extractFeatures(face, pose, hands) {
  if (!face.faceLandmarks || !face.faceLandmarks[0]) return null;
  const b = bsMap(face);
  lastBlendshapes = b;                       // keep raw blendshapes for the trained model
  const noseY = face.faceLandmarks[0][1].y; // nose tip
  noseHist.push(noseY); if (noseHist.length > 30) noseHist.shift();

  const f = {
    smile: (b.mouthSmileLeft + b.mouthSmileRight) / 2 || 0,
    cheek: (b.cheekSquintLeft + b.cheekSquintRight) / 2 || 0,
    eyeSquint: (b.eyeSquintLeft + b.eyeSquintRight) / 2 || 0,        // AU7
    browDown: (b.browDownLeft + b.browDownRight) / 2 || 0,          // AU4
    browInnerUp: b.browInnerUp || 0,                                 // AU1
    browOuterUp: (b.browOuterUpLeft + b.browOuterUpRight) / 2 || 0,  // AU2
    frown: (b.mouthFrownLeft + b.mouthFrownRight) / 2 || 0,          // AU15 lip-corner depressor
    eyeWide: (b.eyeWideLeft + b.eyeWideRight) / 2 || 0,             // AU5 upper-lid raiser
    noseSneer: (b.noseSneerLeft + b.noseSneerRight) / 2 || 0,       // AU9 nose wrinkler
    upperLip: (b.mouthUpperUpLeft + b.mouthUpperUpRight) / 2 || 0,  // AU10 upper-lip raiser
    lipPress: (b.mouthPressLeft + b.mouthPressRight) / 2 || 0,      // AU23/24 lip press
    lipStretch: (b.mouthStretchLeft + b.mouthStretchRight) / 2 || 0,// AU20 lip stretch (fear/tension)
    chinRaise: (b.mouthShrugUpper + b.mouthShrugLower) / 2 || 0,    // AU17 chin raiser (pout)
    dimple: (b.mouthDimpleLeft + b.mouthDimpleRight) / 2 || 0,      // AU14 dimpler (contempt)
    lowerLipDown: (b.mouthLowerDownLeft + b.mouthLowerDownRight)/2 || 0, // AU16 lower-lip depressor
    lipPucker: b.mouthPucker || 0,                                  // AU18 lip pucker (pursed)
    lipFunnel: b.mouthFunnel || 0,                                  // AU22 funneler
    lipSuck: (b.mouthRollLower + b.mouthRollUpper) / 2 || 0,        // AU28 lip suck/bite
    jawJut: b.jawForward || 0,                                      // AU29 jaw thrust (tension/defiance)
    cheekPuff: b.cheekPuff || 0,                                    // AU33/34 cheek puff (exasperation)
    mouthAsym: Math.abs((b.mouthSmileLeft||0) - (b.mouthSmileRight||0)), // unilateral smile -> smirk/contempt
    blinkRate: blinkUpdate((b.eyeBlinkLeft + b.eyeBlinkRight)/2 || 0, performance.now()), // AU45 rate
    jawOpen: b.jawOpen || 0,                                         // AU26 jaw drop
    gazeAway: Math.max(b.eyeLookOutLeft||0, b.eyeLookOutRight||0,
                       b.eyeLookInLeft||0, b.eyeLookInRight||0,
                       b.eyeLookUpLeft||0, b.eyeLookDownLeft||0),
    nod: nodEnergy(),
    // body (pose) features — filled below
    lean: 0, leanBack: 0, closedArms: 0, selfTouch: 0, shrug: 0, headTilt: 0, torsoTurn: 0,
    handsOnHips: 0, expansive: 0, handToNeck: 0, fidget: 0,
    // hand (finger) features — filled below
    openPalm: 0, handsTogether: 0, pointing: 0
  };

  if (pose.landmarks && pose.landmarks[0]) {
    const p = pose.landmarks[0];
    const L = (i) => p[i];
    const dist = (a, b2) => Math.hypot(a.x-b2.x, a.y-b2.y);
    const shW = Math.max(0.05, dist(L(11), L(12))); // shoulder width = body scale
    const shMidZ = (L(11).z + L(12).z) / 2, hipMidZ = (L(23).z + L(24).z) / 2;
    f.lean = hipMidZ - shMidZ;             // shoulders toward cam => forward lean
    f.leanBack = Math.max(0, shMidZ - hipMidZ);
    // ventral denial (Navarro): torso turning away => one shoulder rotates back (z diff grows)
    f.torsoTurn = clip(Math.abs(L(11).z - L(12).z) * 2.5);
    const midX = (L(11).x + L(12).x) / 2;
    const shMidY = (L(11).y + L(12).y) / 2;
    // arms crossed over torso
    f.closedArms = ((L(15).x > midX) && (L(16).x < midX) &&
                    (L(15).y < L(23).y) && (L(16).y < L(24).y)) ? 1 : 0;
    // hand near face (self-touch adaptor)
    const nose = L(0);
    f.selfTouch = (dist(L(15),nose) < 0.18 || dist(L(16),nose) < 0.18) ? 1 : 0;
    // shoulder shrug: shoulders rise toward ears -> neck length shrinks vs baseline
    const earMidY = (L(7).y + L(8).y) / 2;
    f.neckLen = (shMidY - earMidY) / shW;  // normalized; baseline-compared in activations
    // head tilt: angle of ear line off horizontal
    f.headTilt = Math.min(1, Math.abs(Math.atan2(L(8).y - L(7).y, L(8).x - L(7).x)) / 0.5);
    // hands on hips: wrist near hip + elbow flared outside shoulder
    const hip = (s) => dist(L(s===0?15:16), L(s===0?23:24)) / shW;
    const akimbo = (w,h,e,sh) => (dist(L(w),L(h))/shW < 0.6) && (Math.abs(L(e).x-L(sh).x) > Math.abs(L(w).x-L(sh).x)*0.8);
    f.handsOnHips = ((akimbo(15,23,13,11)?1:0) + (akimbo(16,24,14,12)?1:0)) / 2;
    // expansive (power) posture: elbows + wrists spread far from torso center
    const spread = (dist(L(13),{x:midX,y:shMidY}) + dist(L(14),{x:midX,y:shMidY}) +
                    dist(L(15),{x:midX,y:shMidY}) + dist(L(16),{x:midX,y:shMidY})) / (4*shW);
    f.expansive = clip((spread - 0.9) * 1.1);
    // hand to neck/nape (self-soothing)
    f.handToNeck = (dist(L(15),{x:midX,y:shMidY}) < 0.12 || dist(L(16),{x:midX,y:shMidY}) < 0.12) ? 1 : 0;
    // fidget: wrist motion energy over recent frames
    wristHist.push({lx:L(15).x, ly:L(15).y, rx:L(16).x, ry:L(16).y});
    if (wristHist.length > 12) wristHist.shift();
    f.fidget = fidgetEnergy(shW);
  }

  if (hands && hands.landmarks && hands.landmarks.length) {
    const handFeats = hands.landmarks.map(h => handShape(h));
    f.openPalm = Math.max(...handFeats.map(x => x.open));
    f.pointing = Math.max(...handFeats.map(x => x.point));
    if (hands.landmarks.length >= 2) {
      const c0 = centroid(hands.landmarks[0]), c1 = centroid(hands.landmarks[1]);
      f.handsTogether = Math.hypot(c0.x-c1.x, c0.y-c1.y) < 0.12 ? 1 : 0;
    }
  }
  return f;
}
// per-hand shape: openness (fingers extended/spread) & pointing (index out, rest folded)
function handShape(h) {
  const palm = Math.max(0.02, Math.hypot(h[0].x-h[9].x, h[0].y-h[9].y)); // wrist->middle MCP
  const tip = (i) => Math.hypot(h[i].x-h[0].x, h[i].y-h[0].y) / palm;    // tip dist from wrist
  const idx=tip(8), mid=tip(12), rng=tip(16), pky=tip(20);
  const open = clip(((idx+mid+rng+pky)/4 - 1.3) * 1.3);
  const point = clip((idx - Math.max(mid,rng,pky)) * 1.2) * (idx > 1.4 ? 1 : 0);
  return { open, point };
}
function centroid(h){ let x=0,y=0; for(const p of h){x+=p.x;y+=p.y;} return {x:x/h.length, y:y/h.length}; }
function fidgetEnergy(shW){
  if (wristHist.length < 4) return 0;
  let s=0;
  for (let i=1;i<wristHist.length;i++){ const a=wristHist[i],b=wristHist[i-1];
    s += Math.hypot(a.lx-b.lx,a.ly-b.ly) + Math.hypot(a.rx-b.rx,a.ry-b.ry); }
  return clip((s/(wristHist.length*shW)) * 2.2);
}
// ---------- Quality gating (reject bad frames so garbage doesn't drive interpretation) ----------
function sampleBrightness() {
  try { lumaCtx.drawImage(video, 0, 0, 32, 24);
    const d = lumaCtx.getImageData(0,0,32,24).data; let s=0;
    for (let i=0;i<d.length;i+=4) s += 0.299*d[i]+0.587*d[i+1]+0.114*d[i+2];
    return s / (d.length/4) / 255;                 // 0..1 mean luma
  } catch { return 0.5; }
}
function frameQuality(face, pose) {
  if (!face.faceLandmarks || !face.faceLandmarks[0]) return { q: 0, reasons: ["no face"] };
  const lm = face.faceLandmarks[0]; const reasons = []; let q = 1;
  // head yaw via face-mesh symmetry (234 = right cheek, 454 = left cheek, 1 = nose tip)
  const fw = Math.abs(lm[454].x - lm[234].x) || 0.1;
  const yaw = Math.abs((lm[1].x - lm[234].x) - (lm[454].x - lm[1].x)) / fw;
  if (yaw > 0.45) { q *= 0.45; reasons.push("face turned"); }
  // lighting
  const luma = sampleBrightness();
  if (luma < 0.16) { q *= 0.5; reasons.push("too dark"); }
  else if (luma > 0.93) { q *= 0.7; reasons.push("overexposed"); }
  // body visibility
  if (pose.landmarks && pose.landmarks[0]) {
    const vis = ((pose.landmarks[0][11].visibility ?? 1) + (pose.landmarks[0][12].visibility ?? 1)) / 2;
    if (vis < 0.5) { q *= 0.75; reasons.push("body occluded"); }
  }
  return { q: clip(q), reasons };
}

// blink edge-detector -> blinks per recent window, normalized (resting ~0.25-0.45 Hz)
function blinkUpdate(blinkVal, t) {
  if (blinkOpen && blinkVal > 0.5) { blinkOpen = false; blinkTimes.push(t); }
  else if (!blinkOpen && blinkVal < 0.2) { blinkOpen = true; }
  while (blinkTimes.length && t - blinkTimes[0] > 6000) blinkTimes.shift();
  const hz = blinkTimes.length / 6;        // blinks/sec over 6s
  return clip((hz - 0.4) / 0.8);           // >~0.4Hz starts registering as elevated
}
// geometric mean — AU constellation co-activation (ALL components must be present)
const gm = (...xs) => Math.pow(xs.reduce((p,x)=>p*Math.max(0.0001,x),1), 1/xs.length);
// EMA-smooth every numeric feature so per-frame landmark jitter doesn't drive activations
function smoothFeatures(f) {
  const sf = { ...f };
  for (const k in f) {
    const v = f[k];
    if (typeof v === "number" && Number.isFinite(v)) {
      FEAT_EMA[k] = (FEAT_EMA[k] === undefined) ? v : (1 - FEAT_A) * FEAT_EMA[k] + FEAT_A * v;
      sf[k] = FEAT_EMA[k];
    }
  }
  return sf;
}
const deadzone = (x, t) => (Math.abs(x) < t ? 0 : x);   // suppress tiny (noise-level) dimension values
const median = (arr) => { const a = [...arr].sort((x,y)=>x-y); return a[Math.floor(a.length/2)]; };

function nodEnergy() {
  if (noseHist.length < 8) return 0;
  let signChanges = 0, prev = 0;
  for (let i = 1; i < noseHist.length; i++) {
    const d = noseHist[i] - noseHist[i-1];
    if (Math.abs(d) > 0.002) { const s = Math.sign(d); if (prev && s !== prev) signChanges++; prev = s; }
  }
  return Math.min(1, signChanges / 6);
}

// ---------- Personal baseline (honesty rule: calibrate before flagging deviation) ----------
function updateBaseline(f) {
  const keys = ["smile","browDown","browInnerUp","browOuterUp","gazeAway","lean","neckLen",
                "frown","eyeWide","noseSneer","upperLip","lipPress","lipStretch","jawOpen","eyeSquint",
                "chinRaise","dimple","lowerLipDown","lipPucker","lipFunnel","lipSuck","jawJut","torsoTurn"];
  if (!baseline) baseline = {};
  const a = calibrating > 0 ? 0.1 : 0.01; // adapt fast during calibration, slow after
  for (const k of keys) {
    const v = f[k];
    if (!Number.isFinite(v)) continue;            // skip frames missing this feature
    baseline[k] = (baseline[k] === undefined) ? v : (1-a)*baseline[k] + a*v;
  }
}
const dev = (f, k) => Math.max(0, (f[k] - (baseline?.[k] ?? 0)));

// ---------- Interpretation: map features -> active signals -> fused states ----------
function activations(f) {
  // {signalId: activation 0..1}
  return {
    // smile genuineness boosted by Duchenne markers (cheek + eye squint), AU6+AU12
    face_smile_genuine: clip((dev(f,"smile")*1.4) * (0.35 + 0.45*f.cheek + 0.2*f.eyeSquint)),
    face_brow_raise: clip((dev(f,"browInnerUp")+dev(f,"browOuterUp"))*1.2),
    face_brow_lower: clip(dev(f,"browDown")*1.6),
    face_mouth_frown: clip(dev(f,"frown")*1.7),                       // AU15 -> sadness
    face_eye_widen: clip(dev(f,"eyeWide")*1.6),                       // AU5 -> fear/alert
    face_nose_wrinkle: clip((dev(f,"noseSneer")*1.3 + dev(f,"upperLip")*0.8)), // AU9/10 -> disgust
    face_lip_press: clip((dev(f,"lipPress")*1.4 + dev(f,"lipStretch")*0.6)),   // AU23/24/20 -> tension
    face_jaw_drop: clip(dev(f,"jawOpen")*1.5),                        // AU26 -> surprise
    // additional single facial reactions
    face_lip_pucker: clip((dev(f,"lipPucker")*1.3 + dev(f,"lipFunnel")*0.7)),  // AU18/22 -> pursed/disapproval
    face_lip_suck: clip(dev(f,"lipSuck")*1.6),                        // AU28 -> lip bite/anxiety
    face_chin_raise: clip(dev(f,"chinRaise")*1.5),                    // AU17 -> pout/doubt
    face_smirk: clip(f.mouthAsym*2.2 + dev(f,"dimple")*0.8),          // unilateral -> smirk/contempt
    face_jaw_jut: clip(dev(f,"jawJut")*1.6),                          // AU29 -> tension/defiance
    face_cheek_puff: clip(f.cheekPuff*1.4),                           // -> exasperation/sigh
    face_blink_rate: clip(f.blinkRate),                              // AU45 rate -> arousal/stress
    // ---- emotion prototypes: AU constellations (Ekman/EMFACS), higher reliability than single AUs ----
    emotion_happiness: clip(gm(dev(f,"smile"), 0.3+f.cheek) * 2.0),
    emotion_sadness:   clip(gm(dev(f,"frown"), dev(f,"browInnerUp")+dev(f,"browDown")+0.05, dev(f,"chinRaise")+0.05) * 3.0),
    emotion_surprise:  clip(gm(dev(f,"browOuterUp")+dev(f,"browInnerUp"), dev(f,"eyeWide"), dev(f,"jawOpen")) * 3.2),
    emotion_fear:      clip(gm(dev(f,"browInnerUp"), dev(f,"eyeWide"), dev(f,"lipStretch")+0.03) * 3.2),
    emotion_anger:     clip(gm(dev(f,"browDown"), dev(f,"eyeWide")+dev(f,"eyeSquint"), dev(f,"lipPress")+0.03) * 3.2),
    emotion_disgust:   clip(gm(dev(f,"noseSneer"), dev(f,"upperLip"), dev(f,"lowerLipDown")+0.05) * 3.0),
    emotion_contempt:  clip((f.mouthAsym + dev(f,"dimple")*0.6) * 2.4),
    gaze_aversion: clip((dev(f,"gazeAway"))*1.5),
    regulator_head_nod: clip(f.nod),
    // body / posture
    posture_lean_forward: clip(Math.max(0, (f.lean - (baseline?.lean??0)))*6),
    posture_lean_back: clip(f.leanBack*6),
    posture_ventral_denial: clip(dev(f,"torsoTurn")*3.5),   // Navarro: torso turned away from interlocutor
    posture_closed_arms: f.closedArms,
    posture_shrug: clip(((baseline?.neckLen ?? f.neckLen) - f.neckLen) * 3),
    posture_hands_on_hips: f.handsOnHips,
    posture_expansive: f.expansive,
    regulator_head_tilt: clip(f.headTilt),
    adaptor_self_touch_face: f.selfTouch,
    adaptor_hand_to_neck: f.handToNeck,
    body_fidget: f.fidget,
    // hands / gestures
    gesture_open_palm: f.openPalm,
    gesture_hands_together: f.handsTogether,
    gesture_pointing: f.pointing,
    // voice / prosody (only when speaking)
    prosody_pitch_rise: prosody.voiced ? clip(prosody.arousal) : 0,
    prosody_pause_disfluency: prosody.voiced ? clip(prosody.disfluency) : 0,
    // physiological (rPPG) — contactless, harder to fake than posed expression
    physio_arousal: rppg.quality > 0.25 ? clip(rppg.arousal) : 0
  };
}
// Evidence-backed constructs: multiple signals vote; confidence scales with corroboration.
function computeConstructs(act) {
  const C = KB.constructs || {};
  const byId = Object.fromEntries(KB.signals.map(s => [s.id, s]));
  const out = [];
  for (const [key, c] of Object.entries(C)) {
    if (key.startsWith("_")) continue;
    let pos = 0, neg = 0; const active = [];
    for (const [sig, w] of Object.entries(c.positive || {})) {
      const a = act[sig] || 0;
      if (a > 0.12) { pos += w * a; active.push(byId[sig]?.label || sig); }
    }
    for (const [sig, w] of Object.entries(c.negative || {})) neg += w * (act[sig] || 0);
    const nCues = active.length;
    if (nCues === 0) continue;
    const corrob = Math.min(1, nCues / (c.min_cues || 2));
    const conf = Math.min(0.92, clip(pos * 0.7 * (0.55 + 0.45 * corrob) - 0.5 * neg)); // cap: never claim certainty
    out.push({ state: c.label, p: conf, contributors: active,
      prototype: nCues >= (c.min_cues || 2), nCues,
      evid: (c.reliability >= 3 ? "moderate" : "weak"),
      cite: (c.evidence || "").split(";")[0].trim() });
  }
  return out;
}

function interpret(f) {
  const act = activations(f);
  const byId = Object.fromEntries(KB.signals.map(s => [s.id, s]));
  const states = {}; // state -> {p, contributors:[], caveats:Set}
  const activeSignals = [];

  for (const [id, a] of Object.entries(act)) {
    // hysteresis: latch on above SIG_ON, stay on until below SIG_OFF -> no flicker at a single cutoff
    const on = activeSig.has(id) ? (a > SIG_OFF) : (a > SIG_ON);
    if (on) activeSig.add(id); else { activeSig.delete(id); continue; }
    const sig = byId[id]; if (!sig) continue;
    activeSignals.push({ id, label: sig.label, a, rel: sig.inference_reliability });
    for (const interp of sig.interpretations) {
      if (interp.prior_confidence == null) continue;
      const contrib = a * interp.prior_confidence * (sig.inference_reliability / 5) * frameQ;
      const st = states[interp.state] || (states[interp.state] = { p: 0, contributors: [], caveats: new Set(), evid: interp.evidence, prototype: false });
      st.p = 1 - (1 - st.p) * (1 - contrib);   // noisy-OR
      st.contributors.push(sig.label);
      if (interp.prototype) st.prototype = true; // AU constellation = already multi-cue
      if (interp.caveats) st.caveats.add(interp.caveats);
    }
  }

  // fuse the in-browser TRAINED model (data-driven) if present — highest-weight contributor
  if (hasModel() && lastBlendshapes) {
    const mp = modelPredict(lastBlendshapes);
    if (mp) for (const [emo, prob] of Object.entries(mp)) {
      if (prob < 0.2) continue;
      const st = states[emo] || (states[emo] = { p:0, contributors:[], caveats:new Set(), evid:"moderate", prototype:false });
      const contrib = prob * 0.85 * frameQ;              // trained model weight 0.85
      st.p = 1 - (1 - st.p) * (1 - contrib);
      st.contributors.push("trained model"); st.prototype = true;
    }
  }

  // fuse FACS backend emotions (calibrated) if available — second independent estimate
  if (backendEmotions) {
    for (const [emo, prob] of Object.entries(backendEmotions)) {
      if (prob < 0.15) continue;
      const st = states[emo] || (states[emo] = { p:0, contributors:[], caveats:new Set(), evid:"moderate", prototype:false });
      const contrib = prob * 0.7 * frameQ;                 // backend weight 0.7
      st.p = 1 - (1 - st.p) * (1 - contrib);
      st.contributors.push("FACS backend (calibrated)");
      st.prototype = true;
    }
  }

  // evidence-backed CONSTRUCTS: constellations vote -> corroborated, confident, cited reads
  for (const cs of computeConstructs(act)) {
    const st = states[cs.state] || (states[cs.state] = { p:0, contributors:[], caveats:new Set(), evid:cs.evid, prototype:false });
    st.p = 1 - (1 - st.p) * (1 - cs.p * frameQ);
    for (const c of cs.contributors) st.contributors.push(c);
    if (cs.prototype) st.prototype = true;          // >=min_cues corroborating -> not hedged
    if (cs.cite) st.caveats.add("Evidence: " + cs.cite);
  }

  // context reweighting (EMOTIC principle): adjust priors by selected context
  applyContext(states, $("ctx").value);

  // hard constraint: a state needs >=2 corroborating signals to be elevated
  const ranked = Object.entries(states).map(([state, v]) => {
    const contributors = [...new Set(v.contributors)];
    return { state, p: v.p, n: contributors.length, contributors,
      caveats: [...v.caveats], evid: v.evid, weak: contributors.length < 2 && !v.prototype };
  }).sort((x,y) => y.p - x.p).slice(0, 10);

  const dims = estimateDims(f, act);
  dims.v *= frameQ; dims.a *= frameQ; dims.d *= frameQ;   // damp dims on low-quality frames
  activeSignals.sort((x,y)=>y.a-x.a);
  return { signals: activeSignals, states: ranked, dims };
}
function applyContext(states, ctx) {
  const down = (name, factor) => { if (states[name]) states[name].p *= factor; };
  if (ctx === "cold")     down("defensiveness / discomfort", 0.3); // crossed arms likely cold
  if (ctx === "interview"){ if (states["engagement / interest"]) states["engagement / interest"].p *= 1.1; }
  if (ctx === "casual")   down("cognitive_load / thinking", 0.7);
}
function estimateDims(f, act) {
  let v = 0, a = 0, d = 0;
  v += (act.face_smile_genuine||0)*0.8
     - (act.face_brow_lower||0)*0.4 - (act.face_mouth_frown||0)*0.7
     - (act.face_nose_wrinkle||0)*0.7 - (act.face_lip_press||0)*0.5
     - (act.posture_closed_arms||0)*0.3;
  v += (act.gesture_open_palm||0)*0.3
     - (act.adaptor_hand_to_neck||0)*0.3 - (act.posture_lean_back||0)*0.2 - (act.body_fidget||0)*0.2
     - (act.posture_ventral_denial||0)*0.3;   // turning away = discomfort (Navarro)
  a += (act.face_brow_raise||0)*0.4 + (act.face_eye_widen||0)*0.6 + (act.face_jaw_drop||0)*0.5
     + (act.adaptor_self_touch_face||0)*0.4 + (act.regulator_head_nod||0)*0.3
     + (act.body_fidget||0)*0.5 + (act.posture_shrug||0)*0.3 + (act.gesture_hands_together||0)*0.2
     + (act.prosody_pitch_rise||0)*0.6      // vocal arousal is a strong, independent arousal cue
     + (act.physio_arousal||0)*0.5;          // contactless physiological arousal (rPPG)
  d += (act.posture_lean_forward||0)*0.5 + (act.face_nose_wrinkle||0)*0.2
     + (act.posture_expansive||0)*0.6 + (act.posture_hands_on_hips||0)*0.5 + (act.gesture_pointing||0)*0.4
     - (act.gaze_aversion||0)*0.3 - (act.face_eye_widen||0)*0.3 - (act.posture_closed_arms||0)*0.2
     - (act.posture_shrug||0)*0.4 - (act.adaptor_hand_to_neck||0)*0.3;
  // fuse independent voice A/D/V estimate (only while speaking) — multimodal dimensional fusion
  if (prosody.voiced) { v += prosody.v*0.4; a += prosody.a*0.4; d += prosody.d*0.4; }
  return { v: deadzone(clamp(v), 0.07), a: deadzone(clamp(a), 0.06), d: deadzone(clamp(d), 0.07) };
}

// ---------- Smoothing (EMA every frame) + throttled render (~4x/sec) ----------
function smoothUpdate(raw) {
  const D = SMOOTH.dims;
  D.v += SM_A*(raw.dims.v - D.v); D.a += SM_A*(raw.dims.a - D.a); D.d += SM_A*(raw.dims.d - D.d);

  const seenStates = new Set();
  for (const s of raw.states) {
    seenStates.add(s.state);
    const cur = SMOOTH.states.get(s.state) || { p:0 };
    cur.p += SM_A*(s.p - cur.p);
    cur.contributors = s.contributors; cur.caveats = s.caveats; cur.evid = s.evid; cur.weak = s.weak;
    SMOOTH.states.set(s.state, cur);
  }
  for (const [k,v] of SMOOTH.states) {                 // decay states that dropped out
    if (!seenStates.has(k)) { v.p *= (1-SM_A); if (v.p < 0.012) SMOOTH.states.delete(k); }
  }

  const seenSig = new Set();
  for (const s of raw.signals) {
    seenSig.add(s.id);
    const cur = SMOOTH.signals.get(s.id) || { a:0 };
    cur.a += SM_A*(s.a - cur.a); cur.label = s.label; cur.rel = s.rel;
    SMOOTH.signals.set(s.id, cur);
  }
  for (const [k,v] of SMOOTH.signals) {
    if (!seenSig.has(k)) { v.a *= (1-SM_A); if (v.a < 0.05) SMOOTH.signals.delete(k); }
  }
}

function renderSmoothed() {
  const dims = SMOOTH.dims;
  let states = [...SMOOTH.states.entries()].map(([state,v]) => ({ state, ...v }))
                  .filter(s => s.p > 0.02).sort((a,b)=>b.p-a.p);
  // Lead with corroborated (confident) reads; only fall back to hedged single-cue states if none.
  const confident = states.filter(s => !s.weak);
  states = (confident.length ? confident : states).slice(0, 6);
  const signals = [...SMOOTH.signals.entries()].map(([id,v]) => ({ id, ...v }))
                  .filter(s => s.a > 0.15).sort((a,b)=>b.a-a.a).slice(0, 10);
  render(signals, states, dims);
}

function updateQualityUI(qa) {
  const el = $("quality"); if (!el) return;
  const pct = Math.round(frameQ * 100);
  const col = frameQ > 0.7 ? "var(--good)" : frameQ > 0.4 ? "var(--warn)" : "var(--bad)";
  el.style.borderColor = col; el.style.color = col;
  el.textContent = qa.reasons.length ? `tracking ${pct}% · ${qa.reasons.join(", ")}` : `tracking ${pct}%`;
}

// ---------- Coach reasoning (on-device; optional LLM backend = backend/explain.py) ----------
// Self-coaching framing (the chosen product lane): actionable feedback on YOUR OWN presence.
const COACH = {
  "Engagement / Interest": "Strong engagement. You're leaning in and tracking. Keep this energy.",
  "Disengagement / Withdrawal": "Reading as pulled-back. Lean in, uncross, and re-engage eye contact.",
  "Discomfort / Anxiety": "Signs of tension. Slow your pace, drop your shoulders, steady your hands.",
  "Confidence / Dominance": "Confident, expansive presence. Keep it warm so it doesn't overpower.",
  "Rapport / Openness": "Warm and open. This is what builds rapport. Nice.",
  "happiness": "Positive affect reads clearly.", "sadness": "A low, heavy read. Breathe and reset.",
  "anger": "Reads tense. Soften the brow and jaw.", "surprise": "Big reaction registered.",
};
function generateInsight(states, dims) {
  if (frameQ < 0.4) return "Tracking too low for a read. Face the camera in good light.";
  const top = states.find(s => !s.weak) || states[0];
  if (!top || top.p < 0.25) return "Neutral and steady. No strong signal right now.";
  let line = COACH[top.state] || `Reading: ${top.state}.`;
  const drivers = (top.contributors || []).slice(0, 3).join(", ");
  if (drivers) line += `  ·  cues: ${drivers}`;
  return line;
}

// ---------- Render ----------
function render(signals, states, dims) {
  $("valence").textContent = dims.v.toFixed(2);
  $("arousal").textContent = dims.a.toFixed(2);
  $("domin").textContent = dims.d.toFixed(2);
  const coach = $("coach"); if (coach) coach.textContent = generateInsight(states, dims);
  const vit = $("vitals");
  if (vit) vit.textContent = rppg.quality > 0.3 ? `♥ ~${Math.round(rppg.hr)} bpm (rPPG${prosody.voiced ? " · 🎤 voice A/D/V" : ""})`
                                                : (prosody.voiced ? "🎤 voice A/D/V active" : "");

  // publish the current read so the session recorder (session.js) can sample it
  window.PRESENCE = {
    ts: Date.now(), running, frameQ,
    v: dims.v, a: dims.a, d: dims.d,
    voiced: prosody.voiced, hr: rppg.hr, hrQuality: rppg.quality,
    states: states.map(s => ({ state: s.state, p: s.p, weak: s.weak })),
    coach: coach ? coach.textContent : ""
  };

  // precision: commit to a primary read only when confident AND clearly ahead of #2 (else abstain)
  const prim = $("primary");
  if (frameQ < 0.4) { prim.textContent = "⏸ tracking too low to read"; prim.style.color = "var(--bad)"; }
  else if (!states.length || states[0].p < 0.25) { prim.textContent = "insufficient signal"; prim.style.color = "var(--mut)"; }
  else {
    const margin = states[0].p - (states[1]?.p ?? 0);
    if (margin < 0.06) { prim.textContent = `ambiguous: ${states[0].state} / ${states[1].state}`; prim.style.color = "var(--warn)"; }
    else { prim.textContent = `Primary read: ${states[0].state} (${(states[0].p*100).toFixed(0)}%)`; prim.style.color = "var(--accent)"; }
  }

  $("states").innerHTML = states.length ? states.map(s => `
    <div class="state">
      <div class="row"><span>${s.state} ${s.weak ? '<span class="pill e-weak">needs corroboration</span>' : ''}</span>
        <span>${(s.p*100).toFixed(0)}%</span></div>
      <div class="bar"><i style="width:${Math.min(100,s.p*100)}%;${s.weak?'background:var(--warn)':''}"></i></div>
      <div class="contrib">from: ${s.contributors.join(", ")} · evidence: <span class="e-${s.evid}">${s.evid}</span></div>
      ${s.caveats.length ? `<div class="caveat">⚠ ${s.caveats[0]}</div>` : ''}
    </div>`).join("")
    : '<div class="caveat">Insufficient evidence — no state above threshold.</div>';

  $("signals").innerHTML = signals.length ? signals.map(s => `
    <div class="sig"><span class="nm">${s.label}</span>
      <span class="v">${(s.a*100).toFixed(0)}% · rel ${s.rel}/5</span></div>`).join("")
    : '<div class="caveat">No active signals.</div>';
}

// full-body skeleton connections (MediaPipe 33-pt topology)
const POSE_CONN = [
  [11,12],[11,23],[12,24],[23,24],                       // torso
  [11,13],[13,15],[15,17],[15,19],[15,21],[17,19],       // left arm + hand
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],       // right arm + hand
  [23,25],[25,27],[27,29],[27,31],[29,31],               // left leg
  [24,26],[26,28],[28,30],[28,32],[30,32],               // right leg
  [9,10],[0,11],[0,12]                                    // mouth, neck links
];
function drawOverlay(face, pose, hands) {
  ctx2d.clearRect(0,0,canvas.width,canvas.height);
  drawFace(face);
  if (pose.landmarks && pose.landmarks[0]) {
    const p = pose.landmarks[0];
    ctx2d.strokeStyle = "rgba(69,211,173,.5)"; ctx2d.lineWidth = 2.5;
    for (const [a,b] of POSE_CONN){
      if ((p[a].visibility ?? 1) < 0.4 || (p[b].visibility ?? 1) < 0.4) continue;
      ctx2d.beginPath();
      ctx2d.moveTo(p[a].x*canvas.width, p[a].y*canvas.height);
      ctx2d.lineTo(p[b].x*canvas.width, p[b].y*canvas.height); ctx2d.stroke();
    }
    ctx2d.fillStyle = "#45d3ad";
    for (let i=0;i<p.length;i++){
      if ((p[i].visibility ?? 1) < 0.4) continue;
      ctx2d.beginPath(); ctx2d.arc(p[i].x*canvas.width, p[i].y*canvas.height, 3.5, 0, 7); ctx2d.fill();
    }
  }
  if (hands && hands.landmarks && drawer) {
    for (const h of hands.landmarks) {
      drawer.drawConnectors(h, HandLandmarker.HAND_CONNECTIONS, { color:"#45d3ad", lineWidth:2 });
      ctx2d.fillStyle = "#7fe8cf";
      for (const pt of h){ ctx2d.beginPath(); ctx2d.arc(pt.x*canvas.width, pt.y*canvas.height, 2.6, 0, 7); ctx2d.fill(); }
    }
  }
}

function drawFace(face) {
  if (!drawer || !face.faceLandmarks || !face.faceLandmarks[0]) return;
  const lm = face.faceLandmarks[0];
  const C = FaceLandmarker;
  // 1) faint full mesh (the dense ~478-point tessellation)
  drawer.drawConnectors(lm, C.FACE_LANDMARKS_TESSELATION, { color: "rgba(120,200,185,0.16)", lineWidth: 0.5 });
  // 2) every landmark as a small dot — visible density
  ctx2d.fillStyle = "rgba(170,235,215,0.55)";
  for (const pt of lm) { ctx2d.beginPath(); ctx2d.arc(pt.x*canvas.width, pt.y*canvas.height, 1.1, 0, 7); ctx2d.fill(); }
  // 3) highlight expression-relevant contours
  const hl = (set, col) => drawer.drawConnectors(lm, set, { color: col, lineWidth: 1.4 });
  hl(C.FACE_LANDMARKS_FACE_OVAL,     "rgba(69,211,173,0.7)");
  hl(C.FACE_LANDMARKS_LEFT_EYE,      "#7fe8cf");
  hl(C.FACE_LANDMARKS_RIGHT_EYE,     "#7fe8cf");
  hl(C.FACE_LANDMARKS_LEFT_EYEBROW,  "#d8a23a");
  hl(C.FACE_LANDMARKS_RIGHT_EYEBROW, "#d8a23a");
  hl(C.FACE_LANDMARKS_LIPS,          "#f0625f");
  if (C.FACE_LANDMARKS_LEFT_IRIS)  hl(C.FACE_LANDMARKS_LEFT_IRIS,  "#45d3ad");
  if (C.FACE_LANDMARKS_RIGHT_IRIS) hl(C.FACE_LANDMARKS_RIGHT_IRIS, "#45d3ad");
}

// ---------- utils ----------
const clip = (x) => Math.max(0, Math.min(1, x));
const clamp = (x) => Math.max(-1, Math.min(1, x));

// Minimal inline KB so the demo still runs if fetch is blocked (file://).
const FALLBACK_KB = { signals: [
  {id:"face_smile_genuine",label:"Genuine smile (AU6+AU12)",inference_reliability:3,interpretations:[{state:"positive_affect / enjoyment",prior_confidence:0.55,evidence:"moderate",caveats:"Presence != felt emotion (Barrett 2019)."}]},
  {id:"face_brow_raise",label:"Brow raise (AU1+AU2)",inference_reliability:2,interpretations:[{state:"surprise",prior_confidence:0.4,evidence:"moderate",caveats:"Also conversational emphasis."}]},
  {id:"face_brow_lower",label:"Brow lower (AU4)",inference_reliability:2,interpretations:[{state:"concentration / cognitive_effort",prior_confidence:0.35,evidence:"moderate",caveats:"Often mistaken for anger."}]},
  {id:"face_mouth_frown",label:"Lip-corner depressor (AU15)",inference_reliability:3,interpretations:[{state:"sadness / displeasure",prior_confidence:0.45,evidence:"moderate",caveats:"Can be momentary; combine with brow/voice."}]},
  {id:"face_eye_widen",label:"Upper-lid raiser / eye widen (AU5)",inference_reliability:2,interpretations:[{state:"fear / heightened_alertness",prior_confidence:0.3,evidence:"weak",caveats:"Also surprise or bright light; ambiguous alone."}]},
  {id:"face_nose_wrinkle",label:"Nose wrinkler / upper-lip raiser (AU9/10)",inference_reliability:3,interpretations:[{state:"disgust",prior_confidence:0.5,evidence:"moderate",caveats:"Disgust is among the more recognizable, but can signal dislike generally."}]},
  {id:"face_lip_press",label:"Lip press / stretch (AU23/24/20)",inference_reliability:2,interpretations:[{state:"tension / suppressed_negative",prior_confidence:0.3,evidence:"weak",caveats:"Effort or concentration too; not a lie cue."}]},
  {id:"face_jaw_drop",label:"Jaw drop (AU26)",inference_reliability:2,interpretations:[{state:"surprise",prior_confidence:0.4,evidence:"moderate",caveats:"Strongest when co-occurring with brow raise (AU1+2)."}]},
  {id:"face_lip_pucker",label:"Lip pucker / funnel (AU18/22)",inference_reliability:2,interpretations:[{state:"disapproval / skepticism (pursed)",prior_confidence:0.3,evidence:"weak",caveats:"Also speech shape or thinking."}]},
  {id:"face_lip_suck",label:"Lip suck / bite (AU28)",inference_reliability:2,interpretations:[{state:"concentration / anxiety (lip bite)",prior_confidence:0.3,evidence:"weak",caveats:"Habit for many; not a lie cue."}]},
  {id:"face_chin_raise",label:"Chin raiser (AU17)",inference_reliability:2,interpretations:[{state:"doubt / displeasure (pout)",prior_confidence:0.3,evidence:"weak",caveats:"Part of sadness/pout; weak alone."}]},
  {id:"face_smirk",label:"Asymmetric smile / dimpler (AU12+14 unilateral)",inference_reliability:3,interpretations:[{state:"contempt / smugness",prior_confidence:0.4,evidence:"moderate",caveats:"Unilateral lip action is the classic contempt marker, but also irony/playfulness."}]},
  {id:"face_jaw_jut",label:"Jaw thrust (AU29)",inference_reliability:2,interpretations:[{state:"tension / defiance",prior_confidence:0.3,evidence:"weak",caveats:"Also bite alignment/habit."}]},
  {id:"face_cheek_puff",label:"Cheek puff",inference_reliability:2,interpretations:[{state:"exasperation / relief (sigh)",prior_confidence:0.3,evidence:"weak",caveats:"Also blowing/exertion."}]},
  {id:"face_blink_rate",label:"Elevated blink rate (AU45)",inference_reliability:2,interpretations:[{state:"elevated_arousal / stress",prior_confidence:0.3,evidence:"weak",caveats:"Dry eyes, screens, contacts raise blink too. Very low rate = concentration."}]},
  {id:"physio_arousal",label:"Physiological arousal (rPPG heart rate)",inference_reliability:3,interpretations:[{state:"elevated_arousal / stress",prior_confidence:0.4,evidence:"moderate",caveats:"Physiological, harder to fake — but webcam rPPG is noisy (motion/light/exertion). Arousal != valence."}]},
  {id:"emotion_happiness",label:"Happiness prototype (AU6+12)",inference_reliability:4,interpretations:[{state:"happiness",prior_confidence:0.7,evidence:"moderate",prototype:true,caveats:"Constellation is stronger than single AUs, but can still be posed."}]},
  {id:"emotion_sadness",label:"Sadness prototype (AU1+4+15+17)",inference_reliability:3,interpretations:[{state:"sadness",prior_confidence:0.6,evidence:"moderate",prototype:true,caveats:"Brief displays; corroborate with voice/posture."}]},
  {id:"emotion_surprise",label:"Surprise prototype (AU1+2+5+26)",inference_reliability:3,interpretations:[{state:"surprise",prior_confidence:0.6,evidence:"moderate",prototype:true,caveats:"Very brief; can blend into fear."}]},
  {id:"emotion_fear",label:"Fear prototype (AU1+2+5+20)",inference_reliability:3,interpretations:[{state:"fear",prior_confidence:0.55,evidence:"moderate",prototype:true,caveats:"Overlaps surprise; hardest basic emotion to separate."}]},
  {id:"emotion_anger",label:"Anger prototype (AU4+5+7+23)",inference_reliability:3,interpretations:[{state:"anger",prior_confidence:0.55,evidence:"moderate",prototype:true,caveats:"Also intense concentration; need context."}]},
  {id:"emotion_disgust",label:"Disgust prototype (AU9+10+16)",inference_reliability:3,interpretations:[{state:"disgust",prior_confidence:0.6,evidence:"moderate",prototype:true,caveats:"Among the more recognizable; overlaps contempt."}]},
  {id:"emotion_contempt",label:"Contempt prototype (AU12+14 unilateral)",inference_reliability:3,interpretations:[{state:"contempt",prior_confidence:0.5,evidence:"moderate",prototype:true,caveats:"Defined by one-sidedness; cross-cultural validity debated."}]},
  {id:"gaze_aversion",label:"Gaze aversion",inference_reliability:2,interpretations:[{state:"cognitive_load / thinking",prior_confidence:0.3,evidence:"moderate",caveats:"Normal during recall; not evasive; culturally varying."}]},
  {id:"regulator_head_nod",label:"Head nod",inference_reliability:3,interpretations:[{state:"agreement / backchannel",prior_confidence:0.5,evidence:"moderate",caveats:"May be polite continuer, not true agreement."}]},
  {id:"posture_lean_forward",label:"Forward lean",inference_reliability:3,interpretations:[{state:"engagement / interest",prior_confidence:0.45,evidence:"moderate",caveats:"Also hearing difficulty/aggression."}]},
  {id:"posture_closed_arms",label:"Arms crossed",inference_reliability:1,interpretations:[{state:"defensiveness / discomfort",prior_confidence:0.2,evidence:"weak",caveats:"MYTH risk — often just cold or habitual."}]},
  {id:"adaptor_self_touch_face",label:"Hand-to-face self-touch",inference_reliability:1,interpretations:[{state:"anxiety / arousal",prior_confidence:0.2,evidence:"weak",caveats:"Rises with general arousal, NOT specifically lying (DePaulo 2003)."}]},
  {id:"posture_lean_back",label:"Backward lean",inference_reliability:2,interpretations:[{state:"disengagement / skepticism",prior_confidence:0.3,evidence:"weak",caveats:"Also relaxation or comfort. Context decides."}]},
  {id:"posture_ventral_denial",label:"Ventral denial (torso turned away)",inference_reliability:3,interpretations:[{state:"discomfort / withdrawal",prior_confidence:0.4,evidence:"moderate",caveats:"Navarro: we orient our front toward what we favor. Also caused by shifting or addressing someone else."}]},
  {id:"posture_shrug",label:"Shoulder shrug",inference_reliability:3,interpretations:[{state:"uncertainty / I-don't-know",prior_confidence:0.5,evidence:"moderate",caveats:"Recognizable emblem, but can be casual habit."}]},
  {id:"posture_hands_on_hips",label:"Hands on hips (akimbo)",inference_reliability:2,interpretations:[{state:"assertiveness / readiness",prior_confidence:0.35,evidence:"weak",caveats:"Also impatience or simple rest posture."}]},
  {id:"posture_expansive",label:"Expansive / open posture",inference_reliability:2,interpretations:[{state:"confidence / dominance",prior_confidence:0.35,evidence:"weak",caveats:"'Power pose' effects are contested/poorly replicated. Treat cautiously."}]},
  {id:"regulator_head_tilt",label:"Head tilt",inference_reliability:2,interpretations:[{state:"interest / active_listening",prior_confidence:0.3,evidence:"weak",caveats:"Also curiosity or evaluation; mild signal."}]},
  {id:"adaptor_hand_to_neck",label:"Hand to neck/nape",inference_reliability:2,interpretations:[{state:"self_soothing / stress",prior_confidence:0.3,evidence:"weak",caveats:"Comfort gesture; rises with arousal generally, not deception."}]},
  {id:"body_fidget",label:"Fidget / restless hand motion",inference_reliability:2,interpretations:[{state:"restlessness / heightened_arousal",prior_confidence:0.3,evidence:"weak",caveats:"Boredom, energy, or habit too. Not a lie cue."}]},
  {id:"gesture_open_palm",label:"Open palm(s)",inference_reliability:2,interpretations:[{state:"openness / candor",prior_confidence:0.3,evidence:"weak",caveats:"Cultural/illustrative; weak standalone."}]},
  {id:"gesture_hands_together",label:"Hands together / steepled",inference_reliability:2,interpretations:[{state:"contemplation / anxiety (ambiguous)",prior_confidence:0.25,evidence:"weak",caveats:"Steepling=confidence vs clasping=tension — needs finer detail."}]},
  {id:"gesture_pointing",label:"Pointing / index extension",inference_reliability:3,interpretations:[{state:"emphasis / assertion",prior_confidence:0.4,evidence:"moderate",caveats:"Illustrator; can read as aggressive in some cultures."}]}
], constructs: {
  engagement:{label:"Engagement / Interest",positive:{posture_lean_forward:1.0,regulator_head_nod:0.8,regulator_head_tilt:0.6,gesture_open_palm:0.5,face_brow_raise:0.4,prosody_pitch_rise:0.5},negative:{posture_lean_back:0.8,gaze_aversion:0.6,posture_closed_arms:0.4,body_fidget:0.3,posture_ventral_denial:0.6},reliability:4,min_cues:2,evidence:"Thin-slice behavior predicts engagement at r≈.39 (Ambady & Rosenthal 1992); ventral fronting signals comfort (Navarro 2008)"},
  discomfort_anxiety:{label:"Discomfort / Anxiety",positive:{adaptor_self_touch_face:0.9,adaptor_hand_to_neck:0.85,body_fidget:0.8,physio_arousal:0.7,posture_ventral_denial:0.6,face_lip_press:0.5,face_lip_suck:0.5,face_cheek_puff:0.4,gaze_aversion:0.5,posture_closed_arms:0.5,face_blink_rate:0.5,posture_shrug:0.3},negative:{face_smile_genuine:0.6,posture_expansive:0.5},reliability:4,min_cues:2,evidence:"Self-directed displacement / pacifying behaviours reliably read as stress (Navarro 2008); salient cues r≈.50"},
  confidence_dominance:{label:"Confidence / Dominance",positive:{posture_expansive:1.0,posture_hands_on_hips:0.9,gesture_pointing:0.5,prosody_pitch_rise:0.3,regulator_head_nod:0.3},negative:{posture_shrug:0.7,adaptor_self_touch_face:0.5,gaze_aversion:0.5,posture_closed_arms:0.4},reliability:3,min_cues:2,evidence:"Dominance display is a multi-cue constellation: expansiveness + hands-on-hips + head tilt + no smile (Witkower & Tracy 2019)"},
  disengagement:{label:"Disengagement / Withdrawal",positive:{posture_lean_back:1.0,posture_ventral_denial:0.9,gaze_aversion:0.7,posture_closed_arms:0.6,body_fidget:0.4,posture_shrug:0.3},negative:{posture_lean_forward:0.8,regulator_head_nod:0.5,face_smile_genuine:0.4},reliability:3,min_cues:2,evidence:"Contractive posture + gaze aversion signal low involvement; ventral denial is Navarro's (2008) key withdrawal cue"},
  rapport_openness:{label:"Rapport / Openness",positive:{face_smile_genuine:0.9,regulator_head_nod:0.7,gesture_open_palm:0.7,posture_lean_forward:0.6,regulator_head_tilt:0.4},negative:{posture_closed_arms:0.6,gaze_aversion:0.5,face_lip_press:0.3,posture_ventral_denial:0.5},reliability:3,min_cues:2,evidence:"Rapport judged from expressivity/attention/positivity/coordination; low rapport detectable at precision 0.7 (Grahe & Bernieri 1999)"}
}};

// ---------- wire up ----------
$("start").addEventListener("click", async () => {
  $("start").disabled = true;
  try { await loadKB(); await init(); await startCamera(); }
  catch (e) {
    console.error(e);
    showStageMessage(e instanceof CameraError ? e.message : ("Startup error: " + (e.message || e)), true);
    $("start").disabled = false;
  }
});
$("calib").addEventListener("click", startCalibration);
$("saveProf").addEventListener("click", saveProfile);
$("loadProf").addEventListener("click", loadProfile);
$("facs").addEventListener("change", async (e) => {
  if (e.target.checked) {
    try {
      const r = await fetch(BACKEND_URL + "/health", { signal: AbortSignal.timeout(2000) });
      if (!(await r.json()).ok) throw new Error("not ok");
      backendOn = true; $("backendStat").textContent = "FACS: on";
    } catch (err) {
      e.target.checked = false; backendOn = false;
      $("backendStat").textContent = "FACS: unreachable — start backend on :8001";
    }
  } else { backendOn = false; backendEmotions = null; $("backendStat").textContent = "FACS: off"; }
});
