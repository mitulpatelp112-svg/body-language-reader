// Accuracy eval harness — measures the emotion-prototype layer under guided posing.
import { FaceLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { EMOTIONS, scoreEmotions, topPrediction } from "./emotion-core.js";

const $ = (id) => document.getElementById(id);
const video = $("video");
const PREP_MS = 3000, CAP_MS = 2000;
let faceLM, lastT = -1, capturing = false, capCounts = null, capLabel = null;
const trials = [];          // {true, pred}
const dataset = [];         // {label, blendshapes:{...}} — training samples for a future model

async function init() {
  $("status").textContent = "loading model…";
  const fs = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  faceLM = await FaceLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU" },
    runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true });
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 } });
  video.srcObject = stream; await video.play();
  $("status").textContent = "ready";
  requestAnimationFrame(loop);
}
function loop() {
  const t = performance.now();
  if (video.currentTime !== lastT) {
    lastT = video.currentTime;
    const r = faceLM.detectForVideo(video, t);
    const bs = {};
    if (r.faceBlendshapes && r.faceBlendshapes[0])
      for (const c of r.faceBlendshapes[0].categories) bs[c.categoryName] = c.score;
    const pred = topPrediction(bs);
    $("livepred").textContent = pred;
    if (capturing && capCounts) {
      capCounts[pred] = (capCounts[pred]||0) + 1;
      if (Object.keys(bs).length) dataset.push({ label: capLabel, ...bs });  // log feature vector + label
    }
  }
  requestAnimationFrame(loop);
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
function showCue(html) { $("cue").innerHTML = html; }

const REPS = 2;   // trials per emotion (more = more reliable accuracy estimate)
function shuffled(arr) { const a = arr.slice(); for (let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; } return a; }

async function runEval() {
  $("start").disabled = true; $("dl").disabled = true; $("dlData").disabled = true;
  trials.length = 0; dataset.length = 0;
  const sequence = shuffled(EMOTIONS.flatMap(e => Array(REPS).fill(e)));  // randomized, repeated
  for (const emo of sequence) {
    // prep countdown
    for (let s = Math.ceil(PREP_MS/1000); s > 0; s--) {
      showCue(`<div class="sub">Get ready to show</div><div class="big">${emo}</div><div class="count">${s}</div>`);
      await sleep(1000);
    }
    // capture window
    capCounts = {}; capLabel = emo; capturing = true;
    showCue(`<div class="sub">Hold it…</div><div class="big">${emo}</div><div class="count">●</div>`);
    await sleep(CAP_MS);
    capturing = false;
    // modal prediction over the window
    let pred = "neutral", n = -1;
    for (const k in capCounts) if (capCounts[k] > n) { n = capCounts[k]; pred = k; }
    trials.push({ true: emo, pred });
    showCue(`<div class="big">${emo}</div><div class="sub">recorded: <b>${pred}</b> ${pred===emo?"✓":"✗"}</div>`);
    await sleep(800);
  }
  showCue(`<div class="big">Done</div><div class="sub">See results →</div>`);
  computeResults();
  $("start").disabled = false; $("dl").disabled = false; $("dlData").disabled = false;
  $("status").textContent = `complete · ${dataset.length} samples logged`;
}

function computeResults() {
  const labels = [...EMOTIONS, "neutral"];
  const idx = Object.fromEntries(labels.map((l,i)=>[l,i]));
  const M = labels.map(()=>labels.map(()=>0));
  let correct = 0;
  for (const t of trials) { M[idx[t.true]][idx[t.pred]]++; if (t.true===t.pred) correct++; }
  const acc = trials.length ? correct/trials.length : 0;

  let html = `<div class="big-acc">${(acc*100).toFixed(0)}%</div>
    <div class="mut">overall accuracy · ${correct}/${trials.length} trials</div>
    <table style="margin-top:10px"><tr><th>true ＼ pred</th>${labels.map(l=>`<th>${l.slice(0,4)}</th>`).join("")}</tr>`;
  labels.forEach((row,r) => {
    if (!EMOTIONS.includes(row)) return; // only show the 7 prompted rows
    html += `<tr><th>${row}</th>` + labels.map((_,c)=>{
      const v = M[r][c]; const cls = r===c ? "diag" : (v>0?"hit":"");
      return `<td class="${cls}">${v||""}</td>`;
    }).join("") + "</tr>";
  });
  html += `</table>`;

  // per-class precision / recall / F1 + macro
  const P=[], R=[], F=[];
  html += `<h2 style="margin-top:14px">Per-class P / R / F1</h2>
    <table><tr><th>emotion</th><th>prec</th><th>rec</th><th>F1</th></tr>`;
  EMOTIONS.forEach(emo => {
    const r = idx[emo];
    const tp = M[r][r];
    const predCol = labels.reduce((s,_,c)=>s+M[c][r], 0);   // total predicted as emo
    const trueRow = labels.reduce((s,_,c)=>s+M[r][c], 0);   // total truly emo
    const prec = predCol ? tp/predCol : 0, rec = trueRow ? tp/trueRow : 0;
    const f1 = (prec+rec) ? 2*prec*rec/(prec+rec) : 0;
    P.push(prec); R.push(rec); F.push(f1);
    html += `<tr><th>${emo}</th><td>${(prec*100).toFixed(0)}</td><td>${(rec*100).toFixed(0)}</td><td>${(f1*100).toFixed(0)}</td></tr>`;
  });
  const avg = a => a.reduce((x,y)=>x+y,0)/a.length;
  html += `<tr><th>macro</th><td>${(avg(P)*100).toFixed(0)}</td><td>${(avg(R)*100).toFixed(0)}</td><td>${(avg(F)*100).toFixed(0)}</td></tr></table>`;
  html += `<p class="mut" style="margin-top:8px">Diagonal = correct; off-diagonal = confusions
    (watch fear↔surprise, anger↔disgust). Macro-F1 is your headline number. Posing ≠ felt emotion.</p>`;
  $("result").innerHTML = html;
}

function save(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
}
function downloadCSV() {
  const rows = [["true","pred","correct"], ...trials.map(t=>[t.true,t.pred,t.true===t.pred?1:0])];
  save("eval_results.csv", rows.map(r=>r.join(",")).join("\n"), "text/csv");
}
// Training dataset: every captured frame's blendshape vector + its true label, as JSONL.
// This is the (features -> label) corpus for training your own model later.
function downloadDataset() {
  if (!dataset.length) return;
  save("blendshape_dataset.jsonl", dataset.map(r=>JSON.stringify(r)).join("\n"), "application/x-ndjson");
}

$("start").addEventListener("click", async () => {
  if (!faceLM) { try { await init(); } catch(e){ $("status").textContent="error: "+e.message; return; } }
  runEval();
});
$("dl").addEventListener("click", downloadCSV);
$("dlData").addEventListener("click", downloadDataset);
