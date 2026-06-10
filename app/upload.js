// Upload analyser: sends image/video to the py-feat backend and renders calibrated emotion + AU + VAD.
// Backend: backend/server.py — endpoints /analyze_image and /analyze_video.

const BACKEND = localStorage.getItem("presence-backend") || "http://localhost:8001";
document.getElementById("ep-host").textContent = BACKEND;

const EMOTION_COLOURS = {
  happiness: "#45d3ad", sadness: "#5b8def", anger: "#f0625f", surprise: "#d8a23a",
  fear: "#a974f0", disgust: "#7fc66a", neutral: "#94a1b2",
};

const els = {
  dropImg: document.getElementById("drop-img"),
  dropVid: document.getElementById("drop-vid"),
  fileImg: document.getElementById("file-img"),
  fileVid: document.getElementById("file-vid"),
  prevImg: document.getElementById("prev-img"),
  prevVid: document.getElementById("prev-vid"),
  imgEl: document.getElementById("img-el"),
  vidEl: document.getElementById("vid-el"),
  go: document.getElementById("go"),
  clear: document.getElementById("clear"),
  fps: document.getElementById("fps"),
  maxf: document.getElementById("maxf"),
  status: document.getElementById("status"),
  empty: document.getElementById("empty"),
  result: document.getElementById("result"),
  rTop: document.getElementById("r-top"),
  rConf: document.getElementById("r-conf"),
  rKind: document.getElementById("r-kind"),
  rBars: document.getElementById("r-bars"),
  rVal: document.getElementById("r-val"),
  rAro: document.getElementById("r-aro"),
  rAus: document.getElementById("r-aus"),
  rTl: document.getElementById("r-tl"),
  tl: document.getElementById("tl"),
  tlLegend: document.getElementById("tl-legend"),
};

let pendingFile = null;
let pendingKind = null; // "image" | "video"

function setStatus(msg, isErr=false){
  els.status.textContent = msg;
  els.status.classList.toggle("err", !!isErr);
}
function wireDrop(zone, input, kind){
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("over");
    const f = e.dataTransfer.files?.[0]; if (f) accept(f, kind);
  });
  input.addEventListener("change", e => { const f = e.target.files?.[0]; if (f) accept(f, kind); });
}
function accept(file, kind){
  const expected = kind === "image" ? "image/" : "video/";
  if (!file.type.startsWith(expected)) { setStatus(`Expected ${kind}, got ${file.type}`, true); return; }
  pendingFile = file; pendingKind = kind;
  if (kind === "image"){
    els.imgEl.src = URL.createObjectURL(file); els.prevImg.classList.add("show");
    els.prevVid.classList.remove("show"); els.vidEl.removeAttribute("src");
  } else {
    els.vidEl.src = URL.createObjectURL(file); els.prevVid.classList.add("show");
    els.prevImg.classList.remove("show"); els.imgEl.removeAttribute("src");
  }
  els.go.disabled = false;
  setStatus(`Ready: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`);
}
function clearAll(){
  pendingFile = null; pendingKind = null;
  els.imgEl.removeAttribute("src"); els.vidEl.removeAttribute("src");
  els.prevImg.classList.remove("show"); els.prevVid.classList.remove("show");
  els.fileImg.value = ""; els.fileVid.value = "";
  els.go.disabled = true;
  els.empty.style.display = ""; els.result.style.display = "none";
  setStatus("Cleared.");
}
wireDrop(els.dropImg, els.fileImg, "image");
wireDrop(els.dropVid, els.fileVid, "video");
els.clear.addEventListener("click", clearAll);

async function analyse(){
  if (!pendingFile) return;
  els.go.disabled = true; setStatus("Analysing… (first call downloads ~200MB of model weights)");
  const fd = new FormData(); fd.append("file", pendingFile);
  let url;
  if (pendingKind === "image"){
    const model = document.getElementById("model-sel")?.value || "cnn";
    url = `${BACKEND}/${model === "cnn" ? "analyze_cnn" : "analyze_image"}`;
  } else {
    const fps = Math.max(0.5, Math.min(10, parseFloat(els.fps.value) || 2));
    const maxf = Math.max(5, Math.min(600, parseInt(els.maxf.value) || 120));
    url = `${BACKEND}/analyze_video?fps=${fps}&max_frames=${maxf}`;
  }
  const t0 = performance.now();
  try {
    const r = await fetch(url, { method: "POST", body: fd });
    const data = await r.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    if (!data.ok){ setStatus(`Backend returned: ${data.reason || data.error || "error"} (${dt}s)`, true); return; }
    setStatus(`Done in ${dt}s.`);
    render(data, pendingKind);
  } catch (e){
    setStatus(`Couldn't reach ${url} — is the backend running? (${e.message})`, true);
  } finally {
    els.go.disabled = false;
  }
}
els.go.addEventListener("click", analyse);

function render(data, kind){
  els.empty.style.display = "none"; els.result.style.display = "";
  els.rKind.textContent = kind;

  const payload = kind === "image" ? data : (data.aggregate || {});
  const emotions = payload.emotions || {};
  els.rTop.textContent = payload.top || "—";
  els.rTop.style.color = EMOTION_COLOURS[payload.top] || "var(--txt)";
  els.rConf.textContent = payload.confidence != null ? `${Math.round(payload.confidence*100)}% confident` : "—";
  els.rVal.textContent = payload.valence != null ? payload.valence.toFixed(2) : "—";
  els.rAro.textContent = payload.arousal != null ? payload.arousal.toFixed(2) : "—";

  // emotion bars
  els.rBars.innerHTML = "";
  const sorted = Object.entries(emotions).sort((a,b)=>b[1]-a[1]);
  for (const [name, p] of sorted){
    const row = document.createElement("div"); row.className = "bar";
    row.innerHTML = `<span class="nm">${name}</span>
      <span class="tr"><span class="fl" style="width:${Math.round(p*100)}%;background:${EMOTION_COLOURS[name]||'var(--accent)'}"></span></span>
      <span class="pct">${Math.round(p*100)}%</span>`;
    els.rBars.appendChild(row);
  }

  // AUs — show the image's AUs, or an averaged set for video
  let aus = data.aus;
  if (!aus && kind === "video" && data.frames){
    const sums = {}; let n = 0;
    for (const f of data.frames){ if (f.ok !== false){
      // frames in video response don't include AUs (we stripped them for size); skip
    }}
  }
  els.rAus.innerHTML = "";
  if (aus){
    const sortedA = Object.entries(aus).sort((a,b)=>b[1]-a[1]).slice(0, 16);
    for (const [name, v] of sortedA){
      const row = document.createElement("div"); row.className = "au";
      const pct = Math.max(0, Math.min(1, v));
      row.innerHTML = `<span class="nm">${name}</span>
        <span class="tr"><span class="fl" style="width:${Math.round(pct*100)}%"></span></span>
        <span class="pct">${v.toFixed(2)}</span>`;
      els.rAus.appendChild(row);
    }
  } else {
    els.rAus.innerHTML = '<div class="empty" style="grid-column:1/-1"><i class="ph ph-info"></i>Per-frame AUs are not returned for videos to keep payload small. Upload a still for AUs.</div>';
  }

  // timeline (video only)
  if (kind === "video" && data.frames?.length){
    els.rTl.style.display = "";
    drawTimeline(data.frames);
  } else {
    els.rTl.style.display = "none";
  }
}

function drawTimeline(frames){
  const cv = els.tl;
  const W = cv.clientWidth || 600, H = 160;
  cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  cv.style.height = H + "px";
  const ctx = cv.getContext("2d"); ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0,0,W,H);

  const ok = frames.filter(f => f.ok !== false && f.emotions);
  if (!ok.length){
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--mut") || "#94a1b2";
    ctx.font = "13px -apple-system,system-ui";
    ctx.fillText("No frames with a detected face.", 12, 24);
    return;
  }
  const tMin = ok[0].t, tMax = ok[ok.length-1].t;
  const xAt = t => (tMax === tMin) ? W/2 : (12 + (t - tMin) / (tMax - tMin) * (W - 24));
  const labels = Object.keys(EMOTION_COLOURS);
  const yPad = 12, yH = H - yPad*2;

  // stacked area: cumulative bands per emotion at each frame
  for (let i = 0; i < ok.length - 1; i++){
    const a = ok[i], b = ok[i+1];
    const x1 = xAt(a.t), x2 = xAt(b.t);
    let cumA = 0, cumB = 0;
    for (const lab of labels){
      const va = a.emotions[lab] || 0, vb = b.emotions[lab] || 0;
      const y1a = yPad + (1 - cumA) * yH, y1b = yPad + (1 - (cumA+va)) * yH;
      const y2a = yPad + (1 - cumB) * yH, y2b = yPad + (1 - (cumB+vb)) * yH;
      ctx.fillStyle = EMOTION_COLOURS[lab];
      ctx.beginPath();
      ctx.moveTo(x1, y1a); ctx.lineTo(x2, y2a); ctx.lineTo(x2, y2b); ctx.lineTo(x1, y1b); ctx.closePath();
      ctx.fill();
      cumA += va; cumB += vb;
    }
  }
  // x-axis ticks
  ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.font = "11px ui-monospace,Menlo,monospace";
  const ticks = 5;
  for (let i = 0; i <= ticks; i++){
    const t = tMin + (tMax - tMin) * (i/ticks);
    const x = xAt(t);
    ctx.fillRect(x, H - 4, 1, 4);
    ctx.fillText(`${t.toFixed(1)}s`, Math.min(W-30, x + 2), H - 6);
  }

  // legend
  els.tlLegend.innerHTML = "";
  for (const lab of labels){
    const it = document.createElement("span"); it.className = "it";
    it.innerHTML = `<span class="sw" style="background:${EMOTION_COLOURS[lab]}"></span>${lab}`;
    els.tlLegend.appendChild(it);
  }
}

// ping the backend on load, update CNN status pill
fetch(`${BACKEND}/health`).then(r => r.json()).then(d => {
  if (!d.ok) return;
  const cnnSel = document.querySelector('#model-sel option[value="cnn"]');
  if (cnnSel && !d.cnn_built) {
    cnnSel.disabled = true;
    cnnSel.textContent += " — not built yet, run train_cnn.py";
    document.getElementById("model-sel").value = "facs";
  }
  const bits = [];
  bits.push(d.cnn_built ? "CNN ready" : "CNN missing");
  bits.push(d.model_loaded ? "FACS loaded" : "FACS lazy-load on first call");
  setStatus(`Backend up · ${bits.join(" · ")}.`);
}).catch(() => setStatus("Backend unreachable. Start it with: cd backend && uvicorn server:app --port 8001", true));
