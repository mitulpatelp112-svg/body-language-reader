// Practice/session controller: scenario selection, prompts, recording, nudges, report modal.
import { session } from "./session.js";
import { SCENARIOS, scenarioById } from "./scenarios.js";
import { lineChart } from "./charts.js";
import { speech } from "./speech.js";

const $ = (id) => document.getElementById(id);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

let promptIdx = 0;

// ---- populate scenarios ----
const sel = $("scenario");
SCENARIOS.forEach((s) => { const o = document.createElement("option"); o.value = s.id; o.textContent = s.label; sel.appendChild(o); });
function syncBlurb() { const sc = scenarioById(sel.value); $("scenarioBlurb").textContent = sc.blurb; }
sel.addEventListener("change", syncBlurb); syncBlurb();

function setPrompt(i) {
  const sc = scenarioById(sel.value);
  if (!sc.prompts || !sc.prompts.length) { $("promptBox").hidden = true; return; }
  promptIdx = (i + sc.prompts.length) % sc.prompts.length;
  $("promptText").textContent = sc.prompts[promptIdx];
  $("promptBox").hidden = false;
}
$("nextPrompt").addEventListener("click", () => setPrompt(promptIdx + 1));

// ---- toggles ----
$("speechToggle").addEventListener("change", (e) => {
  $("speechNote").hidden = !e.target.checked;
  if (e.target.checked && !speech.supported) showNudge("Speech recognition is not available in this browser. The rest still works.");
});
$("nudgeToggle").addEventListener("change", (e) => { session.nudgesEnabled = e.target.checked; });

// ---- nudge toast ----
let nudgeT = null;
function showNudge(text) {
  const el = $("nudge"); el.textContent = text; el.hidden = false;
  clearTimeout(nudgeT); nudgeT = setTimeout(() => { el.hidden = true; }, 6000);
}

// ---- start / stop ----
const btn = $("sessBtn");
btn.addEventListener("click", async () => {
  if (!session.recording) {
    if (!window.PRESENCE || !window.PRESENCE.running) { showNudge("Start the camera first, then record a session."); return; }
    session.onTick = (t) => { $("sessTimer").textContent = mmss(t); };
    session.onNudge = showNudge;
    session.nudgesEnabled = $("nudgeToggle").checked;
    session.start({ scenario: sel.value, useSpeech: $("speechToggle").checked });
    btn.classList.add("rec"); btn.innerHTML = '<i class="ph ph-stop-fill"></i> Stop session';
    $("sessTimer").hidden = false; $("sessTimer").textContent = "0:00";
    setPrompt(0);
  } else {
    btn.disabled = true;
    const report = await session.stop();
    btn.disabled = false; btn.classList.remove("rec"); btn.innerHTML = '<i class="ph ph-record-fill"></i> Start session';
    $("sessTimer").hidden = true; $("promptBox").hidden = true;
    renderReport(report);
  }
});

// ---- report modal ----
function pct(x) { return Math.round((x || 0) * 100); }
function renderReport(r) {
  const acc = cssVar("--accent") || "#45d3ad", bad = cssVar("--bad") || "#f0625f";
  const sp = r.speech || {};
  const stats = [
    ["Duration", mmss(r.durationSec)],
    ["Engagement", pct(r.avg.eng) + "%"],
    ["Composure", pct(1 - Math.max(r.avg.disc, r.avg.dis)) + "%"],
    ["Talk time", mmss(r.talkSec)],
    ...(sp.used ? [["Filler words", String(sp.fillerTotal)], ["Pace", (sp.wpm || 0) + " wpm"]] : []),
  ];
  $("reportBody").innerHTML = `
    <div class="caveat">${r.scenario ? scenarioById(r.scenario).label + " · " : ""}${new Date(r.startedAt).toLocaleString()}</div>
    <div class="rstats">${stats.map(([l, v]) => `<div class="rstat"><b>${v}</b><span>${l}</span></div>`).join("")}</div>
    <div class="legend-row"><span><i style="background:${acc}"></i>Engagement</span><span><i style="background:${bad}"></i>Discomfort</span></div>
    <div class="chartwrap"><canvas class="chart" id="reportChart"></canvas></div>
    <div class="wgrid">
      <div class="wbox"><h3>What went well</h3><ul>${r.wins.map((w) => `<li>${w}</li>`).join("")}</ul></div>
      <div class="wbox"><h3>Work on</h3><ul>${r.workOn.map((w) => `<li>${w}</li>`).join("")}</ul></div>
    </div>
    ${sp.used && sp.fillerTotal ? `<div class="caveat" style="margin-top:10px">Fillers: ${Object.entries(sp.fillerByType).map(([k, v]) => `"${k}" ×${v}`).join(", ")}</div>` : ""}`;
  $("reportModal").hidden = false;
  const S = r.samples || [];
  requestAnimationFrame(() => lineChart($("reportChart"), [
    { color: acc, data: S.map((p) => ({ x: p.t, y: p.eng })) },
    { color: bad, data: S.map((p) => ({ x: p.t, y: Math.max(p.dis, p.disc) })) },
  ]));
}
$("reportClose").addEventListener("click", () => { $("reportModal").hidden = true; });
$("reportDone").addEventListener("click", () => { $("reportModal").hidden = true; });
$("reportProgress").addEventListener("click", () => { location.href = "./progress.html"; });
