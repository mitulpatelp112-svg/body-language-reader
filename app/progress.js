// Progress page: streak, totals, trend chart, two-take compare, history list.
import { listSessions, deleteSession, saveSession, computeStreak } from "./session.js";
import { lineChart } from "./charts.js";
import { scenarioById } from "./scenarios.js";

const $ = (id) => document.getElementById(id);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (x) => Math.round((x || 0) * 100);
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const mins = (s) => (s >= 60 ? Math.round(s / 60) + "m" : s + "s");

async function render() {
  const sessions = await listSessions();
  $("trendPanel").hidden = true; $("comparePanel").hidden = true;   // reset (re-shown below if >=2)

  // summary
  $("sStreak").textContent = computeStreak(sessions);
  $("sCount").textContent = sessions.length;
  const totalSec = sessions.reduce((a, s) => a + (s.durationSec || 0), 0);
  $("sTime").textContent = totalSec >= 60 ? Math.round(totalSec / 60) + "m" : totalSec + "s";
  const avgEng = sessions.length ? sessions.reduce((a, s) => a + (s.avg?.eng || 0), 0) / sessions.length : 0;
  $("sEng").textContent = pct(avgEng) + "%";

  // trend (needs >=2)
  if (sessions.length >= 2) {
    $("trendPanel").hidden = false;
    const acc = cssVar("--accent"), warn = cssVar("--warn");
    const eng = sessions.map((s, i) => ({ x: i, y: s.avg?.eng || 0 }));
    const maxF = Math.max(1, ...sessions.map((s) => s.speech?.fillerPerMin || 0));
    const fil = sessions.map((s, i) => ({ x: i, y: (s.speech?.fillerPerMin || 0) / maxF }));
    requestAnimationFrame(() => lineChart($("trendChart"), [
      { color: acc, data: eng, dots: true }, { color: warn, data: fil, dots: true },
    ]));
  }

  // compare last two (feature 6)
  if (sessions.length >= 2) {
    $("comparePanel").hidden = false;
    const a = sessions[sessions.length - 2], b = sessions[sessions.length - 1];
    const rows = [
      ["Engagement", pct(a.avg.eng), pct(b.avg.eng), 1],
      ["Composure", pct(1 - Math.max(a.avg.disc, a.avg.dis)), pct(1 - Math.max(b.avg.disc, b.avg.dis)), 1],
      ["Confidence", pct(a.avg.conf), pct(b.avg.conf), 1],
      ["Filler words", a.speech?.fillerTotal || 0, b.speech?.fillerTotal || 0, -1],
      ["Pace (wpm)", a.speech?.wpm || 0, b.speech?.wpm || 0, 0],
    ];
    $("compareBody").innerHTML =
      `<tr><th>Metric</th><th>Take A</th><th>Take B</th><th>Change</th></tr>` +
      rows.map(([label, va, vb, dir]) => {
        const delta = vb - va;
        let cls = "", arrow = "";
        if (dir !== 0 && delta !== 0) { const good = dir > 0 ? delta > 0 : delta < 0; cls = good ? "up" : "down"; arrow = (delta > 0 ? "+" : "") + delta; }
        else if (delta !== 0) arrow = (delta > 0 ? "+" : "") + delta;
        return `<tr><td>${label}</td><td>${va}</td><td>${vb}</td><td class="${cls}">${arrow || "="}</td></tr>`;
      }).join("");
  }

  // history
  if (!sessions.length) {
    $("history").innerHTML = `<div class="empty">No sessions yet. Record one in the <a href="./index.html">demo</a>.</div>`;
    return;
  }
  const rows = [...sessions].reverse().map((s) => `
    <tr>
      <td>${new Date(s.startedAt).toLocaleDateString([], { month: "short", day: "numeric" })}
        <span class="caveat">${s.scenario ? scenarioById(s.scenario).label : ""}</span></td>
      <td>${mmss(s.durationSec)}</td>
      <td>${pct(s.avg.eng)}%</td>
      <td>${s.speech?.used ? s.speech.fillerTotal : "-"}</td>
      <td><button class="delbtn" data-id="${s.id}">Delete</button></td>
    </tr>`).join("");
  $("history").innerHTML = `<table><tr><th>Date</th><th>Length</th><th>Engage</th><th>Fillers</th><th></th></tr>${rows}</table>`;
  $("history").querySelectorAll(".delbtn").forEach((b) =>
    b.addEventListener("click", async () => { await deleteSession(b.dataset.id); render(); }));

  updateGoal(sessions);
}

// ---- Weekly goal ring (new feature) ----
function updateGoal(sessions) {
  const goal = Math.max(1, +(localStorage.getItem("presence-goal") || 3));
  const weekAgo = Date.now() - 7 * 864e5;
  const thisWeek = sessions.filter((s) => s.startedAt >= weekAgo).length;
  const frac = Math.min(1, thisWeek / goal);
  const C = 201;
  $("goalArc").setAttribute("stroke-dashoffset", String(Math.round(C * (1 - frac))));
  $("goalRingText").textContent = `${thisWeek}/${goal}`;
  $("goalInput").value = goal;
  $("goalLabel").textContent = thisWeek >= goal ? "Goal reached this week. Nice work." : `${goal - thisWeek} more to hit your weekly goal.`;
}
$("goalInput").addEventListener("change", (e) => {
  localStorage.setItem("presence-goal", String(Math.max(1, Math.min(21, +e.target.value || 3))));
  render();
});

// ---- Data export / import (new feature; privacy-friendly portability) ----
$("exportBtn").addEventListener("click", async () => {
  const data = {
    app: "presence", exportedAt: new Date().toISOString(),
    sessions: await listSessions(),
    profiles: JSON.parse(localStorage.getItem("presence-profiles") || "{}"),
    goal: localStorage.getItem("presence-goal") || "3",
    model: localStorage.getItem("presence-model") || null,
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: "application/json" }));
  a.download = `presence-backup-${new Date().toISOString().slice(0, 10)}.json`; a.click();
  $("dataStatus").textContent = `exported ${data.sessions.length} sessions`;
});
$("importBtn").addEventListener("click", () => $("importFile").click());
$("importFile").addEventListener("change", async (e) => {
  const file = e.target.files[0]; if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (data.app !== "presence") throw new Error("not a Presence backup");
    for (const s of data.sessions || []) await saveSession(s);
    if (data.profiles) localStorage.setItem("presence-profiles", JSON.stringify(data.profiles));
    if (data.goal) localStorage.setItem("presence-goal", data.goal);
    if (data.model) localStorage.setItem("presence-model", data.model);
    $("dataStatus").textContent = `imported ${(data.sessions || []).length} sessions`;
    render();
  } catch (err) { $("dataStatus").textContent = "import failed: " + err.message; }
  e.target.value = "";
});

render();
