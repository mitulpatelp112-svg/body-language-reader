// Session recorder + report + history (IndexedDB) + real-time nudges for Presence.
import { speech } from "./speech.js";

const CONSTRUCTS = {
  eng:  "Engagement / Interest",
  rap:  "Rapport / Openness",
  conf: "Confidence / Dominance",
  dis:  "Disengagement / Withdrawal",
  disc: "Discomfort / Anxiety",
};
const pOf = (states, label) => { const s = (states || []).find((x) => x.state === label); return s ? s.p : 0; };
const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);

// ---------- IndexedDB ----------
function db() {
  return new Promise((res, rej) => {
    const r = indexedDB.open("presence", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("sessions", { keyPath: "id" });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function dbPut(rec) { const d = await db(); return new Promise((res, rej) => { const t = d.transaction("sessions", "readwrite"); t.objectStore("sessions").put(rec); t.oncomplete = res; t.onerror = () => rej(t.error); }); }
export async function listSessions() { const d = await db(); return new Promise((res) => { const out = []; d.transaction("sessions").objectStore("sessions").openCursor().onsuccess = (e) => { const c = e.target.result; if (c) { out.push(c.value); c.continue(); } else res(out.sort((a, b) => a.startedAt - b.startedAt)); }; }); }
export async function getSession(id) { const d = await db(); return new Promise((res) => { d.transaction("sessions").objectStore("sessions").get(id).onsuccess = (e) => res(e.target.result); }); }
export async function saveSession(rec) { return dbPut(rec); }
export async function deleteSession(id) { const d = await db(); return new Promise((res) => { const t = d.transaction("sessions", "readwrite"); t.objectStore("sessions").delete(id); t.oncomplete = res; }); }

// ---------- audio chime for nudges ----------
let actx = null;
function chime() {
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "sine"; o.frequency.value = 660;
    g.gain.setValueAtTime(0.0001, actx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.12, actx.currentTime + 0.04);
    g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + 0.5);
    o.connect(g).connect(actx.destination); o.start(); o.stop(actx.currentTime + 0.5);
  } catch (e) {}
}

// ---------- recorder ----------
export const session = {
  recording: false,
  scenario: null,
  onTick: null,     // (elapsedSec, last) => {}  for live UI (timer, nudge banner)
  onNudge: null,    // (text) => {}
  nudgesEnabled: true,
  _samples: [], _start: 0, _talk: 0, _timer: null,
  _negStreak: 0, _lastNudge: 0,

  start({ scenario = null, useSpeech = false } = {}) {
    this.recording = true; this.scenario = scenario;
    this._samples = []; this._start = Date.now(); this._talk = 0;
    this._negStreak = 0; this._lastNudge = -99;   // allow the first nudge as soon as the streak builds
    this._useSpeech = useSpeech && speech.supported;
    if (this._useSpeech) speech.start();
    this._timer = setInterval(() => this._tick(), 1000);
    return true;
  },

  _tick() {
    const P = window.PRESENCE || {};
    const t = (Date.now() - this._start) / 1000;
    if (P.voiced) this._talk += 1;
    const s = {
      t: +t.toFixed(0),
      v: +(P.v || 0).toFixed(3), a: +(P.a || 0).toFixed(3), d: +(P.d || 0).toFixed(3),
      eng: +pOf(P.states, CONSTRUCTS.eng).toFixed(3),
      rap: +pOf(P.states, CONSTRUCTS.rap).toFixed(3),
      conf: +pOf(P.states, CONSTRUCTS.conf).toFixed(3),
      dis: +pOf(P.states, CONSTRUCTS.dis).toFixed(3),
      disc: +pOf(P.states, CONSTRUCTS.disc).toFixed(3),
      hr: P.hrQuality > 0.3 ? Math.round(P.hr) : 0,
    };
    this._samples.push(s);

    // real-time nudge: a negative construct sustained -> gentle chime + cue
    const neg = Math.max(s.dis, s.disc);
    if (neg > 0.5) this._negStreak += 1; else this._negStreak = 0;
    if (this.nudgesEnabled && this._negStreak >= 12 && t - this._lastNudge > 25) {
      this._lastNudge = t; this._negStreak = 0;
      const text = s.disc >= s.dis ? "You've tensed up for a bit. Drop your shoulders and steady your hands."
                                   : "You've drifted closed-off. Square up, lean in, re-engage.";
      chime(); if (this.onNudge) this.onNudge(text);
    }
    if (this.onTick) this.onTick(t, s);
  },

  async stop() {
    this.recording = false;
    clearInterval(this._timer); this._timer = null;
    speech.stop();
    const report = this._summarize();
    try { await dbPut(report); } catch (e) {}
    return report;
  },

  _summarize() {
    const S = this._samples;
    const m = speech.metrics(this._talk); m.used = !!this._useSpeech;
    const A = {
      eng: avg(S.map((x) => x.eng)), rap: avg(S.map((x) => x.rap)),
      conf: avg(S.map((x) => x.conf)), dis: avg(S.map((x) => x.dis)),
      disc: avg(S.map((x) => x.disc)),
      v: avg(S.map((x) => x.v)), a: avg(S.map((x) => x.a)), d: avg(S.map((x) => x.d)),
    };
    const durationSec = Math.max(1, Math.round((Date.now() - this._start) / 1000));
    const wins = [], workOn = [];
    if (A.eng > 0.45) wins.push("Strong engagement throughout.");
    if (A.rap > 0.4) wins.push("Warm, open rapport.");
    if (A.disc < 0.25 && A.dis < 0.25) wins.push("Calm and composed.");
    if (A.conf > 0.4) wins.push("Confident, grounded presence.");
    if (A.disc > 0.35) workOn.push("Ease the tension: slower pace, looser shoulders.");
    if (A.dis > 0.35) workOn.push("Stay open and face forward; you drifted away.");
    if (A.eng < 0.25 && A.rap < 0.25) workOn.push("Bring more energy: lean in, nod, hold eye contact.");
    if (m.used) {
      if (m.fillerPerMin > 4) workOn.push(`Cut filler words (${m.fillerTotal} total, ${m.fillerPerMin}/min).`);
      if (m.wpm > 170) workOn.push(`Slow your pace (~${m.wpm} wpm).`);
      else if (m.wpm > 0 && m.wpm < 105) workOn.push(`Pick up the pace a little (~${m.wpm} wpm).`);
    }
    if (!wins.length) wins.push("Session recorded. Keep practicing to see trends.");
    if (!workOn.length) workOn.push("Nothing major to flag. Solid take.");

    return {
      id: "s_" + this._start, startedAt: this._start, date: new Date(this._start).toISOString(),
      durationSec, scenario: this.scenario, talkSec: this._talk,
      samples: S, avg: A, speech: m,
      wins: wins.slice(0, 3), workOn: workOn.slice(0, 3),
    };
  },
};

// ---------- streak (consecutive days with >=1 session) ----------
export function computeStreak(sessions) {
  if (!sessions.length) return 0;
  const days = new Set(sessions.map((s) => new Date(s.startedAt).toDateString()));
  let streak = 0; const d = new Date();
  // allow today OR yesterday as the anchor so a streak isn't lost before today's session
  if (!days.has(d.toDateString())) d.setDate(d.getDate() - 1);
  while (days.has(d.toDateString())) { streak++; d.setDate(d.getDate() - 1); }
  return streak;
}
