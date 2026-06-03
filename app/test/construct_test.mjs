// Verifies the corroboration fix: evidence-backed constructs become CONFIDENT + corroborated
// when a constellation fires, but a single cue stays honestly hedged.
// Replicates app.js computeConstructs against the real knowledge-base/signals.json.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const KB = JSON.parse(readFileSync(join(root, "knowledge-base", "signals.json"), "utf8"));
const clip = (x) => Math.max(0, Math.min(1, x));

function computeConstructs(act) {
  const out = [];
  for (const [key, c] of Object.entries(KB.constructs)) {
    if (key.startsWith("_")) continue;
    let pos = 0, neg = 0; const active = [];
    for (const [sig, w] of Object.entries(c.positive || {})) {
      const a = act[sig] || 0; if (a > 0.12) { pos += w * a; active.push(sig); }
    }
    for (const [sig, w] of Object.entries(c.negative || {})) neg += w * (act[sig] || 0);
    const nCues = active.length; if (!nCues) continue;
    const corrob = Math.min(1, nCues / (c.min_cues || 2));
    const conf = Math.min(0.92, clip(pos * 0.7 * (0.55 + 0.45 * corrob) - 0.5 * neg));
    out.push({ state: c.label, p: conf, nCues, corroborated: nCues >= (c.min_cues || 2) });
  }
  return out.sort((a, b) => b.p - a.p);
}
const top = (act) => computeConstructs(act)[0] || { state: "none", p: 0, nCues: 0, corroborated: false };

let pass = 0, fail = 0;
const check = (name, cond, detail) => { cond ? pass++ : fail++; console.log(`${cond ? "✓" : "✗"} ${name}  ${detail}`); };

console.log("Corroboration fix — constellation vs single cue\n");

// 1) Engagement constellation: lean + nod + tilt -> confident & corroborated
let t = top({ posture_lean_forward: .7, regulator_head_nod: .6, regulator_head_tilt: .4 });
check("engagement constellation is confident", t.state.startsWith("Engagement") && t.p > 0.55 && t.corroborated,
      `-> ${t.state} ${(t.p*100).toFixed(0)}% (${t.nCues} cues, corroborated=${t.corroborated})`);

// 2) Single cue only -> NOT corroborated, low confidence (honest hedge)
t = top({ posture_lean_forward: .7 });
check("single cue stays hedged", !t.corroborated && t.p < 0.5,
      `-> ${t.state} ${(t.p*100).toFixed(0)}% (${t.nCues} cue, corroborated=${t.corroborated})`);

// 3) Discomfort/anxiety constellation: self-touch + fidget + lip-press
t = top({ adaptor_self_touch_face: .7, body_fidget: .6, face_lip_press: .5 });
check("anxiety constellation is confident", t.state.startsWith("Discomfort") && t.p > 0.55 && t.corroborated,
      `-> ${t.state} ${(t.p*100).toFixed(0)}% (${t.nCues} cues)`);

// 4) Dominance constellation: expansive + hands-on-hips
t = top({ posture_expansive: .7, posture_hands_on_hips: .6 });
check("dominance constellation is confident", t.state.startsWith("Confidence") && t.p > 0.5 && t.corroborated,
      `-> ${t.state} ${(t.p*100).toFixed(0)}% (${t.nCues} cues)`);

// 5) Negative evidence suppresses: engagement cues + strong lean-back should NOT read high-engagement
t = computeConstructs({ posture_lean_forward: .3, posture_lean_back: .8, gaze_aversion: .7 }).find(c => c.state.startsWith("Engagement")) || { p: 0 };
check("negative cues suppress engagement", t.p < 0.4, `-> engagement ${(t.p*100).toFixed(0)}% with strong lean-back/gaze-away`);

console.log(`\n${pass}/${pass+fail} passed`);
process.exit(fail ? 1 : 0);
