// Loads a trained logistic-regression model (model.json from backend/train.py) and runs it
// in-browser. When present, the app fuses its calibrated predictions with the heuristic layer,
// progressively replacing hand-tuned priors with a data-trained classifier. Graceful no-op if
// model.json is absent.
let MODEL = null;

export async function loadModel(url = "./model.json") {
  // 1) prefer a model the user trained on THIS device (localStorage)
  try {
    const local = localStorage.getItem("presence-model");
    if (local) {
      const m = JSON.parse(local);
      if (m.features && m.classes && m.coef) { MODEL = m; console.log("personal (on-device) model loaded"); return true; }
    }
  } catch {}
  // 2) fall back to a server-trained model.json if present
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return false;
    const m = await r.json();
    if (!m.features || !m.classes || !m.coef) return false;
    MODEL = m;
    console.log("trained model loaded:", m.classes, `(${m.features.length} features)`);
    return true;
  } catch { return false; }
}
export function hasModel() { return !!MODEL; }
export function modelInfo() { return MODEL ? { classes: MODEL.classes, n: MODEL.features.length, acc: MODEL.cv_accuracy } : null; }

// featObj: { blendshapeName: score }. Returns { class: prob } softmax, or null.
export function predict(featObj) {
  if (!MODEL) return null;
  const x = MODEL.features.map((f, i) => (((featObj[f] || 0) - MODEL.mean[i]) / (MODEL.std[i] || 1)));
  const logits = MODEL.classes.map((_, k) =>
    MODEL.intercept[k] + x.reduce((s, xi, i) => s + xi * MODEL.coef[k][i], 0));
  const mx = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - mx));
  const z = exps.reduce((a, b) => a + b, 0) || 1;
  const out = {};
  MODEL.classes.forEach((c, k) => out[c] = exps[k] / z);
  return out;
}
