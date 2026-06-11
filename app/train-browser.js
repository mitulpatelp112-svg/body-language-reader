// In-browser model training for Presence. Trains a multinomial logistic-regression classifier
// (softmax + gradient descent) on the user's own labelled blendshape samples, entirely on-device.
// Output matches the model.json schema that model-infer.js consumes, so the live app auto-uses it.

export function trainModel(dataset, { epochs = 300, lr = 0.3, holdout = 0.2 } = {}) {
  if (!dataset || dataset.length < 14) return { ok: false, reason: "Need more samples. Run the eval a couple times first." };

  const classes = [...new Set(dataset.map((d) => d.label))].sort();
  const features = [...new Set(dataset.flatMap((d) => Object.keys(d).filter((k) => k !== "label")))].sort();
  if (classes.length < 2) return { ok: false, reason: "Need at least two different expressions." };

  // matrix
  const X = dataset.map((d) => features.map((f) => +d[f] || 0));
  const y = dataset.map((d) => classes.indexOf(d.label));

  // standardize
  const n = X.length, F = features.length, C = classes.length;
  const mean = features.map((_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const std = features.map((_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mean[j]) ** 2, 0) / n) || 1);
  const Xs = X.map((r) => r.map((v, j) => (v - mean[j]) / std[j]));

  // shuffle + split (seeded-ish by index parity for determinism without RNG)
  const idx = Xs.map((_, i) => i).sort((a, b) => ((a * 2654435761) % 1000) - ((b * 2654435761) % 1000));
  const cut = Math.max(1, Math.floor(idx.length * holdout));
  const testI = new Set(idx.slice(0, cut)), trainI = idx.filter((i) => !testI.has(i));

  // weights [C][F], bias [C]
  const W = Array.from({ length: C }, () => new Float64Array(F));
  const b = new Float64Array(C);
  const softmax = (xi) => {
    const z = W.map((wc, c) => b[c] + wc.reduce((s, w, j) => s + w * xi[j], 0));
    const mx = Math.max(...z); const e = z.map((v) => Math.exp(v - mx)); const sum = e.reduce((a, x) => a + x, 0) || 1;
    return e.map((v) => v / sum);
  };

  for (let ep = 0; ep < epochs; ep++) {
    const gW = Array.from({ length: C }, () => new Float64Array(F)); const gB = new Float64Array(C);
    for (const i of trainI) {
      const p = softmax(Xs[i]);
      for (let c = 0; c < C; c++) {
        const err = p[c] - (y[i] === c ? 1 : 0);
        gB[c] += err;
        for (let j = 0; j < F; j++) gW[c][j] += err * Xs[i][j];
      }
    }
    const m = trainI.length;
    for (let c = 0; c < C; c++) { b[c] -= (lr * gB[c]) / m; for (let j = 0; j < F; j++) W[c][j] -= (lr * (gW[c][j] / m + 0.001 * W[c][j])); }
  }

  // holdout accuracy
  let correct = 0;
  for (const i of testI) { const p = softmax(Xs[i]); const pred = p.indexOf(Math.max(...p)); if (pred === y[i]) correct++; }
  const acc = +(correct / testI.size).toFixed(3);

  return {
    ok: true,
    model: {
      features, classes, mean, std,
      coef: W.map((w) => Array.from(w)), intercept: Array.from(b),
      cv_accuracy: acc, n_samples: n, trained_in_browser: true, trained_at: new Date().toISOString(),
    },
    acc, n, classes,
  };
}
