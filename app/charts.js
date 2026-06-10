// Tiny dependency-free canvas line chart for Presence (real data viz; brand-styled, responsive).
export function lineChart(canvas, series, opts = {}) {
  const { yMin = 0, yMax = 1, grid = true } = opts;
  const css = getComputedStyle(document.documentElement);
  const line = css.getPropertyValue("--line").trim() || "#232c37";
  const mut = css.getPropertyValue("--mut").trim() || "#94a1b2";
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600, H = canvas.clientHeight || 200;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const padL = 30, padR = 10, padT = 10, padB = 20;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const allX = series.flatMap((s) => s.data.map((p) => p.x));
  const xMin = Math.min(...allX, 0), xMax = Math.max(...allX, 1) || 1;
  const X = (x) => padL + ((x - xMin) / (xMax - xMin || 1)) * plotW;
  const Y = (y) => padT + (1 - (y - yMin) / (yMax - yMin || 1)) * plotH;

  if (grid) {
    ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.fillStyle = mut; ctx.font = "10px -apple-system,system-ui,sans-serif";
    for (let i = 0; i <= 4; i++) {
      const y = yMin + (i / 4) * (yMax - yMin), py = Y(y);
      ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(W - padR, py); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(Math.round(y * 100) + "", 4, py + 3);
    }
  }
  for (const s of series) {
    if (!s.data.length) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
    s.data.forEach((p, i) => { const x = X(p.x), y = Y(p.y); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    if (s.dots) { ctx.fillStyle = s.color; for (const p of s.data) { ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 2.5, 0, 7); ctx.fill(); } }
  }
}
