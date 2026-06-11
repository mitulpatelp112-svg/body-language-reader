// Builds the Science page from the live knowledge base (knowledge-base/signals.json).
const $ = (id) => document.getElementById(id);

const MODALITY = {
  facial: ["Face", "ph-smiley"], body: ["Body and posture", "ph-person-simple"],
  gesture: ["Hands and gesture", "ph-hand"], subtle_movement: ["Subtle movement", "ph-waveform"],
  speech_prosody: ["Voice", "ph-microphone"], speech_content: ["Voice", "ph-microphone"],
  physiological: ["Physiology", "ph-heartbeat"], external_context: ["Context", "ph-map-pin"],
};
const MOD_ORDER = ["facial", "body", "gesture", "subtle_movement", "speech_prosody", "physiological", "external_context"];
const evClass = (e) => ({ strong: "t-strong", moderate: "t-moderate", weak: "t-weak", folklore: "t-folklore" }[e] || "t-plain");
const esc = (s) => (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

(async () => {
  let kb;
  try { kb = await (await fetch("knowledge-base/signals.json", { cache: "no-store" })).json(); }
  catch { $("signals").innerHTML = "<p class='sub'>Could not load the knowledge base.</p>"; return; }

  const byId = Object.fromEntries(kb.signals.map((s) => [s.id, s.label]));
  const constructs = Object.entries(kb.constructs || {}).filter(([k]) => !k.startsWith("_"));
  $("kSignals").textContent = kb.signals.length;
  $("kConstructs").textContent = constructs.length;

  // constructs
  $("constructs").innerHTML = constructs.map(([, c]) => {
    const top = Object.entries(c.positive || {}).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([id]) => byId[id] || id);
    const rel = c.reliability || 3;
    return `<div class="card reveal">
      <h3>${esc(c.label)}</h3>
      <p>${esc((c.evidence || "").split(";")[0])}.</p>
      <div class="votes"><span class="tag t-plain">rel ${rel}/5</span>${top.map((t) => `<span class="tag t-plain">${esc(t)}</span>`).join("")}</div>
      ${c.sources ? `<div class="src">${esc(c.sources.join(" · "))}</div>` : ""}
    </div>`;
  }).join("");

  // signals grouped by modality
  const groups = {};
  for (const s of kb.signals) { const m = (s.modality || ["body"])[0]; (groups[m] = groups[m] || []).push(s); }
  const order = [...MOD_ORDER.filter((m) => groups[m]), ...Object.keys(groups).filter((m) => !MOD_ORDER.includes(m))];
  $("signals").innerHTML = order.map((m) => {
    const [name, icon] = MODALITY[m] || [m, "ph-dot"];
    const rows = groups[m].map((s) => {
      const it = (s.interpretations || [])[0] || {};
      const ev = it.evidence || "moderate";
      return `<div class="sig reveal">
        <div class="nm">${esc(s.label)}</div>
        <div class="meta"><span class="tag ${evClass(ev)}">${ev}</span><span class="rel">${s.inference_reliability ?? "-"}/5</span></div>
        <div class="ob">${esc(s.observable || it.state || "")}</div>
        ${it.caveats ? `<div class="cav">${esc(it.caveats)}</div>` : ""}
      </div>`;
    }).join("");
    return `<div class="modblock"><div class="modhead"><i class="ph-bold ${icon}"></i> ${esc(name)} · ${groups[m].length}</div>${rows}</div>`;
  }).join("");

  // scroll reveal
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.1 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
})();
