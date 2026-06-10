// Shared theme toggle for AJNA (landing, demo, eval). Pre-paint theme is set by a tiny inline
// script in each <head> to avoid a flash; this only wires the toggle button + persistence.
(function () {
  const root = document.documentElement;
  const btn = document.getElementById("themeToggle");
  if (!btn) return;
  const SUN = '<i class="ph ph-sun"></i>', MOON = '<i class="ph ph-moon"></i>';
  function paint() {
    const dark = root.dataset.theme !== "light";
    btn.innerHTML = dark ? SUN : MOON;                       // show the action it performs
    btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  }
  btn.addEventListener("click", () => {
    const next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try { localStorage.setItem("presence-theme", next); } catch (e) {}
    paint();
  });
  paint();
})();
