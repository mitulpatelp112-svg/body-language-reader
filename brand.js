// Shared script for Presence (landing, demo, eval): theme toggle, install prompt, service worker.
(function () {
  const root = document.documentElement;

  // ---- Theme toggle ----
  const btn = document.getElementById("themeToggle");
  if (btn) {
    const SUN = '<i class="ph ph-sun"></i>', MOON = '<i class="ph ph-moon"></i>';
    const paint = () => {
      const dark = root.dataset.theme !== "light";
      btn.innerHTML = dark ? SUN : MOON;
      btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    };
    btn.addEventListener("click", () => {
      const next = root.dataset.theme === "light" ? "dark" : "light";
      root.dataset.theme = next;
      try { localStorage.setItem("presence-theme", next); } catch (e) {}
      paint();
    });
    paint();
  }

  // ---- Install (Add to Home Screen) ----
  const ib = document.getElementById("installBtn");
  let deferred = null;
  const standalone = window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone;
  window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferred = e; if (ib) ib.hidden = false; });
  if (ib && !standalone) {
    const isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent);
    if (isIOS) ib.hidden = false;   // iOS has no beforeinstallprompt; show with instructions
    ib.addEventListener("click", async () => {
      if (deferred) { deferred.prompt(); await deferred.userChoice; deferred = null; ib.hidden = true; }
      else { alert('To install Presence on iPhone:\n\n1. Tap the Share button.\n2. Choose "Add to Home Screen".'); }
    });
  }
  window.addEventListener("appinstalled", () => { if (ib) ib.hidden = true; });

  // ---- Service worker (offline + instant load) ----
  if ("serviceWorker" in navigator) {
    const swPath = location.pathname.includes("/app/") ? "../sw.js" : "sw.js";
    window.addEventListener("load", () => navigator.serviceWorker.register(swPath).catch(() => {}));
  }
})();
