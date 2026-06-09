// Presence service worker - makes the app installable + offline-capable.
// Cache-first for the app shell, runtime-cache for everything else (incl. CDN models).
const VERSION = "presence-v1";
const SHELL = [
  "./", "index.html", "brand.js", "favicon.svg", "site.webmanifest",
  "assets/og.png", "assets/icon-512.png", "assets/apple-touch-icon.png",
  "app/", "app/index.html", "app/app.js", "app/emotion-core.js",
  "app/model-infer.js", "app/eval.html", "app/eval.js",
  "knowledge-base/signals.json",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(VERSION).then((c) => c.addAll(SHELL).catch(() => {})).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  e.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((res) => {
        // runtime cache successful + opaque (CDN) responses so 2nd load works offline
        if (res && (res.ok || res.type === "opaque")) {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(req, copy).catch(() => {}));
        }
        return res;
      }).catch(() => cached);
    })
  );
});
