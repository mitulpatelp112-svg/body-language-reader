// Lightweight i18n for the Presence landing. Translates [data-i18n] elements; persists choice.
const STR = {
  en: { eyebrow: "On-device. Private by design.", h1: "See how you<br>actually come across.",
    sub: "Presence reads your face, posture, voice, and pulse in real time, then coaches your presence. Nothing leaves your device.",
    demo: "Open the demo", science: "See the science", n1: "How it works", n2: "Science", n3: "Privacy" },
  es: { eyebrow: "En tu dispositivo. Privado por diseño.", h1: "Mira cómo te<br>perciben en realidad.",
    sub: "Presence lee tu rostro, postura, voz y pulso en tiempo real y entrena tu presencia. Nada sale de tu dispositivo.",
    demo: "Abrir la demo", science: "Ver la ciencia", n1: "Cómo funciona", n2: "Ciencia", n3: "Privacidad" },
  fr: { eyebrow: "Sur l'appareil. Privé par conception.", h1: "Voyez comment vous<br>êtes vraiment perçu.",
    sub: "Presence lit votre visage, posture, voix et pouls en temps réel, puis coache votre présence. Rien ne quitte votre appareil.",
    demo: "Ouvrir la démo", science: "Voir la science", n1: "Comment ça marche", n2: "Science", n3: "Confidentialité" },
  de: { eyebrow: "Auf dem Gerät. Privat by design.", h1: "Sieh, wie du<br>wirklich wirkst.",
    sub: "Presence liest dein Gesicht, deine Haltung, Stimme und deinen Puls in Echtzeit und coacht deine Präsenz. Nichts verlässt dein Gerät.",
    demo: "Demo öffnen", science: "Zur Wissenschaft", n1: "So funktioniert es", n2: "Wissenschaft", n3: "Datenschutz" },
  hi: { eyebrow: "आपके डिवाइस पर. डिज़ाइन से निजी.", h1: "देखें आप असल में<br>कैसे दिखते हैं.",
    sub: "Presence आपके चेहरे, मुद्रा, आवाज़ और नब्ज़ को रियल-टाइम में पढ़ता है और आपकी प्रेज़ेंस को कोच करता है. कुछ भी डिवाइस से बाहर नहीं जाता.",
    demo: "डेमो खोलें", science: "विज्ञान देखें", n1: "यह कैसे काम करता है", n2: "विज्ञान", n3: "गोपनीयता" },
};
function apply(lang) {
  const t = STR[lang] || STR.en;
  document.querySelectorAll("[data-i18n]").forEach((el) => { const v = t[el.dataset.i18n]; if (v != null) el.innerHTML = v; });
  document.documentElement.lang = lang;
  try { localStorage.setItem("presence-lang", lang); } catch (e) {}
}
const sel = document.getElementById("langSelect");
if (sel) {
  let saved = "en"; try { saved = localStorage.getItem("presence-lang") || "en"; } catch (e) {}
  sel.value = saved; apply(saved);
  sel.addEventListener("change", () => apply(sel.value));
}
