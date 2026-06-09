# Launching Presence

## Live now (free, today): installable PWA
Presence is a Progressive Web App. It runs fully on-device over HTTPS, works offline after first
load (service worker), and installs to the home screen like a native app. No store, no fee, no review.

**Open:** https://mitulpatelp112-svg.github.io/body-language-reader/

**Install on iPhone / iPad (Safari):**
1. Open the link in Safari.
2. Tap the **Share** button.
3. Choose **Add to Home Screen**.
4. Launch it from the icon. It opens fullscreen; allow camera + mic when asked.

**Install on Android (Chrome):** tap the **Install** button in the header, or the browser's
"Install app" prompt.

**Install on desktop (Chrome/Edge):** click **Install** in the header or the address-bar install icon.

> iOS 14.3+ allows the camera in installed PWAs, so the live reads work from the home-screen app.

## The real App Store path (needs the paid Apple program)
The Apple App Store cannot be done for free or same-day. To publish there you need:
1. **Apple Developer Program - $99/year** (hard requirement to submit anything).
2. A **Mac with Xcode** and signing certificates.
3. **App Review** (~1-2 days; camera/health/emotion apps get extra scrutiny, so the honesty
   framing and privacy policy matter).

When you're ready, the cleanest route wraps this exact web app in a native shell so you don't rebuild it:

```bash
# from the project root
npm create @capacitor/app           # or: npm i @capacitor/core @capacitor/cli && npx cap init
npx cap add ios
# point Capacitor's webDir at the built static site (this repo's root)
npx cap copy ios
npx cap open ios                     # opens Xcode -> set bundle id, signing, then Archive -> upload
```

Capacitor gives native camera/mic permission prompts and an App Store binary while keeping the
MediaPipe + on-device pipeline unchanged. Ask and I can scaffold the Capacitor project and the
required **privacy policy + App Privacy "nutrition label"** (Presence collects no data and sends
nothing off-device, which is the easy, honest answer Apple wants).

## Store-readiness checklist (for later)
- [ ] Apple Developer enrollment ($99)
- [ ] Capacitor iOS wrapper + bundle id
- [ ] `NSCameraUsageDescription` / `NSMicrophoneUsageDescription` strings
- [ ] Privacy policy URL (on-device, no collection) + App Privacy answers
- [ ] App icon set + screenshots (the OG art + UI shots)
- [ ] Honest store copy: "decision-support, not a verdict; no lie detection"
