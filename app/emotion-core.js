// Shared emotion-prototype scorer (pure, no imports) — used by eval.js AND the node test,
// so the thing we test is the thing that runs. Mirrors the live app's AU-constellation logic.

export const EMOTIONS = ["happiness","sadness","surprise","fear","anger","disgust","contempt"];

// geometric mean: AU constellation co-activation (ALL components must be present)
export const gm = (...xs) => Math.pow(xs.reduce((p,x)=>p*Math.max(1e-4,x),1), 1/xs.length);

// bs = { blendshapeName: score } from MediaPipe FaceLandmarker
export function scoreEmotions(bs) {
  const m = (k) => bs[k] || 0;
  const smile=(m("mouthSmileLeft")+m("mouthSmileRight"))/2, cheek=(m("cheekSquintLeft")+m("cheekSquintRight"))/2;
  const browIn=m("browInnerUp"), browOut=(m("browOuterUpLeft")+m("browOuterUpRight"))/2, browDn=(m("browDownLeft")+m("browDownRight"))/2;
  const eyeWide=(m("eyeWideLeft")+m("eyeWideRight"))/2, eyeSq=(m("eyeSquintLeft")+m("eyeSquintRight"))/2;
  const jaw=m("jawOpen"), frown=(m("mouthFrownLeft")+m("mouthFrownRight"))/2, chin=(m("mouthShrugUpper")+m("mouthShrugLower"))/2;
  const noseS=(m("noseSneerLeft")+m("noseSneerRight"))/2, upLip=(m("mouthUpperUpLeft")+m("mouthUpperUpRight"))/2, lowLip=(m("mouthLowerDownLeft")+m("mouthLowerDownRight"))/2;
  const lipPr=(m("mouthPressLeft")+m("mouthPressRight"))/2, lipStr=(m("mouthStretchLeft")+m("mouthStretchRight"))/2;
  const asym=Math.abs(m("mouthSmileLeft")-m("mouthSmileRight")), dimp=(m("mouthDimpleLeft")+m("mouthDimpleRight"))/2;
  return {
    happiness: gm(smile, 0.3+cheek)*2.0,
    sadness:   gm(frown, browIn+browDn+0.05, chin+0.05)*3.0,
    surprise:  gm(browOut+browIn, eyeWide, jaw)*3.2,
    fear:      gm(browIn, eyeWide, lipStr+0.03)*3.2,
    anger:     gm(browDn, eyeWide+eyeSq, lipPr+0.03)*3.2,
    disgust:   gm(noseS, upLip, lowLip+0.05)*3.0,
    contempt:  (asym + dimp*0.6)*2.4
  };
}

// argmax with a neutral floor
export function topPrediction(bs, thresh = 0.18) {
  const s = scoreEmotions(bs);
  let best = "neutral", bv = thresh;
  for (const k of EMOTIONS) if (s[k] > bv) { bv = s[k]; best = k; }
  return best;
}

// the canonical AU patterns each emotion *should* be recognized from (for tests + docs)
export const CANONICAL = {
  happiness: { mouthSmileLeft:.9, mouthSmileRight:.9, cheekSquintLeft:.7, cheekSquintRight:.7 },
  sadness:   { mouthFrownLeft:.8, mouthFrownRight:.8, browInnerUp:.7, browDownLeft:.35, browDownRight:.35, mouthShrugUpper:.6 },
  surprise:  { browOuterUpLeft:.8, browOuterUpRight:.8, browInnerUp:.8, eyeWideLeft:.8, eyeWideRight:.8, jawOpen:.8 },
  fear:      { browInnerUp:.8, browOuterUpLeft:.5, browOuterUpRight:.5, eyeWideLeft:.8, eyeWideRight:.8, mouthStretchLeft:.7, mouthStretchRight:.7 },
  anger:     { browDownLeft:.85, browDownRight:.85, eyeWideLeft:.5, eyeWideRight:.5, eyeSquintLeft:.5, eyeSquintRight:.5, mouthPressLeft:.7, mouthPressRight:.7 },
  disgust:   { noseSneerLeft:.85, noseSneerRight:.85, mouthUpperUpLeft:.7, mouthUpperUpRight:.7, mouthLowerDownLeft:.6, mouthLowerDownRight:.6 },
  contempt:  { mouthSmileLeft:.85, mouthDimpleLeft:.6, mouthDimpleRight:.6 }
};
