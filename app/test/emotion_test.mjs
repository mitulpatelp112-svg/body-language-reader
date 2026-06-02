// Automated test of the emotion-prototype classifier (the logic eval.html runs).
// Verifies each canonical AU constellation is recognized as the correct emotion,
// and prints a synthetic confusion check. Run: node app/test/emotion_test.mjs
import { EMOTIONS, scoreEmotions, topPrediction, CANONICAL } from "../emotion-core.js";

let pass = 0, fail = 0;
console.log("Emotion-prototype classifier — canonical AU pattern test\n");
for (const emo of EMOTIONS) {
  const pred = topPrediction(CANONICAL[emo]);
  const scores = scoreEmotions(CANONICAL[emo]);
  const ranked = EMOTIONS.map(e=>`${e}:${scores[e].toFixed(2)}`).sort((a,b)=>parseFloat(b.split(":")[1])-parseFloat(a.split(":")[1]));
  const ok = pred === emo;
  console.log(`${ok ? "✓" : "✗"} ${emo.padEnd(10)} -> ${pred.padEnd(10)}  [${ranked.slice(0,3).join("  ")}]`);
  ok ? pass++ : fail++;
}
// neutral check: empty face should NOT fire an emotion
const neutralOk = topPrediction({}) === "neutral";
console.log(`${neutralOk ? "✓" : "✗"} neutral    -> ${topPrediction({})}`);
neutralOk ? pass++ : fail++;

console.log(`\n${pass}/${pass+fail} passed`);
process.exit(fail ? 1 : 0);
