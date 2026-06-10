"""
Remap AffectNet folder names to the existing browser app's emotion vocabulary so
the trained model.json fuses into the same emotion-state buckets used by the
heuristic and the FACS backend.

  happy -> happiness
  sad   -> sadness
  (anger, fear, surprise, disgust, contempt, neutral stay as-is)

Usage:
    python3 backend/relabel_for_app.py data/affectnet_blendshapes.jsonl data/affectnet_blendshapes_app.jsonl
"""
import json, sys

REMAP = {"happy": "happiness", "sad": "sadness"}

def main(src, dst):
    counts = {}
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            r["label"] = REMAP.get(r["label"], r["label"])
            counts[r["label"]] = counts.get(r["label"], 0) + 1
            fout.write(json.dumps(r) + "\n")
    print(f"Wrote {sum(counts.values()):,} rows -> {dst}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10s} {v:,}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: relabel_for_app.py <in.jsonl> <out.jsonl>"); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
