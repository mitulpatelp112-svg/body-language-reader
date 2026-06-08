"""
Train an emotion classifier from the eval harness dataset and export it for the browser app.

Input:  blendshape_dataset.jsonl  (from app/eval.html -> "Download dataset")
        each line: {"label": "happiness", "mouthSmileLeft": 0.8, ...}
Output: app/model.json            (the app auto-loads this and fuses it in)

This closes the loop: collect data in the browser -> train here -> app uses the trained model
instead of hand-tuned priors. Reports per-class precision/recall/F1 so you SEE the accuracy.

Run:
    pip install scikit-learn numpy
    python3 train.py path/to/blendshape_dataset.jsonl
"""
import sys, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if len(rows) < 30:
        print(f"Only {len(rows)} samples — collect more (run the eval a few times across people/lighting).")
    labels = sorted({r["label"] for r in rows})
    feats = sorted({k for r in rows for k in r if k != "label"})
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows])
    print(f"{len(rows)} samples · {len(feats)} features · classes: {labels}\n")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    # honest accuracy via stratified cross-validation
    n_splits = min(5, np.min(np.bincount([labels.index(v) for v in y])))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y_cv = cross_val_predict(clf, Xs, y, cv=cv)
        acc = accuracy_score(y, y_cv)
        macro_f1 = f1_score(y, y_cv, average="macro", zero_division=0)
        print("Cross-validated performance:\n")
        print(classification_report(y, y_cv, zero_division=0))
        print(f">>> HEADLINE: accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  (report these)\n")
        # NOTE: for a DIMENSIONAL model (regressing arousal/dominance/valence) the field's metric is
        # CCC (concordance correlation). Swap LogisticRegression for a regressor and score with CCC
        # when you collect A/D/V labels (e.g., via the voice_adv backend). See RESEARCH_REPORT.md §3.
    else:
        acc = float("nan")
        print("Too few per-class samples for CV — accuracy not estimated.")

    clf.fit(Xs, y)
    classes = list(clf.classes_)
    # LogisticRegression with 2 classes yields one coef row; normalize to per-class rows
    if len(classes) == 2:
        coef = [(-clf.coef_[0]).tolist(), clf.coef_[0].tolist()]
        intercept = [float(-clf.intercept_[0]), float(clf.intercept_[0])]
    else:
        coef = clf.coef_.tolist()
        intercept = clf.intercept_.tolist()

    model = {
        "features": feats, "classes": classes,
        "mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(),
        "coef": coef, "intercept": intercept,
        "cv_accuracy": None if np.isnan(acc) else round(float(acc), 4),
        "n_samples": len(rows),
    }
    out = os.path.join(os.path.dirname(__file__), "..", "app", "model.json")
    json.dump(model, open(out, "w"))
    print(f"\nSaved {os.path.abspath(out)} — reload the app; it will auto-load and fuse this model.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 train.py <blendshape_dataset.jsonl>"); sys.exit(1)
    main(sys.argv[1])
