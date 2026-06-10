"""
Export the trained PyTorch checkpoint to ONNX + write the meta JSON.
Standalone so training crashes during export don't waste an epoch.

Usage:
    python3 backend/export_cnn_onnx.py
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_cnn import FilteredFaceDataset, MEAN, STD, INPUT, split_indices, make_model

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app"
CKPT = APP / "emotion_cnn.pt"
ONNX_OUT = APP / "emotion_cnn.onnx"
META_OUT = APP / "emotion_cnn_meta.json"
DATA = HERE.parent / "resources" / "datasets" / "affectnet"


def evaluate(model, loader, device, classes):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); logits = model(x)
            ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "report":   classification_report(y, p, target_names=classes, zero_division=0, digits=3),
        "confusion": confusion_matrix(y, p).tolist(),
    }


def main():
    if not CKPT.exists():
        raise SystemExit(f"checkpoint missing: {CKPT}. Re-run train_cnn.py first.")
    device = torch.device("cpu")  # export on CPU for clean ONNX graph
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    print(f"loaded checkpoint  classes={classes}  saved val_acc={ckpt.get('val_acc'):.3f}")

    model = make_model(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # rebuild the same test split to report final numbers
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(INPUT),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    base = FilteredFaceDataset(DATA, eval_tf)
    tr_idx, va_idx, te_idx = split_indices(len(base), val_frac=0.10, test_frac=0.15, seed=0)
    test_ds = FilteredFaceDataset(DATA, eval_tf, indices=te_idx)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    print(f"re-evaluating on {len(test_ds):,} held-out test images...")
    t = evaluate(model, loader, device, classes)
    print(t["report"])
    print(f">>> HEADLINE: test_acc={t['accuracy']:.3f}  macro-F1={t['macro_f1']:.3f}")

    # ONNX export with the legacy exporter (no onnxscript required)
    print("exporting to ONNX (legacy exporter)...")
    dummy = torch.randn(1, 3, INPUT, INPUT)
    torch.onnx.export(
        model, dummy, str(ONNX_OUT),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # use the legacy TorchScript exporter
    )
    print(f"  -> {ONNX_OUT} ({ONNX_OUT.stat().st_size/1e6:.2f} MB)")

    meta = {
        "classes": classes, "mean": MEAN, "std": STD, "input_size": INPUT,
        "val_acc": float(ckpt.get("val_acc", 0)),
        "test_acc": t["accuracy"], "test_macro_f1": t["macro_f1"],
        "n_test": len(test_ds),
        "confusion": t["confusion"],
        "report": t["report"],
        "trained_on": "AffectNet 6-class (anger, disgust, fear, happiness, sadness, surprise — neutral & contempt excluded)",
        "architecture": "MobileNetV3-Small (ImageNet pretrained, fine-tuned)",
    }
    json.dump(meta, open(META_OUT, "w"), indent=2)
    print(f"  -> {META_OUT}")


if __name__ == "__main__":
    main()
