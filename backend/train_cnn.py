"""
Train a MobileNetV3-Small on the 6-class AffectNet subset (drops `neutral` and `contempt`)
on M-series GPU via MPS. Reports test-set accuracy + per-class P/R/F1 + confusion matrix.

Why this model:
  - 2.5M params, ~10ms inference on M3 Pro MPS, ~30ms on CPU
  - ImageNet-pretrained head transfers cleanly to face crops
  - Exports cleanly to ONNX for in-browser inference later

Output:
  app/emotion_cnn.pt          - PyTorch state_dict (best val-acc checkpoint)
  app/emotion_cnn.onnx        - exported model (224x224 RGB input, 6-way logits)
  app/emotion_cnn_meta.json   - { classes, mean, std, input_size, val_acc, test_acc }

Usage:
  python3 backend/train_cnn.py --data resources/datasets/affectnet --epochs 12 --batch 128
"""
import argparse, json, os, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Drop these two — they're the noisiest classes (per our blendshape baseline).
EXCLUDE = {"neutral", "contempt"}
# Folder-name -> app-vocabulary remap so a downstream fuse with the heuristic layer aligns.
REMAP = {"happy": "happiness", "sad": "sadness"}

# ImageNet stats (MobileNetV3 was pretrained on ImageNet).
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
INPUT = 224

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FilteredFaceDataset(Dataset):
    """ImageFolder-style loader that filters EXCLUDE classes and remaps names."""
    def __init__(self, root: Path, transform, indices=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        classes = sorted(
            REMAP.get(d.name, d.name)
            for d in root.iterdir() if d.is_dir() and d.name not in EXCLUDE
        )
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        # walk in deterministic order
        for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            raw = cls_dir.name
            if raw in EXCLUDE: continue
            label = REMAP.get(raw, raw)
            idx = self.class_to_idx[label]
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    self.samples.append((img, idx))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, y


def split_indices(n: int, val_frac=0.1, test_frac=0.15, seed=0):
    """Deterministic train/val/test split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(n * test_frac); n_val = int(n * val_frac)
    return idx[n_test+n_val:].tolist(), idx[:n_val].tolist(), idx[n_val:n_val+n_test].tolist()


def make_model(n_classes: int) -> nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_feat = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feat, n_classes)
    return m


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device); logits = model(x)
        ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "report":   classification_report(y, p, target_names=classes, zero_division=0, digits=3),
        "confusion": confusion_matrix(y, p).tolist(),
        "y": y.tolist(), "p": p.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="resources/datasets/affectnet")
    ap.add_argument("--epochs",  type=int, default=12)
    ap.add_argument("--batch",   type=int, default=128)
    ap.add_argument("--lr",      type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--out",     default="app")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomResizedCrop(INPUT, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(INPUT),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])

    root = Path(args.data).resolve()
    # one cheap pass to enumerate the file list — we re-instantiate later with the right transform per split
    base = FilteredFaceDataset(root, eval_tf)
    classes = base.classes
    n = len(base)
    print(f"{n:,} images · classes: {classes}")
    counts = {c: 0 for c in classes}
    for _, y in base.samples: counts[classes[y]] += 1
    for c, v in counts.items(): print(f"  {c:<10s} {v:,}")

    tr_idx, va_idx, te_idx = split_indices(n, val_frac=0.10, test_frac=0.15, seed=args.seed)
    train_ds = FilteredFaceDataset(root, train_tf, indices=tr_idx)
    val_ds   = FilteredFaceDataset(root, eval_tf,  indices=va_idx)
    test_ds  = FilteredFaceDataset(root, eval_tf,  indices=te_idx)
    print(f"split: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # Class weights for the loss (helps disgust/contempt-style minority classes).
    cls_counts = np.array([counts[c] for c in classes], dtype=np.float64)
    weights = (cls_counts.mean() / cls_counts).astype(np.float32)
    print(f"class weights: {dict(zip(classes, weights.round(2).tolist()))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)

    model = make_model(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), label_smoothing=0.05)

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)
    best_val = 0.0
    best_path = out_dir / "emotion_cnn.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); running, n_seen, n_correct = 0.0, 0, 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x); loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0); n_seen += x.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            if i % 25 == 0:
                print(f"  e{epoch} step {i:4d}/{len(train_loader)}  loss={loss.item():.3f}  "
                      f"running_acc={n_correct/max(n_seen,1):.3f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        sched.step()
        train_loss = running / max(n_seen, 1); train_acc = n_correct / max(n_seen, 1)

        v = evaluate(model, val_loader, device, classes)
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_acc={v['accuracy']:.3f} val_macroF1={v['macro_f1']:.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if v["accuracy"] > best_val:
            best_val = v["accuracy"]
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "mean": MEAN, "std": STD, "input": INPUT,
                        "val_acc": best_val}, best_path)
            print(f"  -> new best, saved {best_path}")

    # final test eval with the best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    t = evaluate(model, test_loader, device, classes)
    print("\n===== HELD-OUT TEST SET =====")
    print(t["report"])
    print(f">>> HEADLINE: test_accuracy={t['accuracy']:.3f}  macro-F1={t['macro_f1']:.3f}")

    # ONNX export
    onnx_path = out_dir / "emotion_cnn.onnx"
    model.eval()
    dummy = torch.randn(1, 3, INPUT, INPUT, device=device)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported -> {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    meta = {
        "classes": classes, "mean": MEAN, "std": STD, "input_size": INPUT,
        "val_acc": best_val, "test_acc": t["accuracy"], "test_macro_f1": t["macro_f1"],
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "confusion": t["confusion"],
        "trained_on": "AffectNet (6 classes: anger, disgust, fear, happiness, sadness, surprise — neutral & contempt excluded)",
    }
    meta_path = out_dir / "emotion_cnn_meta.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"meta    -> {meta_path}")


if __name__ == "__main__":
    main()
