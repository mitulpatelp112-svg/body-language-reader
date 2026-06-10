"""
Upgraded emotion CNN trainer — the accuracy-push version.

Stacks the gains that actually move AffectNet:
  - trains on FACE-CROPPED data (run crop_faces.py first)
  - stronger backbone (efficientnet_b2 default; convnext_tiny / efficientnet_b0 selectable)
  - WeightedRandomSampler -> every class seen equally per epoch
  - RandAugment + RandomErasing + horizontal flip + color jitter
  - mixup / cutmix (per-batch, alternating)
  - label smoothing + AdamW + cosine schedule with linear warmup
  - test-time augmentation (center + h-flip) at eval

Realistic target on AffectNet 6-class: ~75-78%. The label noise floor (~35% human
disagreement) makes 95% unreachable on this dataset — see RESEARCH_REPORT.md / Barrett 2019.

Usage:
    python3 backend/train_cnn2.py --data resources/datasets/affectnet_faces \
        --arch efficientnet_b2 --epochs 22 --batch 64
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

EXCLUDE = {"neutral", "contempt"}
REMAP = {"happy": "happiness", "sad": "sadness"}
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ARCHS = {
    # name -> (constructor, weights enum, input size)
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights", 224),
    "efficientnet_b2": (models.efficientnet_b2, "EfficientNet_B2_Weights", 260),
    "efficientnet_b4": (models.efficientnet_b4, "EfficientNet_B4_Weights", 300),
    "convnext_tiny":   (models.convnext_tiny,   "ConvNeXt_Tiny_Weights",   224),
}


class FaceDataset(Dataset):
    def __init__(self, root: Path, transform, indices=None):
        self.transform = transform
        self.samples = []
        classes = sorted(REMAP.get(d.name, d.name)
                         for d in root.iterdir() if d.is_dir() and d.name not in EXCLUDE)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            if cls_dir.name in EXCLUDE: continue
            idx = self.class_to_idx[REMAP.get(cls_dir.name, cls_dir.name)]
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    self.samples.append((img, idx))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self): return len(self.samples)
    def labels(self): return [y for _, y in self.samples]

    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), y


def split_indices(n, val_frac=0.10, test_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(n * test_frac); n_val = int(n * val_frac)
    return idx[n_test+n_val:].tolist(), idx[:n_val].tolist(), idx[n_val:n_val+n_test].tolist()


def build_model(arch: str, n_classes: int):
    ctor, wname, size = ARCHS[arch]
    weights = getattr(models, wname).IMAGENET1K_V1
    m = ctor(weights=weights)
    # replace the final classifier linear regardless of family
    if arch.startswith("efficientnet"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif arch.startswith("convnext"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m, size


def mixup_cutmix(x, y, n_classes, alpha=0.2, use_cutmix=False):
    """Return mixed inputs and a soft target matrix."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    y1 = F.one_hot(y, n_classes).float()
    y2 = F.one_hot(y[perm], n_classes).float()
    if use_cutmix:
        H, W = x.shape[2], x.shape[3]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0p, y1p = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x0p, x1p = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x[:, :, y0p:y1p, x0p:x1p] = x[perm, :, y0p:y1p, x0p:x1p]
        lam = 1 - ((x1p - x0p) * (y1p - y0p) / (H * W))
    else:
        x = lam * x + (1 - lam) * x[perm]
    target = lam * y1 + (1 - lam) * y2
    return x, target


def soft_ce(logits, target, smoothing=0.1):
    n = logits.size(1)
    target = target * (1 - smoothing) + smoothing / n
    return -(target * F.log_softmax(logits, dim=1)).sum(1).mean()


@torch.no_grad()
def evaluate(model, loader, device, classes, tta=True):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if tta:
            logits = logits + model(torch.flip(x, dims=[3]))  # h-flip average
        ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "report": classification_report(y, p, target_names=classes, zero_division=0, digits=3),
        "confusion": confusion_matrix(y, p).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="resources/datasets/affectnet_faces")
    ap.add_argument("--arch", default="efficientnet_b2", choices=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="app")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, _, INPUT = ARCHS[args.arch]
    print(f"device={device}  arch={args.arch}  input={INPUT}")

    train_tf = transforms.Compose([
        transforms.Resize(int(INPUT * 1.14)),
        transforms.RandomResizedCrop(INPUT, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(0.2, 0.2, 0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(INPUT * 1.14)), transforms.CenterCrop(INPUT),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])

    root = Path(args.data).resolve()
    base = FaceDataset(root, eval_tf)
    classes = base.classes; n = len(base)
    counts = np.bincount(base.labels(), minlength=len(classes))
    print(f"{n:,} images · classes={classes}")
    for c, v in zip(classes, counts): print(f"  {c:<10s} {v:,}")

    tr_idx, va_idx, te_idx = split_indices(n, 0.10, 0.15, args.seed)
    train_ds = FaceDataset(root, train_tf, tr_idx)
    val_ds   = FaceDataset(root, eval_tf,  va_idx)
    test_ds  = FaceDataset(root, eval_tf,  te_idx)
    print(f"split: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    # class-balanced sampler over the TRAIN split
    tr_labels = np.array(train_ds.labels())
    tr_counts = np.bincount(tr_labels, minlength=len(classes))
    samp_w = (1.0 / tr_counts)[tr_labels]
    sampler = WeightedRandomSampler(torch.tensor(samp_w, dtype=torch.double),
                                    num_samples=len(train_ds), replacement=True)

    dl = lambda ds, **k: DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                    pin_memory=True, persistent_workers=True, **k)
    train_loader = dl(train_ds, sampler=sampler)
    val_loader = dl(val_ds, shuffle=False)
    test_loader = dl(test_ds, shuffle=False)

    model, _ = build_model(args.arch, len(classes)); model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - args.warmup)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"emotion_cnn_{args.tag}.pt"
    best_val = 0.0
    nC = len(classes)

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0; seen = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            use_mix = (i % 2 == 0)
            if use_mix:
                x, soft = mixup_cutmix(x, y, nC, alpha=0.2, use_cutmix=(i % 4 == 1))
            opt.zero_grad()
            logits = model(x)
            loss = soft_ce(logits, soft) if use_mix else soft_ce(logits, F.one_hot(y, nC).float())
            loss.backward(); opt.step()
            run += loss.item() * x.size(0); seen += x.size(0)
            if i % 40 == 0:
                print(f"  e{epoch} {i:4d}/{len(train_loader)} loss={loss.item():.3f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        sched.step()
        v = evaluate(model, val_loader, device, classes, tta=False)
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={run/seen:.3f}  "
              f"val_acc={v['accuracy']:.3f} val_f1={v['macro_f1']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if v["accuracy"] > best_val:
            best_val = v["accuracy"]
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "mean": MEAN, "std": STD, "input": INPUT, "arch": args.arch,
                        "val_acc": best_val}, best_path)
            print(f"  -> new best {best_val:.3f}, saved", flush=True)

    # final test with the best checkpoint + TTA
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    t = evaluate(model, test_loader, device, classes, tta=True)
    print("\n===== HELD-OUT TEST (with TTA) =====")
    print(t["report"])
    print(f">>> HEADLINE: test_acc={t['accuracy']:.3f}  macro-F1={t['macro_f1']:.3f}")

    # export ONNX (overwrites the production model + meta)
    onnx_path = out_dir / "emotion_cnn.onnx"
    model.eval()
    torch.onnx.export(model.to("cpu"), torch.randn(1, 3, INPUT, INPUT), str(onnx_path),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"exported -> {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    meta = {
        "classes": classes, "mean": MEAN, "std": STD, "input_size": INPUT,
        "val_acc": best_val, "test_acc": t["accuracy"], "test_macro_f1": t["macro_f1"],
        "n_test": len(test_ds), "confusion": t["confusion"], "report": t["report"],
        "architecture": f"{args.arch} (ImageNet pretrained, fine-tuned)",
        "preprocessing": "MediaPipe face crop + eye-roll align, 256px",
        "training": "WeightedRandomSampler + RandAugment + RandomErasing + mixup/cutmix + label smoothing + TTA eval",
        "trained_on": "AffectNet 6-class (anger, disgust, fear, happiness, sadness, surprise — neutral & contempt excluded)",
    }
    json.dump(meta, open(out_dir / "emotion_cnn_meta.json", "w"), indent=2)
    print(f"meta -> {out_dir / 'emotion_cnn_meta.json'}")


if __name__ == "__main__":
    main()
