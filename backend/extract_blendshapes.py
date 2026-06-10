"""
Extract MediaPipe 52 blendshape scores from a labelled image folder (AffectNet layout):

    <root>/
      anger/ image*.jpg
      happy/ image*.jpg
      ...

Output one JSONL line per detected face:
    {"label": "happy", "browInnerUp": 0.12, ..., "mouthSmileLeft": 0.82, ...}

The labels match the existing browser app's blendshape names, so training on this output
produces a model.json the live demo can fuse in directly.

Usage:
    python3 backend/extract_blendshapes.py resources/datasets/affectnet data/affectnet_blendshapes.jsonl
"""
import os, sys, json, time, argparse, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Same model the browser app uses (the .task file is the cross-platform bundle).
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return
    print(f"Downloading FaceLandmarker model -> {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  done ({MODEL_PATH.stat().st_size/1e6:.1f} MB)")


def make_landmarker():
    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def list_images(root: Path):
    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name
        for img in cls_dir.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                rows.append((label, img))
    return rows


def load_mp_image(path: Path):
    """Read with PIL, fix EXIF rotation, convert to RGB ndarray, wrap as mp.Image."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.array(im, dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="root of labelled folders (one folder per class)")
    ap.add_argument("out", help="output JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="cap images (0 = no cap, for smoke tests)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_model()
    landmarker = make_landmarker()

    rows = list_images(root)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"{len(rows):,} images across {len(set(r[0] for r in rows))} classes -> {out_path}")

    by_label_total = {}
    for lab, _ in rows:
        by_label_total[lab] = by_label_total.get(lab, 0) + 1
    print("  per class:", by_label_total)

    n_ok = n_noface = n_err = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for label, path in tqdm(rows, smoothing=0.05):
            try:
                img = load_mp_image(path)
                res = landmarker.detect(img)
                if not res.face_blendshapes:
                    n_noface += 1
                    continue
                bs_list = res.face_blendshapes[0]  # list[Category]
                rec = {"label": label}
                for cat in bs_list:
                    # category_name is the camelCase blendshape ("mouthSmileLeft", etc.)
                    rec[cat.category_name] = float(cat.score)
                f.write(json.dumps(rec) + "\n")
                n_ok += 1
            except Exception as e:
                n_err += 1
                if n_err < 5:
                    print(f"  err {path}: {e}", file=sys.stderr)

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min.")
    print(f"  detected: {n_ok:,}   no_face: {n_noface:,}   errors: {n_err:,}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
