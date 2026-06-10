"""
Face-crop + light-align preprocessing for the AffectNet folders.

The raw AffectNet images are loose crops (background, neck, hair, jewelry all eat pixels).
A tight, consistent face crop concentrates the model's capacity on the expressive region —
the single biggest accuracy lever before changing the model.

Pipeline per image:
  1. MediaPipe FaceLandmarker (478 pts) -> face bounding box
  2. roll-align: rotate so the eye line is horizontal (small but free gain)
  3. expand bbox to a square with margin, crop (edge-pad if it runs off-frame)
  4. resize to --size (default 256) and save mirroring the class-folder layout

Usage:
    python3 backend/crop_faces.py resources/datasets/affectnet resources/datasets/affectnet_faces
"""
import argparse, math, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# FaceMesh landmark indices for eye-corner alignment.
L_EYE = [33, 133]    # left-eye outer/inner corners
R_EYE = [362, 263]   # right-eye inner/outer corners


def make_landmarker():
    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1, output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def list_images(root: Path):
    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                rows.append((cls_dir.name, img))
    return rows


def crop_one(pil: Image.Image, landmarker, size: int, margin: float):
    """Return a cropped+aligned PIL image, or None if no face."""
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    W, H = pil.size
    arr = np.asarray(pil, dtype=np.uint8)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    xs = np.array([p.x for p in lm]) * W
    ys = np.array([p.y for p in lm]) * H

    # roll alignment from eye line
    le = np.array([xs[L_EYE].mean(), ys[L_EYE].mean()])
    re = np.array([xs[R_EYE].mean(), ys[R_EYE].mean()])
    angle = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    cx, cy = float(xs.mean()), float(ys.mean())

    if abs(angle) > 1.0:
        pil = pil.rotate(angle, center=(cx, cy), resample=Image.BILINEAR)
        # rotate the landmark coords too (about center) so the bbox stays valid
        th = math.radians(angle)
        cos, sin = math.cos(th), math.sin(th)
        dx, dy = xs - cx, ys - cy
        xs = cos * dx + sin * dy + cx
        ys = -sin * dx + cos * dy + cy

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    side = max(x1 - x0, y1 - y0) * (1.0 + margin)
    bx0 = int(round(cx - side / 2)); by0 = int(round(cy - side / 2))
    bx1 = int(round(cx + side / 2)); by1 = int(round(cy + side / 2))

    # crop with edge padding if the box runs off-frame
    pad_l = max(0, -bx0); pad_t = max(0, -by0)
    pad_r = max(0, bx1 - pil.size[0]); pad_b = max(0, by1 - pil.size[1])
    if pad_l or pad_t or pad_r or pad_b:
        pil = ImageOps.expand(pil, border=(pad_l, pad_t, pad_r, pad_b), fill=(0, 0, 0))
        bx0 += pad_l; bx1 += pad_l; by0 += pad_t; by1 += pad_t
    face = pil.crop((bx0, by0, bx1, by1)).resize((size, size), Image.BILINEAR)
    return face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.40, help="bbox expansion beyond the face")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src).resolve(); dst = Path(args.dst).resolve()
    landmarker = make_landmarker()
    rows = list_images(src)
    if args.limit: rows = rows[:args.limit]
    print(f"{len(rows):,} images -> {dst}")

    n_ok = n_noface = n_err = 0
    for cls, path in tqdm(rows, smoothing=0.05):
        out_dir = dst / cls; out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path.stem + ".jpg")
        try:
            with Image.open(path) as im:
                face = crop_one(im, landmarker, args.size, args.margin)
            if face is None:
                n_noface += 1; continue
            face.save(out_path, "JPEG", quality=92)
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err < 5: print(f"  err {path}: {e}", file=sys.stderr)

    print(f"\ncropped: {n_ok:,}   no_face: {n_noface:,}   errors: {n_err:,}")


if __name__ == "__main__":
    main()
