"""Trash/Overexposed Defect Classifier — fully self-contained, single-file.

Drop this file alongside `trash_logreg.json` + `trash_pca.npz` (both produced by
notebook 17 export). Detects eggs that need manual re-review — overexposed,
trash, or otherwise unclassifiable shells.

Dependencies:
    pip install numpy opencv-python torch open-clip-torch pillow

On Jetson (ARM64 + CUDA), install torch from NVIDIA's prebuilt wheels — see
https://forums.developer.nvidia.com/t/pytorch-for-jetson  (e.g. JetPack 5.x:
torch-2.0.0+nv23.05).

Quickstart (CLI):

    python trash_classifier_standalone.py path/to/egg_crop.jpg

Quickstart (Library):

    import cv2
    from trash_classifier_standalone import classify_crop

    img = cv2.imread("egg_crop.jpg")            # BGR egg-crop, any size
    p, is_trash = classify_crop(img)
    print(f"P(trash) = {p:.3f} — {'TRASH' if is_trash else 'OK'}")

Per-frame loop on Jetson:

    cap = cv2.VideoCapture(...)
    rois = [(x, y, w, h), ...]
    while True:
        ok, frame = cap.read()
        if not ok: break
        for (x, y, w, h) in rois:
            crop = frame[y:y+h, x:x+w]
            p, is_trash = classify_crop(crop)
            if is_trash:
                handle_trash_event(p, ...)

Architecture:
    Crop (BGR)
      ├─→ Hue/Lum features (11) — OpenCV HSV stats inside connected-component egg mask
      ├─→ Shape features (8)   — OpenCV contour analysis on the same mask
      └─→ CLIP ViT-B/32 embed (512) → PCA-30 (precomputed)
                                 ↓
                       43-feature vector
                                 ↓
                       StandardScaler (precomputed mean/std)
                                 ↓
                       LogReg (precomputed coef + intercept)
                                 ↓
                       sigmoid → P(trash) ≥ threshold = trash

Compute budget (Jetson Orin, 256×256 crop, fp16):
    feature extraction (cv2)  ~3-5 ms
    CLIP encode_image          ~12-25 ms
    PCA + LogReg               <1 ms
    total                      ~15-30 ms / crop
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────
# Load trained params + PCA next to this script
# ──────────────────────────────────────────────────────────────────────────
def _load_artifacts():
    params_path = _HERE / "trash_logreg.json"
    pca_path    = _HERE / "trash_pca.npz"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path} — copy it next to the script")
    if not pca_path.exists():
        raise FileNotFoundError(f"Missing {pca_path} — copy it next to the script")
    params = json.loads(params_path.read_text())
    pca = np.load(pca_path)
    return params, pca["components"], pca["mean"]

_PARAMS, _PCA_COMPONENTS, _PCA_MEAN = _load_artifacts()
THRESHOLD       = float(_PARAMS["threshold"])
_FEATURE_NAMES  = list(_PARAMS["feature_names"])
_HUE_FEATURES   = list(_PARAMS["hue_features"])
_SHAPE_FEATURES = list(_PARAMS["shape_features"])
_CLIP_FEATURES  = list(_PARAMS["clip_features"])
_LR_MEAN        = np.asarray(_PARAMS["mean"],     dtype=np.float64)
_LR_STD         = np.asarray(_PARAMS["std"],      dtype=np.float64)
_LR_COEF        = np.asarray(_PARAMS["coef"],     dtype=np.float64)
_LR_INTERCEPT   = float(_PARAMS["intercept"])

# ──────────────────────────────────────────────────────────────────────────
# CLIP (lazy-loaded — first call costs ~1-2s on Jetson)
# ──────────────────────────────────────────────────────────────────────────
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None

def _load_clip():
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    import torch
    import open_clip
    _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(_CLIP_DEVICE)
    _CLIP_MODEL, _CLIP_PREPROCESS = model, preprocess
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def _clip_embed(img_bgr: np.ndarray) -> np.ndarray:
    """Return 512-dim CLIP ViT-B/32 image embedding for a BGR image."""
    import torch
    from PIL import Image
    model, preprocess, device = _load_clip()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    with torch.no_grad():
        x = preprocess(pil).unsqueeze(0).to(device)
        v = model.encode_image(x).cpu().numpy()[0]
    return v.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────
# Egg mask + hue + shape features (mirrors notebook 17 + scripts/hard_negative_mining.py)
# ──────────────────────────────────────────────────────────────────────────
def _egg_mask(img_bgr: np.ndarray, *, min_area: int = 500) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bin_mask = (gray > 40).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    center_label = int(labels[cy, cx])
    if center_label != 0 and stats[center_label, cv2.CC_STAT_AREA] >= min_area:
        return labels == center_label
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0 or areas.max() < min_area:
        return np.zeros((h, w), dtype=bool)
    return labels == 1 + int(areas.argmax())


def _hue_stats(img_bgr: np.ndarray) -> dict:
    """Hue/lum features. Same definition as scripts/hard_negative_mining.py:compute_crop_stats."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)

    out = {
        "mean_S":       float(hsv[..., 1].mean()),
        "mean_L":       float(gray.mean()),
        "pct_white":    float((gray > 230).sum()) / (h * w) * 100,
        "edge_density": float(edges.sum()) / (h * w) / 255.0 * 100,
    }

    mask = _egg_mask(img_bgr)
    n_egg = int(mask.sum())
    if n_egg < 100:
        out.update({
            "n_egg_px": 0, "mean_S_egg": 0.0, "mean_L_egg": 0.0,
            "pct_yellow": 0.0, "pct_bright_yellow": 0.0,
            "pct_red_egg": 0.0, "yellow_over_red": 0.0,
        })
        return out

    H = hsv[..., 0][mask]; S = hsv[..., 1][mask]; V = hsv[..., 2][mask]; L = gray[mask]
    red    = ((H <= 10) | (H >= 170)) & (S > 100)
    yellow = (H > 20) & (H <= 35) & (S > 100)
    bright_yellow = (H >= 18) & (H <= 35) & (S > 120) & (V > 200)
    n_red = int(red.sum())

    out.update({
        "n_egg_px":          n_egg,
        "mean_S_egg":        float(S.mean()),
        "mean_L_egg":        float(L.mean()),
        "pct_yellow":        float(yellow.sum()) / n_egg * 100,
        "pct_bright_yellow": float(bright_yellow.sum()) / n_egg * 100,
        "pct_red_egg":       float(n_red) / n_egg * 100,
        "yellow_over_red":   float(yellow.sum()) / max(n_red, 1),
    })
    return out


def _shape_stats(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Shape features. Same definition as notebook 17:compute_shape_stats."""
    H, W = img_bgr.shape[:2]
    n_px = int(mask.sum())
    if n_px < 50:
        return {
            "shape_coverage": 0.0, "shape_aspect": 0.0, "shape_extent": 0.0,
            "shape_solidity": 0.0, "shape_circularity": 0.0,
            "shape_n_components": 0, "shape_cx_off": 1.0, "shape_cy_off": 1.0,
        }
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "shape_coverage": 0.0, "shape_aspect": 0.0, "shape_extent": 0.0,
            "shape_solidity": 0.0, "shape_circularity": 0.0,
            "shape_n_components": 0, "shape_cx_off": 1.0, "shape_cy_off": 1.0,
        }
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = max(w, h) / max(min(w, h), 1)
    extent = area / max(w * h, 1)
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = area / max(hull_area, 1.0)
    circularity = 4 * math.pi * area / max(perim * perim, 1.0)
    coverage = n_px / max(H * W, 1)
    n_components, _ = cv2.connectedComponents(m8)
    M = cv2.moments(m8)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
        cx_off = abs(cx - W / 2) / (W / 2)
        cy_off = abs(cy - H / 2) / (H / 2)
    else:
        cx_off = cy_off = 1.0
    return {
        "shape_coverage":     float(coverage),
        "shape_aspect":       float(aspect),
        "shape_extent":       float(extent),
        "shape_solidity":     float(solidity),
        "shape_circularity":  float(circularity),
        "shape_n_components": int(n_components - 1),
        "shape_cx_off":       float(cx_off),
        "shape_cy_off":       float(cy_off),
    }


# ──────────────────────────────────────────────────────────────────────────
# Full feature pipeline + prediction
# ──────────────────────────────────────────────────────────────────────────
def compute_features(img_bgr: np.ndarray) -> np.ndarray:
    """Build the 43-dim feature vector in the exact order the model expects."""
    hue   = _hue_stats(img_bgr)
    mask  = _egg_mask(img_bgr)
    shape = _shape_stats(img_bgr, mask)
    clip  = _clip_embed(img_bgr)
    clip_pcs = (clip - _PCA_MEAN) @ _PCA_COMPONENTS.T   # (30,)

    by_name = {**hue, **shape}
    for i, n in enumerate(_CLIP_FEATURES):
        by_name[n] = float(clip_pcs[i])

    return np.array([by_name.get(n, 0.0) or 0.0 for n in _FEATURE_NAMES], dtype=np.float64)


def predict_proba(features: np.ndarray) -> float:
    x_std = (features - _LR_MEAN) / _LR_STD
    z = _LR_INTERCEPT + float(np.dot(_LR_COEF, x_std))
    return 1.0 / (1.0 + math.exp(-z))


def classify_crop(img_bgr: np.ndarray) -> tuple[float, bool]:
    """Returns (P(trash), is_trash)."""
    feats = compute_features(img_bgr)
    p = predict_proba(feats)
    return p, p >= THRESHOLD


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _main() -> int:
    if len(sys.argv) < 2:
        print("usage: python trash_classifier_standalone.py <crop.jpg> [...]")
        return 1
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read"); continue
        p, is_trash = classify_crop(img)
        verdict = "TRASH" if is_trash else "OK   "
        print(f"{verdict}  P={p:.3f}  {Path(path).name}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
