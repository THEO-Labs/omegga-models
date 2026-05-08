"""Orange-Defect Classifier — fully self-contained, single-file.

Drop this file anywhere. Only deps: numpy + opencv-python.

  pip install numpy opencv-python

Quickstart (CLI):

    python orange_classifier_standalone.py path/to/egg_crop.jpg

Quickstart (Library):

    import cv2
    from orange_classifier_standalone import classify_crop

    img = cv2.imread("egg_crop.jpg")            # BGR, any size
    p, is_orange = classify_crop(img)
    print(f"P(orange) = {p:.3f} — {'ORANGE' if is_orange else 'OK'}")

Per-frame loop on Jetson:

    cap = cv2.VideoCapture(...)
    rois = [(x, y, w, h), ...]                  # one tuple per egg position
    while True:
        ok, frame = cap.read()
        if not ok: break
        for (x, y, w, h) in rois:
            crop = frame[y:y+h, x:x+w]
            p, is_orange = classify_crop(crop)
            if is_orange:
                handle_orange_event(p, ...)

Model: L1-Logistic-Regression on 5 hue features, trained on 290 hand-labeled
orange-defect anchors + 300 OK eggs. 10-fold CV AUC ≈ 0.998. Threshold 0.483
gives ~95% recall at ~98% precision.

Inputs are BGR egg-crops (one egg roughly centered in the image, any size).
The classifier first masks the egg via connected-components, then computes
hue/saturation/luminance statistics inside that mask only — background is
ignored.

Compute budget on Jetson Orin: ~1-3 ms per crop including mask + features.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Trained model parameters (frozen — do not edit)
# Source: config/orange_logreg.json, trained in notebooks/16_HardNegativeReview.ipynb
# ──────────────────────────────────────────────────────────────────────────
_FEATURE_NAMES = ("mean_L_egg", "pct_bright_yellow", "yellow_over_red", "pct_red_egg", "mean_S_egg")
_MEAN      = np.array([ 95.90804811, 14.54187189,  4.44522904, 44.47161834, 232.07417196], dtype=np.float64)
_STD       = np.array([ 35.26492685, 15.51395018, 76.18615543, 29.35443110,  31.08181234], dtype=np.float64)
_COEF      = np.array([  3.63428065,  1.42404110,  0.00000000, -2.15723161,   4.20918018], dtype=np.float64)
_INTERCEPT = 0.5233337026178602
THRESHOLD  = 0.483

# ──────────────────────────────────────────────────────────────────────────
# Egg mask: keep only the central connected component of bright pixels.
# ──────────────────────────────────────────────────────────────────────────
def _egg_mask(img_bgr: np.ndarray, *, min_area: int = 500) -> np.ndarray:
    """Boolean mask isolating the primary egg in the crop."""
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


# ──────────────────────────────────────────────────────────────────────────
# Feature extraction. Returns the 5 features in the order the model expects.
# ──────────────────────────────────────────────────────────────────────────
def compute_features(img_bgr: np.ndarray) -> np.ndarray | None:
    """Compute the 5 hue/lum features for a BGR egg crop. Returns None if no egg is visible."""
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask = _egg_mask(img_bgr)
    n_egg = int(mask.sum())
    if n_egg < 100:
        return None

    H_e, S_e, V_e, L_e = H[mask], S[mask], V[mask], gray[mask]

    # OpenCV Hue is 0..179.  red ∈ [0,10] ∪ [170,179] ; yellow ∈ (20,35] ;
    # bright_yellow = saturated bright yellow (orange-defect signature).
    red_mask           = ((H_e <= 10) | (H_e >= 170)) & (S_e > 100)
    yellow_mask        = (H_e > 20)  & (H_e <= 35)    & (S_e > 100)
    bright_yellow_mask = (H_e >= 18) & (H_e <= 35)    & (S_e > 120) & (V_e > 200)

    n_red = int(red_mask.sum())
    return np.array([
        float(L_e.mean()),                              # 0  mean_L_egg
        float(bright_yellow_mask.sum()) / n_egg * 100,  # 1  pct_bright_yellow
        float(yellow_mask.sum())        / max(n_red, 1),# 2  yellow_over_red  (coef=0, harmlos)
        float(n_red)                    / n_egg * 100,  # 3  pct_red_egg
        float(S_e.mean()),                              # 4  mean_S_egg
    ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────
def predict_proba(features: np.ndarray) -> float:
    """Return P(orange) given the 5-feature vector."""
    x_std = (features - _MEAN) / _STD
    z = _INTERCEPT + float(np.dot(_COEF, x_std))
    return 1.0 / (1.0 + math.exp(-z))


def classify_crop(img_bgr: np.ndarray) -> tuple[float, bool]:
    """Returns (P(orange), is_orange).

    P(orange) is 0.0 if no egg is visible in the crop (degenerate input).
    """
    feats = compute_features(img_bgr)
    if feats is None:
        return 0.0, False
    p = predict_proba(feats)
    return p, p >= THRESHOLD


# ──────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────
def _main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.split("Quickstart (CLI):")[1].split("Quickstart (Library):")[0].strip())
        return 1
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        p, is_orange = classify_crop(img)
        verdict = "ORANGE" if is_orange else "OK    "
        print(f"{verdict}  P={p:.3f}  {Path(path).name}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
