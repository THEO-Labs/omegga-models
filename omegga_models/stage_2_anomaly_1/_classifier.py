"""Stage 2 — Anomaly Low-Cost v2 (handcrafted features only).

L1-LogReg on 19 cv2/numpy features (hue + shape). No CLIP, no torch.
Pure numpy + cv2 — Jetson-friendly, ~5-10ms per crop.

Version: v2 — see trash_logreg.json for trained constants.
Trained on all 5 days (henne-01 + henne-02), target = primary=unclassifiable
AND dark_image != 'dark'. Stage 1 (dark) should run first to filter out the
~94% of unclassifiable eggs that are also dark.

Dependencies:
    pip install numpy opencv-python

Public API (via __init__.py):
    is_anomaly(crop_bgr) -> (bool, float)
        Returns (is_trash, P(trash)).

Threshold 0.75 = 84% Recall / 44% Precision / 1.6% FPR on per-video held-out
across all 5 days. Adjust if you want more recall (lower) or fewer FP (higher).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_WEIGHTS_FILE = _HERE / "trash_logreg.json"

_WEIGHTS = None


def _load_weights():
    global _WEIGHTS
    if _WEIGHTS is not None:
        return _WEIGHTS
    with open(_WEIGHTS_FILE) as f:
        w = json.load(f)
    _WEIGHTS = {
        "feature_names": w["feature_names"],
        "mean": np.array(w["scaler_mean"], dtype=np.float64),
        "std": np.array(w["scaler_std"], dtype=np.float64),
        "coef": np.array(w["coef"], dtype=np.float64),
        "intercept": float(w["intercept"]),
        "threshold": float(w["threshold"]),
        "version": w.get("version", "unknown"),
    }
    return _WEIGHTS


def _egg_mask(img_bgr: np.ndarray, min_area: int = 500) -> np.ndarray:
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


def compute_features(img_bgr: np.ndarray) -> np.ndarray:
    """Compute the 19-dim feature vector in the exact order the v2 model expects."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H_, W_ = gray.shape
    edges = cv2.Canny(gray, 50, 150)

    mean_S = float(hsv[..., 1].mean())
    mean_L = float(gray.mean())
    pct_white = float((gray > 230).sum()) / (H_ * W_) * 100
    edge_density = float(edges.sum()) / (H_ * W_) / 255.0 * 100

    mask = _egg_mask(img_bgr)
    n_egg = int(mask.sum())
    if n_egg < 100:
        n_egg_px = 0
        mean_S_egg = mean_L_egg = pct_yellow = pct_bright_yellow = 0.0
        pct_red_egg = yellow_over_red = 0.0
    else:
        H = hsv[..., 0][mask]; S = hsv[..., 1][mask]
        V = hsv[..., 2][mask]; L = gray[mask]
        red = ((H <= 10) | (H >= 170)) & (S > 100)
        yellow = (H > 20) & (H <= 35) & (S > 100)
        bright_yellow = (H >= 18) & (H <= 35) & (S > 120) & (V > 200)
        n_red = int(red.sum())
        n_egg_px = n_egg
        mean_S_egg = float(S.mean())
        mean_L_egg = float(L.mean())
        pct_yellow = float(yellow.sum()) / n_egg * 100
        pct_bright_yellow = float(bright_yellow.sum()) / n_egg * 100
        pct_red_egg = float(n_red) / n_egg * 100
        yellow_over_red = float(yellow.sum()) / max(n_red, 1)

    # Shape features
    n_px = int(mask.sum())
    if n_px < 50:
        coverage = aspect = extent = solidity = circularity = 0.0
        n_components = 0; cx_off = cy_off = 1.0
    else:
        m8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            coverage = aspect = extent = solidity = circularity = 0.0
            n_components = 0; cx_off = cy_off = 1.0
        else:
            cnt = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            perim = float(cv2.arcLength(cnt, True))
            x, y, ww, hh = cv2.boundingRect(cnt)
            aspect = max(ww, hh) / max(min(ww, hh), 1)
            extent = area / max(ww * hh, 1)
            hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
            solidity = area / max(hull_area, 1.0)
            circularity = 4 * math.pi * area / max(perim * perim, 1.0)
            coverage = n_px / max(H_ * W_, 1)
            n_components_raw, _ = cv2.connectedComponents(m8)
            n_components = int(n_components_raw - 1)
            M_ = cv2.moments(m8)
            if M_["m00"] > 0:
                cxx = M_["m10"] / M_["m00"]; cyy = M_["m01"] / M_["m00"]
                cx_off = abs(cxx - W_ / 2) / (W_ / 2)
                cy_off = abs(cyy - H_ / 2) / (H_ / 2)
            else:
                cx_off = cy_off = 1.0

    return np.array([
        mean_S, mean_L, pct_white, edge_density,
        n_egg_px, mean_S_egg, mean_L_egg,
        pct_yellow, pct_bright_yellow, pct_red_egg, yellow_over_red,
        coverage, aspect, extent, solidity, circularity,
        n_components, cx_off, cy_off,
    ], dtype=np.float64)


def predict_proba(img_bgr: np.ndarray) -> float:
    """Return P(anomaly) for a single BGR egg crop."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    w = _load_weights()
    feats = compute_features(img_bgr)
    scaled = (feats - w["mean"]) / w["std"]
    logit = float(np.dot(scaled, w["coef"]) + w["intercept"])
    return 1.0 / (1.0 + np.exp(-logit))


def classify_crop(img_bgr: np.ndarray) -> Tuple[float, bool]:
    """Returns (P(anomaly), is_anomaly)."""
    p = predict_proba(img_bgr)
    return p, bool(p >= _load_weights()["threshold"])


def classify_crops_batch(images_bgr: list) -> list:
    return [classify_crop(im) for im in images_bgr]


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} egg_crop.jpg", file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read {sys.argv[1]}", file=sys.stderr)
        sys.exit(2)
    p, flag = classify_crop(img)
    w = _load_weights()
    print(f"P(anomaly) = {p:.4f}  ->  {'ANOMALY' if flag else 'OK'}  [threshold={w['threshold']}, version={w['version']}]")
