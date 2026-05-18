"""Too-Dark detector — L1-LogReg on V/L/S features.

Drop-in replacement for app/pipeline/dark_heuristic.py.
Exposes is_too_dark(crop_bgr) -> (bool, dict) matching the existing signature.

Version: 2026-05-18_logreg_v2_t0p484
Trained: 2026-05-18
Dates: ['2026-04-09', '2026-04-27', '2026-04-29', '2026-04-30', '2026-05-12']
Jetsons: ['henne-01', 'henne-02']
n_too_dark: 10202 (train) + 3909 (test)
n_not_too_dark: 45392 (train) + 15775 (test)
Test AUC: 0.9990
Test AP:  0.9976
Threshold: 0.4840 (Recall >= 99.0% with min FPR (per-video held-out))

Pure numpy + cv2. No sklearn at runtime.
"""
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

_FEATURES = ['V_mean', 'V_std', 'V_p95', 'pct_V_below_20', 'pct_V_below_30', 'pct_V_below_50', 'L_mean', 'L_std', 'L_p95', 'S_mean']
_MEAN = np.array([92.097790, 56.704490, 194.597807, 16.065695, 22.973856, 32.978924, 56.810012, 42.364185, 144.524814, 214.427901], dtype=np.float64)
_STD  = np.array([32.660618, 19.034230, 66.190742, 18.344286, 21.172583, 22.538002, 23.313764, 16.519486, 65.314377, 22.308043], dtype=np.float64)
_COEF = np.array([-0.332182, -6.056979, -0.176027, 0.857362, -1.161299, 2.548901, -2.260461, 2.861020, 0.000000, 0.304777], dtype=np.float64)
_INTERCEPT = -4.902378
_THRESHOLD = 0.4840


def _compute_features(crop_bgr: np.ndarray) -> np.ndarray:
    view = crop_bgr[::4, ::4]
    v = view.max(axis=2)
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1]
    n = v.size
    return np.array([
        float(v.mean()),
        float(v.std()),
        float(np.percentile(v, 95)),
        float((v < 20).sum()) / n * 100,
        float((v < 30).sum()) / n * 100,
        float((v < 50).sum()) / n * 100,
        float(l_chan.mean()),
        float(l_chan.std()),
        float(np.percentile(l_chan, 95)),
        float(s_chan.mean()),
    ], dtype=np.float64)


def is_too_dark(crop_bgr: np.ndarray, gray: Optional[np.ndarray] = None) -> Tuple[bool, dict]:
    if crop_bgr is None or crop_bgr.size == 0:
        return False, {"prob": 0.0, "metric": "logreg"}
    feats = _compute_features(crop_bgr)
    scaled = (feats - _MEAN) / _STD
    logit = float(np.dot(scaled, _COEF) + _INTERCEPT)
    prob = 1.0 / (1.0 + np.exp(-logit))
    is_dark = bool(prob >= _THRESHOLD)
    return is_dark, {"prob": float(prob), "logit": logit, "metric": "logreg"}
