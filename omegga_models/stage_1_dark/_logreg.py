"""Too-Dark detector — L1-LogReg on V/L/S features.

Drop-in replacement for app/pipeline/dark_heuristic.py.
Exposes is_too_dark(crop_bgr) -> (bool, dict) matching the existing signature.

Version: 2026-05-08_logreg_t0p57
Trained: 2026-05-08
Dates: ['2026-04-27', '2026-04-29']
n_too_dark: 2002
n_not_too_dark: 32982
Test AUC: 0.9989
Test AP: 0.9703
Threshold: 0.5680 (Recall >= 99.5% with min FPR)

Pure numpy + cv2. No sklearn at runtime.
"""
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

_FEATURES = ['V_mean', 'V_std', 'V_p95', 'pct_V_below_20', 'pct_V_below_30', 'pct_V_below_50', 'L_mean', 'L_std', 'L_p95', 'S_mean']
_MEAN = np.array([96.894110, 60.822402, 209.264800, 14.757736, 21.240465, 29.870590, 62.105662, 47.435095, 162.920792, 208.918260], dtype=np.float64)
_STD  = np.array([27.006142, 16.154284, 55.847290, 16.925221, 20.073497, 20.101572, 19.668222, 14.352788, 60.445540, 21.801283], dtype=np.float64)
_COEF = np.array([6.119338, -2.600125, -0.702123, -0.341421, -1.003922, -1.471393, -11.754046, 2.938669, 0.298177, -2.269262], dtype=np.float64)
_INTERCEPT = -6.071730
_THRESHOLD = 0.5680


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
