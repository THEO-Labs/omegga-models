"""Too-Dark detector — L1-LogReg on V/L/S features.

Drop-in replacement for app/pipeline/dark_heuristic.py.
Exposes is_too_dark(crop_bgr) -> (bool, dict) matching the existing signature.

Version: 2026-05-22_logreg_v3_t0p345
Trained: 2026-05-22
Dates: ['2026-04-09', '2026-04-27', '2026-04-29', '2026-04-30', '2026-05-07',
        '2026-05-08', '2026-05-11', '2026-05-12']
Jetsons: ['henne-01', 'henne-02']
n_too_dark: 15664 (train) + 5577 (test)
n_not_too_dark: 76440 (train) + 25749 (test)
Test AUC: 0.9998
Test AP:  0.9994
Threshold: 0.3445 (Recall >= 99.5% with min FPR (per-video held-out))

Pure numpy + cv2. No sklearn at runtime.
"""
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

_FEATURES = ['V_mean', 'V_std', 'V_p95', 'pct_V_below_20', 'pct_V_below_30', 'pct_V_below_50', 'L_mean', 'L_std', 'L_p95', 'S_mean']
_MEAN = np.array([93.136043, 57.457108, 196.793383, 17.471206, 24.350954, 33.508217, 57.840718, 43.165159, 147.470126, 213.192446], dtype=np.float64)
_STD  = np.array([34.108737, 19.957500, 68.599574, 20.844971, 23.884122, 24.288883, 23.673221, 16.815659, 66.091960, 23.749525], dtype=np.float64)
_COEF = np.array([0.000000, -8.549931, -0.346145, 1.713043, 0.000000, 3.108595, -1.735229, 2.741589, 0.575815, 0.919123], dtype=np.float64)
_INTERCEPT = -4.897093
_THRESHOLD = 0.3445


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
