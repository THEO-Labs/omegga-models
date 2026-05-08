"""Stage 2 — Anomaly Low-Cost (CLIP + LogReg trash classifier).

Public API:
    is_anomaly(crop_bgr) -> (bool, float)
        Returns (is_trash, P(trash)). Lazy-loads torch + open_clip
        on first call.

Bundle files (loaded relative to this package):
    trash_logreg.json — scaler + LogReg coef + threshold
    trash_pca.npz     — PCA components for CLIP-512 → 30 features
"""
from __future__ import annotations

import numpy as np

# The standalone module loads JSON/npz from its own directory using
# Path(__file__).parent. That works fine when installed as a package.
from . import _classifier as _impl


def is_anomaly(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """Returns (is_trash, P(trash))."""
    p, flag = _impl.classify_crop(crop_bgr)
    return bool(flag), float(p)


__all__ = ["is_anomaly"]
