"""Stage 7b — Only-Marbling Detector (EfficientNet-B0 binary, NoisyStudent).

Single-class CNN trained 2026-05-21 on 6 days henne-01
(09.04/27.04/29.04/30.04/11.05/12.05). 303 positives (defect_marbling, incl
multi-secondary), 67k negatives (OK only). Test AUC 0.977.

This is an alternative to Stage 7 BBD that focuses exclusively on the
marbling defect type — substantially better precision than the multi-class
BBD model at the cost of not catching dead/rare defects.

Public API:
    is_marbling(crop_bgr) -> (bool, float)
        Returns (is_marbling, P(marbling)). Lazy-loads torch + timm on
        first call.
"""
from __future__ import annotations

import numpy as np

from . import _classifier as _impl


def is_marbling(crop_bgr: np.ndarray) -> tuple[bool, float]:
    p = _impl.predict_proba(crop_bgr)
    return p >= _impl.THRESHOLD, p


__all__ = ["is_marbling"]
