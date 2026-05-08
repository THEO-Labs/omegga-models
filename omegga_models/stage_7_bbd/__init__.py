"""Stage 7 — Black Box Detector (EfficientNet-B0 binary).

Final OK / NOK gate — anything that survived stages 1-4 lands here.

Public API:
    is_bad(crop_bgr) -> (bool, float)
        Returns (is_bad, P(defect)). Lazy-loads torch + timm on first call.
"""
from __future__ import annotations

import numpy as np

from . import _classifier as _impl


def is_bad(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """Returns (is_bad, P(defect))."""
    p, flag = _impl.classify_crop(crop_bgr)
    return bool(flag), float(p)


__all__ = ["is_bad"]
