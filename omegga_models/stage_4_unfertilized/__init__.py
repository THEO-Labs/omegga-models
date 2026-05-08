"""Stage 4 — Unfertilized (orange) defect detector.

L1-LogReg on 5 hue features. Pure numpy + cv2.

Public API:
    is_orange(crop_bgr) -> (bool, float)
        Returns (is_orange, P(orange)).
"""
from __future__ import annotations

import numpy as np

from . import _classifier as _impl


def is_orange(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """Returns (is_orange, P(orange))."""
    p, flag = _impl.classify_crop(crop_bgr)
    return bool(flag), float(p)


__all__ = ["is_orange"]
