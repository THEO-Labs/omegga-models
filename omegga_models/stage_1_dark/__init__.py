"""Stage 1 — Dark Heuristic.

L1-LogReg on V/L/S brightness features. Pure numpy + cv2.

Public API:
    is_too_dark(crop_bgr) -> (bool, dict)
"""
from __future__ import annotations

from ._logreg import is_too_dark

__all__ = ["is_too_dark"]
