"""Stage 4 — Bubble (Luftblasen-Defekt-Erkennung, PLACEHOLDER).

No model has shipped yet. The future implementation will:
1. Use the masking utility to segment egg + air bubble
2. Extract bubble pose (centroid, size, orientation relative to egg axis)
3. Decide rule-based whether the bubble position/size indicates a defect
   (e.g., bubble at wrong end -> egg laid upside-down)

Until then, this stage returns (False, 0.0) so the pipeline orchestrator
can call it unconditionally.

Public API:
    is_bubble_defect(crop_bgr) -> (bool, float)
        Returns (is_defect, confidence).
"""
from __future__ import annotations

import numpy as np


def is_bubble_defect(crop_bgr: np.ndarray) -> tuple[bool, float]:  # noqa: ARG001
    """Placeholder — always returns (False, 0.0)."""
    return False, 0.0


__all__ = ["is_bubble_defect"]
