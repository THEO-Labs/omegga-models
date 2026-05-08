"""Stage 6 — Pose (PLACEHOLDER).

No model has shipped yet. Returns (False, 0.0) so the pipeline can
call it unconditionally.
"""
from __future__ import annotations

import numpy as np


def is_bad_pose(crop_bgr: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:  # noqa: ARG001
    """Placeholder — always returns (False, 0.0)."""
    return False, 0.0


__all__ = ["is_bad_pose"]
