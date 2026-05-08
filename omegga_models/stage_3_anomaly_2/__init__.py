"""Stage 3 — Anomaly High-Cost (PLACEHOLDER).

No model has shipped yet. The function returns (False, 0.0) so the
pipeline orchestrator can call it unconditionally without crashing.

When the real model lands, replace this file with a real
is_anomaly(crop_bgr) -> (bool, float) implementation.
"""
from __future__ import annotations

import numpy as np


def is_anomaly(crop_bgr: np.ndarray) -> tuple[bool, float]:  # noqa: ARG001
    """Placeholder — always returns (False, 0.0)."""
    return False, 0.0


__all__ = ["is_anomaly"]
