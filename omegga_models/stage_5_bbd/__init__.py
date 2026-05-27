"""Stage 5 — BBD (Black Box Detector, Multi-Head, EfficientNet-B0 NoisyStudent).

Final OK / NOK gate — anything that survived stages 1-4 lands here.

Trained 2026-05-27 as multi-head with three sigmoid outputs:
    marbling (AUC 0.9705)  -> P(defect_marbling)
    dead     (AUC 0.9939)  -> P(defect_dead)
    rare     (AUC 0.8784)  -> P(defect_rare)

Mean AUC 0.9476 (best of 30 epochs on per-video held-out test set).
Replaces the legacy single-class binary BBD (was AUC 0.90) since 2026-05-27.

Public API:
    predict_multi(crop_bgr) -> {"marbling": float, "dead": float, "rare": float}
        Lazy-loads torch + timm on first call. Single forward pass.

    is_bad(crop_bgr) -> (bool, float)
        Backwards-compat: returns (is_defect_at_any_default_threshold, max_score).

    THRESHOLDS  dict[head -> float]
        Default per-head thresholds (marbling=0.252, dead=0.462, rare=0.500).
        Callers may override by computing per-head thresholds against
        predict_multi() output directly.
"""
from __future__ import annotations

import numpy as np

from . import _classifier as _impl

HEADS = _impl.HEADS
THRESHOLDS = _impl.THRESHOLDS


def predict_multi(crop_bgr: np.ndarray) -> dict:
    """Returns {head_name: probability} for all 3 heads (single forward pass)."""
    return _impl.predict_multi(crop_bgr)


def is_bad(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """Returns (is_defect, max_score). is_defect is True iff at least one head
    exceeds its default threshold. Kept for backwards-compat with the legacy
    single-head BBD API."""
    p, flag = _impl.classify_crop(crop_bgr)
    return bool(flag), float(p)


__all__ = ["predict_multi", "is_bad", "HEADS", "THRESHOLDS"]
