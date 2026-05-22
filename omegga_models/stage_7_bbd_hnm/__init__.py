"""Stage 7 BBD HNM v2 — Black Box Detector with hard-negative-mining.

EfficientNet-B0 NoisyStudent trained 2026-05-22 on 7 days henne-01 (80326
eggs: 1315 pos / 79011 neg, 1692 hard-negatives weighted 5x). Hard-negs
were OK-eggs that the previous BBD wrongly flagged as defect at threshold
0.30 — included in training to teach the model NOT to fire on them.

Single-class sigmoid head, 128px input, per-image normalisation. Different
architecture from the legacy stage_7_bbd module (2-class softmax, 224px).

Test val_auc 0.9011. Reduced FPs by 10-13% vs v3 at operating points
R=80-95%. Threshold 0.33 corresponds to R=80% / P=10.9%.

Public API:
    is_bad_hnm(crop_bgr) -> (bool, float)
        Returns (is_bad, P(defect)). Lazy-loads torch + timm on first call.
"""
from __future__ import annotations

import numpy as np

from . import _classifier as _impl


def is_bad_hnm(crop_bgr: np.ndarray) -> tuple[bool, float]:
    p = _impl.predict_proba(crop_bgr)
    return p >= _impl.THRESHOLD, p


__all__ = ["is_bad_hnm"]
