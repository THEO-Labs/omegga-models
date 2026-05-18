"""Stage 7 — Black Box Detector (EfficientNet-B0 binary, Noisy Student).

Final OK / NOK gate — anything that survived stages 1-4 lands here.

Current weights: `tf_efficientnet_b0.ns_jft_in1k` fine-tuned in Sweep Phase 4c
(cw=3.0, lr=5e-4 on B-binary 27.+29.04.2026). Test F1 0.824, Recall 0.753,
Precision 0.910 at threshold=0.5 — see `_classifier.py` for the production
threshold (currently 0.405 inherited from BG-Aug; refresh pending).

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
