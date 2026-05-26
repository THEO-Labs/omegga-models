"""Stage 5 — BBD (Black Box Detector, EfficientNet-B0 NoisyStudent).

Final OK / NOK gate — anything that survived stages 1-4 lands here.

NOTE (2026-05-26): legacy single-class binary model. Multi-Head variant
(marbling / dead / rare) is in training. Once landed, this module's API
will gain `predict_multi(crop_bgr) -> dict[head -> P]` in addition to the
existing `is_bad()` for backwards compat during cutover.

Current weights: `tf_efficientnet_b0.ns_jft_in1k` fine-tuned in Sweep Phase 4c
(cw=3.0, lr=5e-4 on B-binary 27.+29.04.2026). Test F1 0.824, Recall 0.753,
Precision 0.910 at threshold=0.5.

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
