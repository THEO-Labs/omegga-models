"""Masking utility — egg cutout (U-Net + MobileNetV3-small).

UTILITY module (no pipeline-stage gate). Produces a binary mask consumed
by Stage 4 Bubble (when implemented) or any future mask-aware classifier.

Public API:
    predict_mask(crop_bgr) -> np.ndarray   # uint8 in {0, 1}, same H/W as input

Lazy-loads torch + segmentation_models_pytorch on first call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_DEFAULT_CKPT = _HERE / "mask_distill_unet_mbv3.pth"
_INPUT_SIZE = 224
_THRESHOLD = 0.5

_MODEL = None
_DEVICE: Optional[str] = None


def _load_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import segmentation_models_pytorch as smp
    import torch

    if not _DEFAULT_CKPT.exists():
        raise FileNotFoundError(
            f"Missing weights: {_DEFAULT_CKPT}"
        )

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(
        encoder_name="mobilenet_v3_small",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state = torch.load(_DEFAULT_CKPT, map_location=_DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval().to(_DEVICE)
    _MODEL = model
    logger.info("masking: loaded %s on %s", _DEFAULT_CKPT.name, _DEVICE)
    return _MODEL, _DEVICE


def predict_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Returns binary mask (uint8 {0,1}) at same H/W as input."""
    import torch

    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    model, device = _load_model()

    h, w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    tensor = torch.from_numpy(arr).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

    mask_small = (probs >= _THRESHOLD).astype(np.uint8)
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


__all__ = ["predict_mask"]
