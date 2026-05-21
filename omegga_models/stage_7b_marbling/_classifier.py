"""Stage 7b — Marbling-only classifier (EfficientNet-B0, single-class sigmoid).

Architecture (mirrors omegga-ml-training/scripts/train_bbd_v2.py recipe):
- backbone: tf_efficientnet_b0.ns_jft_in1k, num_classes=1 (sigmoid head)
- input: 128x128 RGB, per-image-normalised then ImageNet-normalised
- inference: P(marbling) = sigmoid(logit)

Differs from stage_7_bbd (2-class softmax, 224px, no per-image-norm) — same
backbone but different head and preprocessing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_DEFAULT_CKPT = _HERE / "marbling_v1.pth"
_BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"

# Operating point chosen from sweep on the v1 test set (AUC 0.977):
#   t=0.263  R=81%  P=40.5%   <- production candidate
#   t=0.423  R=71%  P=50.0%   <- conservative
THRESHOLD = 0.263

_INPUT_SIZE = 128
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_MODEL = None
_DEVICE = None


def _per_image_egg_mask(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    cy0, cy1 = h * 1 // 5, h * 4 // 5
    cx0, cx1 = w * 1 // 5, w * 4 // 5
    m[cy0:cy1, cx0:cx1] = True
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    m &= (v >= 50) & (v <= 230)
    return m


def _per_image_normalize(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Subtract egg-pixel mean, divide by egg-pixel std, then squash to [0,255]."""
    img_f = img_rgb_uint8.astype(np.float32) / 255.0
    mask = _per_image_egg_mask(img_rgb_uint8)
    if mask.sum() < 500:
        mu = img_f.reshape(-1, 3).mean(0)
        sd = img_f.reshape(-1, 3).std(0) + 1e-6
    else:
        egg = img_f[mask]
        mu = egg.mean(0)
        sd = egg.std(0) + 1e-6
    out = (img_f - mu) / sd
    out = np.clip(out, -4, 4)
    out = (out + 4) / 8.0
    return (out * 255).astype(np.uint8)


def _preprocess(img_bgr: np.ndarray):
    import torch

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_norm = _per_image_normalize(rgb)
    resized = cv2.resize(rgb_norm, (_INPUT_SIZE, _INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(_IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return tensor


def _load_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE
    import timm
    import torch

    if not _DEFAULT_CKPT.exists():
        raise FileNotFoundError(f"Missing weights: {_DEFAULT_CKPT}")

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(_DEFAULT_CKPT), map_location=_DEVICE, weights_only=False)

    backbone = _BACKBONE
    if isinstance(ckpt, dict) and ckpt.get("model_name"):
        backbone = ckpt["model_name"]

    model = timm.create_model(backbone, pretrained=False, num_classes=1, in_chans=3)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(_DEVICE)
    _MODEL = model
    return _MODEL, _DEVICE


def predict_proba(img_bgr: np.ndarray) -> float:
    import torch

    model, device = _load_model()
    x = _preprocess(img_bgr).to(device)
    with torch.no_grad():
        logit = model(x).squeeze(-1)
        prob = torch.sigmoid(logit).item()
    return float(prob)


def classify_crop(img_bgr: np.ndarray) -> tuple[float, bool]:
    p = predict_proba(img_bgr)
    return p, p >= THRESHOLD
