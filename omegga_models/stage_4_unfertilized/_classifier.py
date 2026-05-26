"""Orange (unfertilized) Defect Classifier — CNN v4 (2026-05-22).

EfficientNet-B0 NoisyStudent (4.01M params, 128x128) trained on 8 days of
v2-schema data (94,073 eggs / 4,263 unfertilized across henne-01 + henne-02:
09.04, 27.04, 29.04, 30.04, 07.05, 08.05, 11.05, 12.05). Recipe = same as
BBD-v2 (focal loss alpha=0.25 gamma=2, MixUp 0.2, pos_weight=8, weighted
sampler). Uses **per-image normalization** preprocessing so absolute brightness
drift across days does not shift the score distribution.

Test AUC: 0.9964 (best at epoch 21/30).

Threshold sweep (per-video held-out):
    thr=0.476  R=95.02%  P=86.1%  FP=142  FN=46
    thr=0.603  R=93.07%  P=96.2%  FP= 34  FN=64   <- current default
    thr=0.769  R=90.25%  P=99.3%  FP=  6  FN=90

Default THRESHOLD = 0.60: balanced — R=93% / P=96% / FPR very low. The upstream
black detector + BBD downstream still rescue the residual misses.

Dependencies: numpy + opencv-python + torch + timm. First call lazy-loads
the model (~1-3s on Jetson Orin).

API:
    classify_crop(img_bgr) -> (P(orange), is_orange_bool)
    classify_crops_batch(imgs_bgr) -> list[(P, is_orange_bool)]
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent

# Operating point — v4 balanced (2026-05-22).
# Per-video held-out sweep on 8-day training data:
#   thr=0.476  R=95.02%  P=86.1%  FP=142  FN=46
#   thr=0.603  R=93.07%  P=96.2%  FP= 34  FN=64   <- chosen
#   thr=0.769  R=90.25%  P=99.3%  FP=  6  FN=90
# thr=0.60 picked: R=93% catches almost all unfert, P=96% means very few OK
# eggs leaked to AGAIN bucket.
THRESHOLD = 0.60

# ImageNet RGB stats (model trained with this normalization on top of per-image-norm)
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 128
_BACKBONE = "tf_efficientnet_b0_ns"

_DEFAULT_CKPT = _HERE / "orange_cnn.pth"

_MODEL = None
_DEVICE = None


def _per_image_egg_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of likely-egg pixels (center 60%, V between 50-230)."""
    h, w = img_rgb.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    m[h // 5: h * 4 // 5, w // 5: w * 4 // 5] = True
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    m &= (v >= 50) & (v <= 230)
    return m


def _per_image_norm(img_rgb: np.ndarray) -> np.ndarray:
    """Subtract egg-pixel mean and divide by egg-pixel std, per channel.

    Eliminates absolute brightness drift across days. Falls back to global
    image stats if the egg-pixel mask has <500 pixels.
    """
    f = img_rgb.astype(np.float32) / 255.0
    mask = _per_image_egg_mask(img_rgb)
    if mask.sum() < 500:
        mu = f.reshape(-1, 3).mean(0)
        sd = f.reshape(-1, 3).std(0) + 1e-6
    else:
        egg = f[mask]
        mu = egg.mean(0)
        sd = egg.std(0) + 1e-6
    out = (f - mu) / sd
    out = np.clip(out, -4, 4)
    out = (out + 4) / 8.0
    return (out * 255).astype(np.uint8)


def _preprocess(img_bgr: np.ndarray):
    """BGR uint8 -> torch tensor (1, 3, 128, 128), per-image-norm + ImageNet-norm."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = _per_image_norm(rgb)
    rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE))
    f = rgb.astype(np.float32) / 255.0
    f = (f - _RGB_MEAN) / _RGB_STD
    f = np.transpose(f, (2, 0, 1))
    import torch
    return torch.from_numpy(f).unsqueeze(0)


def _load_model(checkpoint_path: Path | None = None):
    """Build EfficientNet-B0 + load v5 weights. Idempotent."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import timm
    import torch

    ckpt_path = checkpoint_path or _DEFAULT_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing weights: {ckpt_path}")

    _DEVICE = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")

    model = timm.create_model(_BACKBONE, pretrained=False, num_classes=1, in_chans=3)
    ckpt = torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(_DEVICE)

    _MODEL = model
    return _MODEL, _DEVICE


def predict_proba(img_bgr: np.ndarray) -> float:
    """Return P(orange) for a single BGR egg crop."""
    import torch

    model, device = _load_model()
    x = _preprocess(img_bgr).to(device)
    with torch.no_grad():
        logit = model(x).squeeze().item()
    return float(1.0 / (1.0 + np.exp(-logit)))


def classify_crop(img_bgr: np.ndarray) -> tuple[float, bool]:
    """Returns (P(orange), is_orange_at_default_threshold)."""
    p = predict_proba(img_bgr)
    return p, p >= THRESHOLD


def classify_crops_batch(images_bgr: list[np.ndarray]) -> list[tuple[float, bool]]:
    """Batched inference. Faster on GPU."""
    import torch

    if not images_bgr:
        return []
    model, device = _load_model()
    batch = torch.cat([_preprocess(im) for im in images_bgr], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return [(float(p), bool(p >= THRESHOLD)) for p in probs]


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} egg_crop.jpg", file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read {sys.argv[1]}", file=sys.stderr)
        sys.exit(2)
    p, is_orange = classify_crop(img)
    print(f"P(orange) = {p:.4f}  ->  {'ORANGE (unfertilized)' if is_orange else 'OK'}  [threshold={THRESHOLD}]")
