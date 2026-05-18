"""Orange (unfertilized) Defect Classifier — CNN v5.

EfficientNet-B0 (4.01M params, 128x128) trained on 5 days of v2-schema data
(60310 eggs / 2259 unfertilized across henne-01 + henne-02). Uses **per-image
normalization** preprocessing so absolute brightness drift across days does not
shift the score distribution.

Default THRESHOLD = 0.175 (production recall-priority: 99.4% recall, ~10% OK-FP
on the 5-day eval). Per-Stage-4-Pipeline contract the upstream big-black
detector catches the residual FP — so we err on the recall side here.

Dependencies: numpy + opencv-python + torch + timm. First call lazy-loads
the model (~1-3s on Jetson Orin).

API:
    classify_crop(img_bgr) -> (P(orange), is_orange_bool)
    classify_crops_batch(imgs_bgr) -> list[(P, is_orange_bool)]

Trained 2026-05-15, see models/Stage 4/2026-05-17_cnn_efficientnetb0_v5_t0p175
in omegga-ml-training for full performance breakdown and per-day metrics.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent

# Operating point — pipeline-balanced (2026-05-18).
# Re-tuned via sampled cascade evaluation (14.5k eggs, post-black-purge + Stage 2 v2):
#   thr=0.175  Pipeline F1 0.38%  F2 14.15%   (legacy recall-priority)
#   thr=0.30   Pipeline F1 0.43%  F2  8.70%
#   thr=0.40   Pipeline F1 0.48%  F2  7.54%   ← current default
#   thr=0.50   Pipeline F1 0.50%  F2  7.04%
# thr=0.40 picked because F1 stays well under 1% while F2 halves vs the old
# 0.175 default — i.e., we throw away ~half as many OK eggs without missing
# notably more unfertilized eggs.
THRESHOLD = 0.40

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
