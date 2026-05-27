"""Stage 5 — BBD Multi-Head Classifier (EfficientNet-B0 NoisyStudent, 128px).

Three sigmoid heads on a shared backbone:
    marbling -> P(defect_marbling)
    dead     -> P(defect_dead)
    rare     -> P(defect_rare)

Single forward pass returns all three probabilities. The egg is treated as
defect if ANY head exceeds its threshold.

Trained 2026-05-27 on 8-day v2 data (95.4k eggs across henne-01/02:
626 marbling / 137 dead / 396 rare positives). Test AUCs:
    marbling = 0.9705
    dead     = 0.9939
    rare     = 0.8784
    mean     = 0.9476  (epoch 19 of 30 — best mean)

Default thresholds (per-video held-out sweep, ~21k test eggs):

    HEAD       THR    RECALL   PRECISION   FP    FN
    marbling   0.252  80.0 %   24.3 %      337   27   (balanced)
    dead       0.462  81.5 %   59.5 %       15    5   (balanced)
    rare       0.500  ~30 %    high         few  many (conservative)

The rare-head AUC plateaus around 0.88 because the class is visually
heterogeneous. The conservative threshold trades recall for low FP rate —
adjust upward for fewer FPs, downward for more recall (with steep FP cost).

Dependencies: numpy + opencv-python + torch + timm. First call lazy-loads
the model (~1-3s on Jetson Orin).

API:
    predict_multi(img_bgr) -> dict   {"marbling": float, "dead": float, "rare": float}
    classify_crop(img_bgr) -> (float, bool)
        Returns (max_score_across_heads, is_defect_at_any_default_threshold).
    classify_crops_batch(imgs_bgr) -> list[(float, bool)]
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent

HEADS = ("marbling", "dead", "rare")

# Per-head default thresholds (sweep-tuned for production).
# marbling / dead = balanced operating point.
# rare           = conservative — keeps FPs low at the cost of recall.
THRESHOLDS = {
    "marbling": 0.252,
    "dead":     0.462,
    "rare":     0.500,
}

# Legacy single-threshold alias for code that still expects one number — the
# minimum across all heads ("an egg fires if any head exceeds its threshold,
# at least min(THRESHOLDS) is needed somewhere").
THRESHOLD = min(THRESHOLDS.values())

_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 128
_BACKBONE = "tf_efficientnet_b0_ns"

_DEFAULT_CKPT = _HERE / "bbd_multi_v1.pth"

_MODEL = None
_DEVICE = None


def _per_image_egg_mask(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    m[h // 5: h * 4 // 5, w // 5: w * 4 // 5] = True
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    m &= (v >= 50) & (v <= 230)
    return m


def _per_image_norm(img_rgb: np.ndarray) -> np.ndarray:
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
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = _per_image_norm(rgb)
    rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE))
    f = rgb.astype(np.float32) / 255.0
    f = (f - _RGB_MEAN) / _RGB_STD
    f = np.transpose(f, (2, 0, 1))
    import torch
    return torch.from_numpy(f).unsqueeze(0)


def _load_model(checkpoint_path: Path | None = None):
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

    model = timm.create_model(_BACKBONE, pretrained=False,
                              num_classes=len(HEADS), in_chans=3)
    ckpt = torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(_DEVICE)

    _MODEL = model
    return _MODEL, _DEVICE


def predict_multi(img_bgr: np.ndarray) -> dict:
    """Returns {"marbling": P, "dead": P, "rare": P}."""
    import torch

    model, device = _load_model()
    x = _preprocess(img_bgr).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {h: float(probs[i]) for i, h in enumerate(HEADS)}


def classify_crop(img_bgr: np.ndarray) -> tuple[float, bool]:
    """Returns (max_head_score, is_defect_at_any_default_threshold).

    is_defect is True iff at least one head's score exceeds its own threshold.
    The returned float is the maximum across the three heads — useful for
    sorting / single-channel display.
    """
    scores = predict_multi(img_bgr)
    max_score = max(scores.values())
    is_defect = any(scores[h] >= THRESHOLDS[h] for h in HEADS)
    return max_score, is_defect


def classify_crops_batch(images_bgr: list[np.ndarray]) -> list[tuple[float, bool]]:
    """Batched inference. Faster on GPU."""
    import torch

    if not images_bgr:
        return []
    model, device = _load_model()
    batch = torch.cat([_preprocess(im) for im in images_bgr], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    out = []
    for row in probs:
        scores = {h: float(row[i]) for i, h in enumerate(HEADS)}
        max_score = max(scores.values())
        is_defect = any(scores[h] >= THRESHOLDS[h] for h in HEADS)
        out.append((max_score, is_defect))
    return out


def predict_proba(img_bgr: np.ndarray) -> float:
    """Legacy single-score API — returns max probability across all heads."""
    return max(predict_multi(img_bgr).values())


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} egg_crop.jpg", file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read {sys.argv[1]}", file=sys.stderr)
        sys.exit(2)
    scores = predict_multi(img)
    print(f"marbling = {scores['marbling']:.4f}  (thr={THRESHOLDS['marbling']})")
    print(f"dead     = {scores['dead']:.4f}  (thr={THRESHOLDS['dead']})")
    print(f"rare     = {scores['rare']:.4f}  (thr={THRESHOLDS['rare']})")
    fired = [h for h in HEADS if scores[h] >= THRESHOLDS[h]]
    print(f"-> {'DEFECT (' + ','.join(fired) + ')' if fired else 'OK'}")
