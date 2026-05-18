"""Binary OK / Defect Classifier — fully self-contained, single-file.

EfficientNet-B0 fine-tuned on egg crops, binary classification:
    Class 0 = OK (no defect)
    Class 1 = Defect (incl. undetected_defect)

Trained in notebook 7 (omegga-ml-training, transfer learning). Model only sees
quality-passing eggs — overexposed/trash crops MUST be filtered upstream by the
quality-gate classifiers (see `trash_classifier_standalone.py`,
`led_glare_standalone.py`, `too_dark_standalone.py`). Sending a trash/glare
crop to this model yields meaningless predictions.

Drop this file alongside `effnet_binary.pth`. Folder layout:
    dist/effnet_binary/
      ├── effnet_binary_standalone.py   ← this file
      ├── effnet_binary.pth             ← weights
      └── README.md

Dependencies:
    pip install numpy opencv-python pillow torch timm

On Jetson (ARM64+CUDA), install torch from NVIDIA's prebuilt wheel
(<https://forums.developer.nvidia.com/t/pytorch-for-jetson>), then:
    pip install numpy opencv-python pillow timm

Quickstart (CLI):
    python effnet_binary_standalone.py egg_crop.jpg

Quickstart (Library):
    import cv2
    from effnet_binary_standalone import classify_crop
    img = cv2.imread("egg_crop.jpg")             # BGR egg crop, any size
    p, is_defect = classify_crop(img)
    print(f"P(defect) = {p:.3f}  ->  {'DEFECT' if is_defect else 'OK'}")

Per-frame loop on Jetson (with quality gating upstream):
    cap = cv2.VideoCapture(...)
    rois = [(x, y, w, h), ...]
    while True:
        ok, frame = cap.read()
        if not ok: break
        for (x, y, w, h) in rois:
            crop = frame[y:y+h, x:x+w]
            # 1. Quality gate (cheap, numpy/cv2 only)
            if is_glare(crop) or is_too_dark(crop) or is_trash(crop):
                continue
            # 2. Defect classification (~15-30 ms on Orin, batchable)
            p, is_defect = classify_crop(crop)
            if is_defect:
                handle_defect_event(p, ...)

Compute on Jetson Orin (fp16): ~15-25 ms per crop. Batchable for higher
throughput — see `classify_crops_batch()` below.

Architecture:
    Crop (BGR, any size)
      → cv2.cvtColor BGR→RGB
      → PIL.Image.fromarray
      → torchvision.transforms (Resize 224, ToTensor, ImageNet-Normalize)
      → EfficientNet-B0 (timm, num_classes=2)
      → softmax
      → P(defect) = probs[1] >= THRESHOLD
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────
# Operating-point.
#
# Legacy BG-Aug model used 0.405 (picked in notebook 7 to hit α≈0.013/β≈0.012
# on the v1 data distribution). The noisy_student model trained in Phase 4c
# with cw=3.0 reports the following at threshold=0.5 on the B-binary test set:
#   F1=0.824, Recall=0.753, Precision=0.910, α=0.247
# A dedicated threshold-tuning pass for noisy_student is pending (see
# Phase 4-d/Threshold-Tuning entries in TRAINING_RESULTS.md). Until then we
# keep 0.405 as a conservative low-threshold default — it pulls the operating
# point toward higher recall, which matches the recall-priority KPI for the
# Abnahme. Adjust here when fresh threshold sweep numbers are available.
# ──────────────────────────────────────────────────────────────────────────
THRESHOLD = 0.405

# ImageNet RGB stats (model was trained on this normalization)
_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD  = (0.229, 0.224, 0.225)
_INPUT_SIZE = 224

# Default checkpoint location (next to this file).
#
# Stage 7 is on the noisy_student EfficientNet-B0 variant since 2026-05-18
# (Sweep Campaign 2, Phase 4c winner). Earlier weights are kept as fallback:
#   - efficientnet_b0_noisy_student.pth (current default, F1=0.824, R=0.753)
#   - efficientnet_b0_bgaug.pth          (legacy BG-Aug, F1=0.770, R=0.684)
# The state_dicts have the same shape (binary head over EfficientNet-B0
# backbone), but the noisy_student variant requires the
# tf_efficientnet_b0.ns_jft_in1k timm tag for clean loading — see
# _load_model() below.
_DEFAULT_CKPT = _HERE / "efficientnet_b0_noisy_student.pth"
# Backbone tag that matches the checkpoint above. Override via env if you
# need to roll back to the BG-Aug model.
_BACKBONE_NAME = "tf_efficientnet_b0.ns_jft_in1k"


# ──────────────────────────────────────────────────────────────────────────
# Lazy model loading — first call costs ~1-3 s on Jetson
# ──────────────────────────────────────────────────────────────────────────
_MODEL = None
_PREPROCESS = None
_DEVICE = None


def _build_preprocess():
    """torchvision transform: numpy HWC uint8 BGR -> normalized tensor (1,3,224,224)."""
    import torchvision.transforms as T
    from PIL import Image

    base = T.Compose([
        T.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=_RGB_MEAN, std=_RGB_STD),
    ])

    def preprocess(img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return base(pil).unsqueeze(0)

    return preprocess


def _load_model(checkpoint_path: Path | None = None):
    """Build EfficientNet-B0 + load fine-tuned weights. Idempotent — caches the model."""
    global _MODEL, _PREPROCESS, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _DEVICE

    import timm
    import torch

    ckpt_path = checkpoint_path or _DEFAULT_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing weights: {ckpt_path} — copy effnet_binary.pth next to the script")

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)
    # Pick backbone tag from the checkpoint metadata when available so we can
    # transparently load both legacy bgaug ("efficientnet_b0") and the current
    # noisy_student ("tf_efficientnet_b0.ns_jft_in1k") weights.
    if isinstance(ckpt, dict) and ckpt.get("model_name"):
        backbone = ckpt["model_name"]
    else:
        backbone = _BACKBONE_NAME

    model = timm.create_model(backbone, pretrained=False, num_classes=2, in_chans=3)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt  # legacy bare state_dict

    model.load_state_dict(state_dict)
    model.eval().to(_DEVICE)

    _MODEL = model
    _PREPROCESS = _build_preprocess()
    return _MODEL, _PREPROCESS, _DEVICE


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────
def predict_proba(img_bgr: np.ndarray) -> float:
    """Return P(defect) for a single BGR egg crop."""
    import torch

    model, preprocess, device = _load_model()
    x = preprocess(img_bgr).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].cpu().item())


def classify_crop(img_bgr: np.ndarray) -> tuple[float, bool]:
    """Returns (P(defect), is_defect)."""
    p = predict_proba(img_bgr)
    return p, p >= THRESHOLD


def classify_crops_batch(images_bgr: list[np.ndarray]) -> list[tuple[float, bool]]:
    """Batched inference. Faster than per-crop calls on GPU.

    Args:
        images_bgr: list of BGR uint8 arrays (any size)
    Returns:
        list of (P(defect), is_defect) tuples in same order as input
    """
    import torch

    if not images_bgr:
        return []
    model, preprocess, device = _load_model()
    batch = torch.cat([preprocess(im) for im in images_bgr], dim=0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
    return [(float(p), bool(p >= THRESHOLD)) for p in probs]


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _main() -> int:
    if len(sys.argv) < 2:
        print("usage: python effnet_binary_standalone.py <crop.jpg> [...]")
        return 1
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read"); continue
        p, is_defect = classify_crop(img)
        verdict = "DEFECT" if is_defect else "OK    "
        print(f"{verdict}  P={p:.3f}  {Path(path).name}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
