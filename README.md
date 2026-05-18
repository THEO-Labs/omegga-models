# omegga-models

Shared inference bundles for the omegga egg-sorting pipeline. Each of the
seven pipeline stages is an importable submodule with a stable public API.

Consumed by:
- `omegga-ml-training` (notebooks evaluate models)
- `omegga-labelling-tool` (auto-tag features)
- `omegga-production/henne` (live inference on the Jetson)

## Install

```bash
# Lightweight (stages 1, 3, 4, 6 only — pure numpy + cv2)
pip install "omegga-models @ git+ssh://git@github.com/THEO-Labs/omegga-models.git@main"

# Full inference incl. NN stages 2 / 5 / 7
pip install "omegga-models[ml] @ git+ssh://git@github.com/THEO-Labs/omegga-models.git@main"
```

Pin to a tag in production:

```toml
"omegga-models[ml] @ git+ssh://git@github.com/THEO-Labs/omegga-models.git@2026.5.8"
```

## Public API per stage

| Stage | Module | Function | Verdict |
|-------|--------|----------|---------|
| 1 | `omegga_models.stage_1_dark` | `is_too_dark(crop) -> (bool, dict)` | AGAIN |
| 2 | `omegga_models.stage_2_anomaly_1` | `is_anomaly(crop) -> (bool, float)` | AGAIN |
| 3 | `omegga_models.stage_3_anomaly_2` | `is_anomaly(crop) -> (bool, float)` | AGAIN (placeholder) |
| 4 | `omegga_models.stage_4_unfertilized` | `is_orange(crop) -> (bool, float)` | NOK |
| 5 | `omegga_models.stage_5_masking` | `predict_mask(crop) -> np.ndarray` | utility |
| 6 | `omegga_models.stage_6_pose` | `is_bad_pose(crop, mask) -> (bool, float)` | NOK (placeholder) |
| 7 | `omegga_models.stage_7_bbd` | `is_bad(crop) -> (bool, float)` | NOK |

`crop` is always BGR `np.ndarray` of arbitrary size (typically 740×890 in
production).

## Adding a new model version

1. Train in `omegga-ml-training` (e.g. notebook 24 for stage 1).
2. Copy the standalone `.py` (and any JSON/.pth/.npz weights) into
   `omegga-models/omegga_models/stage_<N>_<name>/`. Keep the `_<noun>.py`
   filename so the existing `__init__.py` re-export stays valid.
3. Bump `__version__` in `omegga_models/__init__.py` and `version` in
   `pyproject.toml` (CalVer: `YYYY.M.D`).
4. Tag and push:
   ```bash
   git tag 2026.5.8
   git push origin main --tags
   ```
5. In each consumer (`omegga-labelling-tool`, `production/henne`), bump the
   pinned version in `pyproject.toml` / `requirements.txt` and redeploy.
   For "always latest" deployments (Docker rebuild), just pinning to `@main`
   gets the new code on every build.

## Stage status

- ✅ Stage 1 — L1-LogReg on V/L/S features (notebook 24, 2026-05-08)
- ✅ Stage 2 — CLIP + LogReg (notebook 17)
- ❌ Stage 3 — placeholder, returns `False`
- ✅ Stage 4 — Hue-LogReg (5 features)
- ✅ Stage 5 — U-Net mobilenet_v3_small (notebook 22)
- ❌ Stage 6 — placeholder, returns `False`
- ✅ Stage 7 — EfficientNet-B0 binary, **Noisy-Student weights** (Sweep Phase 4c, 2026-05-18)
  - Backbone: `tf_efficientnet_b0.ns_jft_in1k`, cw=3.0, lr=5e-4
  - Test F1 0.824, Recall 0.753, Precision 0.910 (vs legacy BG-Aug: F1 0.770, R 0.684)
  - Legacy `efficientnet_b0_bgaug.pth` is kept in the package for rollback
