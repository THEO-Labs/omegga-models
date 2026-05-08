"""omegga-models — shared inference bundles for the egg-sorting pipeline.

Each stage is an importable submodule with a stable public API:

    from omegga_models.stage_1_dark import is_too_dark
    from omegga_models.stage_2_anomaly_1 import is_anomaly
    from omegga_models.stage_4_unfertilized import is_orange
    from omegga_models.stage_5_masking import predict_mask
    from omegga_models.stage_7_bbd import is_bad

Stages 3 (anomaly_2) and 6 (pose) are placeholders — their public
functions return False so the pipeline orchestrator can call them
unconditionally without crashing.

Heavy stages (anomaly_1, masking, bbd) lazy-load their dependencies
(torch / open_clip / timm / segmentation_models_pytorch) on first call,
so importing this package is cheap.
"""
from __future__ import annotations

__version__ = "2026.5.8"
