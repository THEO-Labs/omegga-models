"""omegga-models — shared inference bundles for the egg-sorting pipeline.

Pipeline stages 1-5 (post 2026-05-26 renumbering):

    from omegga_models.stage_1_dark           import is_too_dark
    from omegga_models.stage_2_anomaly_1      import is_anomaly
    from omegga_models.stage_3_unfertilized   import is_orange
    from omegga_models.stage_4_bubble         import is_bubble_defect  # placeholder
    from omegga_models.stage_5_bbd            import is_bad

Internal utilities (not pipeline-gates):

    from omegga_models.masking                import predict_mask

Stage 4 (bubble) is a placeholder — its public function returns (False, 0.0)
so the pipeline orchestrator can call it unconditionally. Real bubble-pose
detection will land later using the masking utility internally.

Heavy stages (anomaly_1, masking, bbd) lazy-load their dependencies
(torch / open_clip / timm / segmentation_models_pytorch) on first call,
so importing this package is cheap.
"""
from __future__ import annotations

__version__ = "2026.5.26.2"
