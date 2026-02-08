"""Evaluation framework for Typogenesis ML models.

Modules:
    metrics: FID, style consistency, kerning MAE, retrieval accuracy
    visualize: Sample grids, loss curves, embedding plots
    compare: A/B comparison tooling
"""

from .metrics import (
    compute_fid,
    critical_pair_mae,
    cross_font_similarity,
    evaluate_kerning_net,
    evaluate_style_encoder,
    kerning_direction_accuracy,
    kerning_mae,
    retrieval_accuracy,
    same_font_similarity,
    style_consistency,
)
from .visualize import (
    plot_embedding_space,
    plot_kerning_comparison,
    plot_sample_grid,
    plot_training_curve,
)

__all__ = [
    "same_font_similarity",
    "cross_font_similarity",
    "retrieval_accuracy",
    "kerning_mae",
    "kerning_direction_accuracy",
    "critical_pair_mae",
    "style_consistency",
    "compute_fid",
    "evaluate_style_encoder",
    "evaluate_kerning_net",
    "plot_sample_grid",
    "plot_embedding_space",
    "plot_kerning_comparison",
    "plot_training_curve",
]
