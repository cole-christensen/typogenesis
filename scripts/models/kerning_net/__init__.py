"""
KerningNet - Siamese CNN for Kerning Prediction

This module provides a neural network model for predicting optimal kerning
values between glyph pairs in fonts.

Main components:
- KerningNet: Siamese CNN model architecture
- KerningPredictor: High-level inference interface
- Training utilities for model training

Example usage:
    from models.kerning_net import KerningNet, KerningPredictor
    from models.kerning_net.config import DEFAULT_MODEL_CONFIG

    # Create model
    model = KerningNet(DEFAULT_MODEL_CONFIG)

    # Or use the high-level predictor
    predictor = KerningPredictor(model_path="checkpoints/kerning_net/best_model.pt")
    kerning = predictor.predict_single(left_glyph, right_glyph)
"""

from .config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    get_all_critical_pairs,
    get_negative_kerning_pairs,
    get_zero_kerning_pairs,
    validate_config,
)
from .model import (
    ConvBlock,
    GlyphEncoder,
    KerningNet,
    KerningNetWithAuxiliary,
    RegressionHead,
    create_model,
    load_model,
)
from .predict import (
    KerningPredictor,
    format_csv,
    format_json,
    format_table,
    load_glyph_image,
    verify_model_predictions,
)

__all__ = [
    # Config
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    "InferenceConfig",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_DATASET_CONFIG",
    "DEFAULT_INFERENCE_CONFIG",
    "get_all_critical_pairs",
    "get_negative_kerning_pairs",
    "get_zero_kerning_pairs",
    "validate_config",
    # Model
    "KerningNet",
    "KerningNetWithAuxiliary",
    "ConvBlock",
    "GlyphEncoder",
    "RegressionHead",
    "create_model",
    "load_model",
    # Predict
    "KerningPredictor",
    "load_glyph_image",
    "format_table",
    "format_json",
    "format_csv",
    "verify_model_predictions",
]

__version__ = "0.1.0"
