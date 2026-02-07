"""StyleEncoder: CNN-based style encoder with contrastive learning for font style embeddings.

This module provides a neural network that learns to encode glyph images into
128-dimensional style embeddings. Glyphs from the same font are mapped to similar
embeddings, while glyphs from different fonts are mapped to distant embeddings.

Architecture:
    - Backbone: ResNet-18 or EfficientNet-B0
    - Input: 64x64 grayscale glyph images
    - Output: 128-dimensional normalized embedding
    - Training: Contrastive learning (NT-Xent, Triplet, InfoNCE)

Usage:
    from scripts.models.style_encoder import StyleEncoder, StyleEncoderConfig

    config = StyleEncoderConfig()
    model = StyleEncoder(config)
    embedding = model(glyph_image)  # Returns 128-dim vector
"""

from .config import StyleEncoderConfig, TrainingConfig, default_config
from .model import StyleEncoder, create_style_encoder
from .losses import NTXentLoss, TripletLoss, InfoNCELoss
from .embed import StyleEmbedder

__all__ = [
    "StyleEncoderConfig",
    "TrainingConfig",
    "default_config",
    "StyleEncoder",
    "create_style_encoder",
    "NTXentLoss",
    "TripletLoss",
    "InfoNCELoss",
    "StyleEmbedder",
]

__version__ = "0.1.0"
