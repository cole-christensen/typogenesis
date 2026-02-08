"""KerningNet: Siamese CNN for predicting optimal kerning values between glyph pairs.

This module provides a neural network that predicts kerning adjustments
for pairs of glyph images using a Siamese architecture with a shared encoder.

Architecture:
    - Shared CNN encoder for left and right glyph images
    - Input: Two 64x64 grayscale glyph images
    - Concatenated embeddings fed through FC regression layers
    - Output: Single kerning value in units per em

Usage:
    from scripts.models.kerning_net import KerningNet, KerningNetConfig

    config = KerningNetConfig()
    model = KerningNet(config)
    kerning = model(left_glyph, right_glyph)  # Returns kerning value
"""

from .model import KerningNet, create_kerning_net, KerningNetConfig

__all__ = [
    "KerningNet",
    "KerningNetConfig",
    "create_kerning_net",
]

__version__ = "0.1.0"
