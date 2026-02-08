"""KerningNet Siamese CNN model architecture.

This module implements a Siamese CNN that predicts optimal kerning values
for pairs of glyph images. The architecture uses a shared encoder to extract
features from both the left and right glyph, then concatenates the embeddings
and passes them through fully connected layers to regress a kerning value.

Architecture:
    1. Shared CNN encoder (applied to both left and right glyph images)
    2. Global Average Pooling
    3. Concatenation of both glyph embeddings
    4. FC regression head -> single kerning value (in units per em)

Example:
    >>> from scripts.models.kerning_net.model import KerningNet, create_kerning_net
    >>>
    >>> model = create_kerning_net()
    >>> left = torch.randn(1, 1, 64, 64)
    >>> right = torch.randn(1, 1, 64, 64)
    >>> kerning = model(left, right)  # Shape: (1, 1)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type alias
Tensor = torch.Tensor


@dataclass
class KerningNetConfig:
    """Configuration for KerningNet model architecture.

    Attributes:
        input_channels: Number of input channels (1 for grayscale).
        image_size: Input image size (height and width, must be square).
        encoder_channels: Channel sizes for each encoder conv block.
        encoder_dim: Dimension of the encoder output embedding per glyph.
        fc_hidden_dim: Hidden dimension of the regression FC layers.
        dropout_rate: Dropout rate in FC layers.
    """

    input_channels: int = 1
    image_size: int = 64
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    encoder_dim: int = 256
    fc_hidden_dim: int = 128
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.input_channels not in (1, 3):
            raise ValueError(f"input_channels must be 1 or 3, got {self.input_channels}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if len(self.encoder_channels) == 0:
            raise ValueError("encoder_channels must not be empty")
        if self.encoder_dim <= 0:
            raise ValueError(f"encoder_dim must be positive, got {self.encoder_dim}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")


class GlyphEncoder(nn.Module):
    """Shared CNN encoder for extracting features from glyph images.

    Processes a single glyph image through a series of convolutional blocks
    with batch normalization and ReLU activation, followed by global average
    pooling to produce a fixed-size embedding.

    Attributes:
        blocks: Sequential convolutional blocks.
        pool: Global average pooling layer.
    """

    def __init__(self, config: KerningNetConfig) -> None:
        """Initialize GlyphEncoder.

        Args:
            config: Model configuration.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = config.input_channels

        for out_ch in config.encoder_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final linear projection to encoder_dim if last conv channel != encoder_dim
        final_ch = config.encoder_channels[-1]
        if final_ch != config.encoder_dim:
            self.proj = nn.Linear(final_ch, config.encoder_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Extract features from a glyph image.

        Args:
            x: Glyph image tensor of shape (batch, channels, height, width).

        Returns:
            Feature embedding of shape (batch, encoder_dim).
        """
        h = self.blocks(x)
        h = self.pool(h)
        h = h.flatten(start_dim=1)
        h = self.proj(h)
        return h


class KerningNet(nn.Module):
    """Siamese CNN for kerning prediction.

    Uses a shared encoder to extract features from both left and right glyph
    images, concatenates the embeddings, and regresses a single kerning value.

    The predicted kerning value is in units per em (typically negative for
    pairs like AV, AW, AT where glyphs should be brought closer together).

    Attributes:
        config: Model configuration.
        encoder: Shared CNN encoder for both glyphs.
        regressor: FC regression head.
    """

    def __init__(self, config: Optional[KerningNetConfig] = None) -> None:
        """Initialize KerningNet.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        super().__init__()
        self.config = config or KerningNetConfig()

        # Shared encoder (Siamese architecture)
        self.encoder = GlyphEncoder(self.config)

        # Regression head: concatenated embeddings -> kerning value
        concat_dim = self.config.encoder_dim * 2
        self.regressor = nn.Sequential(
            nn.Linear(concat_dim, self.config.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.fc_hidden_dim, self.config.fc_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.fc_hidden_dim // 2, 1),
        )

    def forward(
        self,
        left_glyph: Tensor,
        right_glyph: Tensor,
    ) -> Tensor:
        """Predict kerning value for a glyph pair.

        Args:
            left_glyph: Left glyph image of shape (batch, 1, 64, 64).
            right_glyph: Right glyph image of shape (batch, 1, 64, 64).

        Returns:
            Predicted kerning value of shape (batch, 1), in units per em.
        """
        left_features = self.encoder(left_glyph)
        right_features = self.encoder(right_glyph)

        combined = torch.cat([left_features, right_features], dim=1)
        kerning = self.regressor(combined)

        return kerning

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: If True, only count trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_kerning_net(
    config: Optional[KerningNetConfig] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> KerningNet:
    """Factory function to create KerningNet model.

    Args:
        config: Model configuration. If None, uses default configuration.
        checkpoint_path: Path to checkpoint file to load weights from.
        device: Device to load model on. If None, uses CUDA if available.

    Returns:
        Initialized KerningNet model.
    """
    if config is None:
        config = KerningNetConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KerningNet(config)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    config = KerningNetConfig()
    model = create_kerning_net(config)
    print(f"Model parameters: {model.num_parameters():,}")

    # Test forward pass
    batch_size = 4
    left = torch.randn(batch_size, 1, config.image_size, config.image_size)
    right = torch.randn(batch_size, 1, config.image_size, config.image_size)

    with torch.no_grad():
        output = model(left, right)

    print(f"Left input shape: {left.shape}")
    print(f"Right input shape: {right.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted kerning values: {output.squeeze().tolist()}")
    assert output.shape == (batch_size, 1), "Output shape mismatch!"
    print("Forward pass successful!")
