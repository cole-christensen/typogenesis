"""
KerningNet Model Architecture

Siamese CNN for predicting optimal kerning values between glyph pairs.
The model extracts visual features from both glyphs using a shared encoder,
concatenates the embeddings with font metrics, and regresses to a single
kerning value.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, DEFAULT_MODEL_CONFIG


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv2d -> BatchNorm -> ReLU -> MaxPool.

    A standard building block for the CNN encoder with optional pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        use_pool: bool = True,
    ) -> None:
        """
        Initialize convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolutional kernel.
            pool_size: Size of max pooling kernel.
            use_pool: Whether to apply max pooling after convolution.
        """
        super().__init__()

        # Same padding to maintain spatial dimensions before pooling
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,  # BatchNorm will handle bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class GlyphEncoder(nn.Module):
    """
    CNN encoder for extracting features from glyph images.

    Architecture:
        - Stack of ConvBlocks with increasing channels
        - Global Average Pooling to fixed-size embedding
        - Shared between left and right glyph paths (Siamese architecture)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize glyph encoder.

        Args:
            config: Model configuration with encoder parameters.
        """
        super().__init__()
        self.config = config

        # Build convolutional blocks
        layers = []
        in_ch = config.in_channels

        for i, out_ch in enumerate(config.encoder_channels):
            # Use pooling for all layers except possibly the last
            use_pool = i < len(config.encoder_channels) - 1
            layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.kernel_size,
                    pool_size=config.pool_size,
                    use_pool=use_pool,
                )
            )
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection to embedding dimension if needed
        final_channels = config.encoder_channels[-1]
        if final_channels != config.embedding_dim:
            self.projection = nn.Linear(final_channels, config.embedding_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from glyph image.

        Args:
            x: Input tensor of shape (batch, 1, height, width).

        Returns:
            Embedding tensor of shape (batch, embedding_dim).
        """
        # Convolutional feature extraction
        x = self.conv_layers(x)

        # Global average pooling
        x = self.gap(x)

        # Flatten: (batch, channels, 1, 1) -> (batch, channels)
        x = x.view(x.size(0), -1)

        # Project to embedding dimension
        x = self.projection(x)

        return x


class RegressionHead(nn.Module):
    """
    Regression head for predicting kerning value.

    Takes concatenated glyph embeddings and metrics as input,
    passes through FC layers, and outputs a single kerning value.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize regression head.

        Args:
            config: Model configuration with head parameters.
        """
        super().__init__()
        self.config = config

        # Build FC layers
        layers = []
        in_dim = config.combined_dim

        for hidden_dim in config.fc_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(config.dropout_rate))
            in_dim = hidden_dim

        # Output layer (no activation - regression output)
        layers.append(nn.Linear(in_dim, config.output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict kerning value from combined features.

        Args:
            x: Combined features of shape (batch, combined_dim).

        Returns:
            Kerning predictions of shape (batch, 1).
        """
        return self.fc(x)


class KerningNet(nn.Module):
    """
    Siamese CNN for kerning prediction.

    Architecture:
        1. Shared GlyphEncoder processes both left and right glyph images
        2. Embeddings are concatenated with font metrics
        3. RegressionHead predicts the kerning value

    The model predicts kerning values normalized to [-1, 1] range,
    which should be denormalized to UPM units for actual use.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize KerningNet model.

        Args:
            config: Model configuration. Uses default if not provided.
        """
        super().__init__()
        self.config = config or DEFAULT_MODEL_CONFIG

        # Shared encoder (Siamese architecture)
        self.encoder = GlyphEncoder(self.config)

        # Regression head
        self.head = RegressionHead(self.config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        left_glyph: torch.Tensor,
        right_glyph: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict kerning value for a glyph pair.

        Args:
            left_glyph: Left glyph image, shape (batch, 1, H, W).
            right_glyph: Right glyph image, shape (batch, 1, H, W).
            metrics: Optional font metrics, shape (batch, metrics_dim).
                     If not provided, zeros are used.

        Returns:
            Predicted kerning values, shape (batch, 1).
            Values are normalized to approximately [-1, 1] range.
        """
        batch_size = left_glyph.size(0)

        # Encode both glyphs with shared encoder
        left_emb = self.encoder(left_glyph)
        right_emb = self.encoder(right_glyph)

        # Handle missing metrics
        if metrics is None:
            metrics = torch.zeros(
                batch_size,
                self.config.metrics_dim,
                device=left_glyph.device,
                dtype=left_glyph.dtype,
            )

        # Concatenate features
        combined = torch.cat([left_emb, right_emb, metrics], dim=1)

        # Predict kerning
        kerning = self.head(combined)

        return kerning

    def get_glyph_embedding(self, glyph: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding for a single glyph.

        Useful for caching embeddings when computing kerning for many pairs
        involving the same glyph.

        Args:
            glyph: Glyph image, shape (batch, 1, H, W).

        Returns:
            Glyph embedding, shape (batch, embedding_dim).
        """
        return self.encoder(glyph)

    def predict_from_embeddings(
        self,
        left_emb: torch.Tensor,
        right_emb: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict kerning from pre-computed embeddings.

        Args:
            left_emb: Left glyph embedding, shape (batch, embedding_dim).
            right_emb: Right glyph embedding, shape (batch, embedding_dim).
            metrics: Optional font metrics, shape (batch, metrics_dim).

        Returns:
            Predicted kerning values, shape (batch, 1).
        """
        batch_size = left_emb.size(0)

        if metrics is None:
            metrics = torch.zeros(
                batch_size,
                self.config.metrics_dim,
                device=left_emb.device,
                dtype=left_emb.dtype,
            )

        combined = torch.cat([left_emb, right_emb, metrics], dim=1)
        return self.head(combined)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a string summary of the model architecture."""
        lines = [
            "KerningNet Model Summary",
            "=" * 50,
            f"Input size: {self.config.image_size}x{self.config.image_size}",
            f"Encoder channels: {self.config.encoder_channels}",
            f"Embedding dimension: {self.config.embedding_dim}",
            f"Metrics dimension: {self.config.metrics_dim}",
            f"Combined dimension: {self.config.combined_dim}",
            f"FC hidden dims: {self.config.fc_hidden_dims}",
            f"Dropout rate: {self.config.dropout_rate}",
            f"Total parameters: {self.num_parameters:,}",
        ]
        return "\n".join(lines)


class KerningNetWithAuxiliary(KerningNet):
    """
    KerningNet with auxiliary outputs for multi-task learning.

    In addition to the main kerning prediction, this model can also:
    - Predict whether kerning is needed (classification)
    - Predict kerning direction (positive/negative/zero)

    This can help with learning, especially for imbalanced datasets
    where most pairs have zero kerning.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize model with auxiliary heads.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        # Auxiliary head: needs kerning? (binary classification)
        self.needs_kerning_head = nn.Sequential(
            nn.Linear(self.config.combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Auxiliary head: kerning direction (negative/zero/positive)
        self.direction_head = nn.Sequential(
            nn.Linear(self.config.combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # 3-class: negative, zero, positive
        )

    def forward(
        self,
        left_glyph: torch.Tensor,
        right_glyph: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with auxiliary outputs.

        Args:
            left_glyph: Left glyph image.
            right_glyph: Right glyph image.
            metrics: Optional font metrics.

        Returns:
            Tuple of:
                - kerning: Predicted kerning value (batch, 1)
                - needs_kerning: Binary logits (batch, 1)
                - direction: Direction logits (batch, 3)
        """
        batch_size = left_glyph.size(0)

        # Encode both glyphs
        left_emb = self.encoder(left_glyph)
        right_emb = self.encoder(right_glyph)

        # Handle missing metrics
        if metrics is None:
            metrics = torch.zeros(
                batch_size,
                self.config.metrics_dim,
                device=left_glyph.device,
                dtype=left_glyph.dtype,
            )

        # Concatenate features
        combined = torch.cat([left_emb, right_emb, metrics], dim=1)

        # Main kerning prediction
        kerning = self.head(combined)

        # Auxiliary predictions
        needs_kerning = self.needs_kerning_head(combined)
        direction = self.direction_head(combined)

        return kerning, needs_kerning, direction


def create_model(
    config: Optional[ModelConfig] = None,
    use_auxiliary: bool = False,
) -> nn.Module:
    """
    Factory function to create KerningNet model.

    Args:
        config: Model configuration.
        use_auxiliary: Whether to use auxiliary outputs.

    Returns:
        KerningNet or KerningNetWithAuxiliary model.
    """
    if use_auxiliary:
        return KerningNetWithAuxiliary(config)
    return KerningNet(config)


def load_model(
    checkpoint_path: str,
    config: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None,
    use_auxiliary: bool = False,
) -> nn.Module:
    """
    Load a trained KerningNet model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        config: Model configuration. If None, attempts to load from checkpoint.
        device: Device to load model to. If None, uses CUDA if available.
        use_auxiliary: Whether to load auxiliary model variant.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If checkpoint is incompatible with model.
    """
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to get config from checkpoint if not provided
    if config is None and "config" in checkpoint:
        config = checkpoint["config"]

    # Create model
    model = create_model(config, use_auxiliary)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Test the model architecture
    print("Testing KerningNet architecture...\n")

    config = DEFAULT_MODEL_CONFIG
    model = KerningNet(config)

    print(model.summary())
    print()

    # Test forward pass
    batch_size = 4
    left = torch.randn(batch_size, 1, config.image_size, config.image_size)
    right = torch.randn(batch_size, 1, config.image_size, config.image_size)
    metrics = torch.randn(batch_size, config.metrics_dim)

    # Without metrics
    output_no_metrics = model(left, right)
    print(f"Output shape (no metrics): {output_no_metrics.shape}")

    # With metrics
    output_with_metrics = model(left, right, metrics)
    print(f"Output shape (with metrics): {output_with_metrics.shape}")

    # Test embedding extraction
    emb = model.get_glyph_embedding(left)
    print(f"Embedding shape: {emb.shape}")

    # Test prediction from embeddings
    left_emb = model.get_glyph_embedding(left)
    right_emb = model.get_glyph_embedding(right)
    output_from_emb = model.predict_from_embeddings(left_emb, right_emb, metrics)
    print(f"Output from embeddings: {output_from_emb.shape}")

    # Verify outputs match
    diff = (output_with_metrics - output_from_emb).abs().max().item()
    print(f"Max difference between direct and embedding-based prediction: {diff:.6f}")

    # Test auxiliary model
    print("\nTesting KerningNetWithAuxiliary...")
    aux_model = KerningNetWithAuxiliary(config)
    kerning, needs_kern, direction = aux_model(left, right, metrics)
    print(f"Kerning shape: {kerning.shape}")
    print(f"Needs kerning shape: {needs_kern.shape}")
    print(f"Direction shape: {direction.shape}")
    print(f"Auxiliary model parameters: {aux_model.num_parameters:,}")
