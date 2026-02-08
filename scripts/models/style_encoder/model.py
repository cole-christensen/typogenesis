"""StyleEncoder CNN model architecture.

This module implements the StyleEncoder neural network that learns to encode
glyph images into style embeddings using contrastive learning.

Architecture:
    1. Backbone: ResNet-18 or EfficientNet-B0 (modified for single-channel input)
    2. Global Average Pooling
    3. Projection Head: FC -> ReLU -> Dropout -> FC
    4. L2 Normalization (optional)

The model outputs 128-dimensional embeddings where:
    - Same-font glyphs have high cosine similarity (> 0.9 target)
    - Different-font glyphs have low cosine similarity (< 0.5 target)

Example:
    >>> from scripts.models.style_encoder.model import StyleEncoder, create_style_encoder
    >>> from scripts.models.style_encoder.config import StyleEncoderConfig
    >>>
    >>> config = StyleEncoderConfig(backbone="resnet18", embedding_dim=128)
    >>> model = create_style_encoder(config)
    >>>
    >>> # Single image inference
    >>> image = torch.randn(1, 1, 64, 64)
    >>> embedding = model(image)  # Shape: (1, 128)
    >>>
    >>> # Get projection (for training with contrastive loss)
    >>> embedding, projection = model(image, return_projection=True)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from .config import StyleEncoderConfig

# Type alias for clarity
Tensor = torch.Tensor


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects backbone features to a lower-dimensional space for
    computing contrastive loss. This head is typically discarded
    after training.

    Architecture: Linear -> ReLU -> Dropout -> Linear

    Attributes:
        fc1: First fully-connected layer
        relu: ReLU activation
        dropout: Dropout layer
        fc2: Second fully-connected layer (output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize projection head.

        Args:
            input_dim: Input feature dimension from backbone
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through projection head.

        Args:
            x: Input features from backbone, shape (batch_size, input_dim)

        Returns:
            Projected features, shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StyleEncoder(nn.Module):
    """CNN-based style encoder for glyph images.

    Encodes 64x64 glyph images into 128-dimensional style embeddings
    using a pretrained backbone (ResNet-18 or EfficientNet-B0) followed
    by a projection head for contrastive learning.

    The model supports two modes:
    1. Inference mode: Returns only the embedding (after backbone + final projection)
    2. Training mode: Returns both embedding and projection head output

    Attributes:
        config: Model configuration
        backbone: CNN feature extractor
        pool: Global average pooling
        embedding_head: Projects to final embedding dimension
        projection_head: Additional projection for contrastive learning
    """

    def __init__(self, config: StyleEncoderConfig) -> None:
        """Initialize StyleEncoder.

        Args:
            config: Model configuration specifying backbone, dimensions, etc.

        Raises:
            ValueError: If unsupported backbone is specified
        """
        super().__init__()
        self.config = config

        # Build backbone
        if config.backbone == "resnet18":
            self._build_resnet18_backbone()
        elif config.backbone == "efficientnet_b0":
            self._build_efficientnet_backbone()
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding head: maps backbone features to embedding_dim
        self.embedding_head = nn.Sequential(
            nn.Linear(self._backbone_dim, config.embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Projection head for contrastive learning (used during training)
        self.projection_head = ProjectionHead(
            input_dim=config.embedding_dim,
            hidden_dim=config.projection_dim,
            output_dim=config.projection_dim,
            dropout_rate=config.dropout_rate,
        )

    def _build_resnet18_backbone(self) -> None:
        """Build ResNet-18 backbone with single-channel input adaptation."""
        # Load pretrained ResNet-18
        weights = ResNet18_Weights.DEFAULT if self.config.pretrained else None
        resnet = resnet18(weights=weights)

        # Modify first conv layer for single-channel input if needed
        if self.config.input_channels == 1:
            # Average pretrained weights across RGB channels
            original_conv = resnet.conv1
            new_conv = nn.Conv2d(
                1,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            if self.config.pretrained:
                # Average RGB weights for grayscale
                new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            resnet.conv1 = new_conv

        # Remove final FC layer (we'll add our own)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self._backbone_dim = 512  # ResNet-18 output channels

    def _build_efficientnet_backbone(self) -> None:
        """Build EfficientNet-B0 backbone with single-channel input adaptation."""
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for EfficientNet backbone. "
                "Install with: pip install timm"
            ) from e

        # Load pretrained EfficientNet-B0
        effnet = timm.create_model(
            "efficientnet_b0",
            pretrained=self.config.pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # We'll add our own pooling
        )

        # Modify first conv for single-channel input if needed
        if self.config.input_channels == 1:
            original_conv = effnet.conv_stem
            new_conv = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            if self.config.pretrained:
                # Average RGB weights for grayscale
                new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            effnet.conv_stem = new_conv

        self.backbone = effnet
        self._backbone_dim = 1280  # EfficientNet-B0 output channels

    def forward(
        self,
        x: Tensor,
        return_projection: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass through StyleEncoder.

        Args:
            x: Input glyph images, shape (batch_size, channels, height, width)
               Expected shape: (N, 1, 64, 64) for grayscale
            return_projection: If True, also return projection head output
                              (used during contrastive training)

        Returns:
            If return_projection is False:
                embedding: Style embedding, shape (batch_size, embedding_dim)
            If return_projection is True:
                Tuple of (embedding, projection):
                    - embedding: Shape (batch_size, embedding_dim)
                    - projection: Shape (batch_size, projection_dim)
        """
        # Extract features through backbone
        features = self.backbone(x)

        # Global average pooling
        features = self.pool(features)
        features = features.flatten(start_dim=1)

        # Project to embedding dimension
        embedding = self.embedding_head(features)

        # L2 normalize embedding if configured
        if self.config.normalize_embedding:
            embedding = F.normalize(embedding, p=2, dim=1)

        if return_projection:
            # Also compute projection for contrastive loss
            projection = self.projection_head(embedding)
            projection = F.normalize(projection, p=2, dim=1)
            return embedding, projection

        return embedding

    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings.

        Returns:
            Embedding dimension (default: 128)
        """
        return self.config.embedding_dim

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode images to embeddings (inference mode).

        Convenience method that sets model to eval mode and disables
        gradient computation for efficient inference.

        Args:
            x: Input glyph images, shape (batch_size, channels, height, width)

        Returns:
            Style embeddings, shape (batch_size, embedding_dim)
        """
        was_training = self.training
        self.eval()
        embedding = self.forward(x, return_projection=False)
        if was_training:
            self.train()
        return embedding


def create_style_encoder(
    config: Optional[StyleEncoderConfig] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> StyleEncoder:
    """Factory function to create StyleEncoder model.

    Creates and optionally loads a pretrained StyleEncoder model.

    Args:
        config: Model configuration. If None, uses default configuration.
        checkpoint_path: Path to checkpoint file to load weights from.
        device: Device to load model on. If None, uses CUDA if available.

    Returns:
        Initialized StyleEncoder model.

    Example:
        >>> # Create with default config
        >>> model = create_style_encoder()
        >>>
        >>> # Create with custom config
        >>> config = StyleEncoderConfig(backbone="efficientnet_b0")
        >>> model = create_style_encoder(config)
        >>>
        >>> # Load from checkpoint
        >>> model = create_style_encoder(checkpoint_path="best_model.pt")
    """
    if config is None:
        config = StyleEncoderConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StyleEncoder(config)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: StyleEncoder) -> str:
    """Generate a summary of the StyleEncoder model.

    Args:
        model: StyleEncoder instance

    Returns:
        String containing model summary with architecture and parameters.
    """
    lines = [
        "StyleEncoder Summary",
        "=" * 50,
        f"Backbone: {model.config.backbone}",
        f"Input size: {model.config.input_size}",
        f"Input channels: {model.config.input_channels}",
        f"Embedding dimension: {model.config.embedding_dim}",
        f"Projection dimension: {model.config.projection_dim}",
        f"Normalize embedding: {model.config.normalize_embedding}",
        f"Pretrained: {model.config.pretrained}",
        "",
        f"Total parameters: {count_parameters(model):,}",
        f"Trainable parameters: {count_parameters(model):,}",
        "=" * 50,
    ]
    return "\n".join(lines)
