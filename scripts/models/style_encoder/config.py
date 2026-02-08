"""Configuration for StyleEncoder model and training.

This module defines all hyperparameters and settings for the StyleEncoder,
including model architecture, training parameters, and default paths.

Example:
    >>> from scripts.models.style_encoder.config import StyleEncoderConfig, TrainingConfig
    >>> model_config = StyleEncoderConfig(backbone="efficientnet_b0", embedding_dim=128)
    >>> train_config = TrainingConfig(batch_size=256, learning_rate=3e-4)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class StyleEncoderConfig:
    """Configuration for StyleEncoder model architecture.

    Attributes:
        backbone: CNN backbone architecture. Supported: "resnet18", "efficientnet_b0"
        embedding_dim: Dimension of output style embedding vector
        projection_dim: Dimension of projection head (for contrastive learning)
        input_size: Input image size (height, width)
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate in projection head
        normalize_embedding: Whether to L2-normalize output embeddings
    """

    backbone: Literal["resnet18", "efficientnet_b0"] = "resnet18"
    embedding_dim: int = 128
    projection_dim: int = 256
    input_size: tuple[int, int] = (64, 64)
    input_channels: int = 1
    pretrained: bool = True
    dropout_rate: float = 0.1
    normalize_embedding: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.backbone not in ("resnet18", "efficientnet_b0"):
            raise ValueError(
                f"Unsupported backbone: {self.backbone}. "
                "Supported: 'resnet18', 'efficientnet_b0'"
            )
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.projection_dim <= 0:
            raise ValueError(
                f"projection_dim must be positive, got {self.projection_dim}"
            )
        if self.input_channels not in (1, 3):
            raise ValueError(f"input_channels must be 1 or 3, got {self.input_channels}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {self.dropout_rate}"
            )


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation during contrastive training.

    Attributes:
        rotation_range: Random rotation range in degrees
        scale_range: Random scaling range (min, max)
        translate_range: Random translation range as fraction of image size
        elastic_alpha: Elastic deformation alpha (0 to disable)
        elastic_sigma: Elastic deformation sigma
        erosion_prob: Probability of applying erosion
        dilation_prob: Probability of applying dilation
        noise_std: Standard deviation of Gaussian noise (0 to disable)
        invert_prob: Probability of inverting image colors
    """

    rotation_range: float = 10.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate_range: float = 0.1
    elastic_alpha: float = 0.0
    elastic_sigma: float = 5.0
    erosion_prob: float = 0.1
    dilation_prob: float = 0.1
    noise_std: float = 0.02
    invert_prob: float = 0.0


@dataclass
class LossConfig:
    """Configuration for contrastive loss functions.

    Attributes:
        loss_type: Type of contrastive loss. Supported: "nt_xent", "triplet", "infonce", "supcon"
        temperature: Temperature parameter for NT-Xent and InfoNCE losses
        triplet_margin: Margin for triplet loss
        hard_negative_mining: Whether to use hard negative mining for triplet loss
        label_smoothing: Label smoothing factor (0 to disable)
    """

    loss_type: Literal["nt_xent", "triplet", "infonce", "supcon"] = "nt_xent"
    temperature: float = 0.07
    triplet_margin: float = 0.5
    hard_negative_mining: bool = True
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.loss_type not in ("nt_xent", "triplet", "infonce", "supcon"):
            raise ValueError(
                f"Unsupported loss_type: {self.loss_type}. "
                "Supported: 'nt_xent', 'triplet', 'infonce', 'supcon'"
            )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.triplet_margin <= 0:
            raise ValueError(f"triplet_margin must be positive, got {self.triplet_margin}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )


@dataclass
class TrainingConfig:
    """Configuration for training the StyleEncoder.

    Attributes:
        batch_size: Training batch size (number of fonts per batch)
        glyphs_per_font: Number of glyphs sampled per font for contrastive pairs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight decay
        epochs: Number of training epochs
        warmup_epochs: Number of warmup epochs for learning rate
        min_lr: Minimum learning rate for cosine annealing
        gradient_clip_norm: Maximum gradient norm (0 to disable clipping)
        mixed_precision: Whether to use automatic mixed precision (AMP)
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory for data loading
        seed: Random seed for reproducibility
        checkpoint_interval: Save checkpoint every N epochs
        log_interval: Log metrics every N batches
        val_interval: Run validation every N epochs
    """

    batch_size: int = 256
    glyphs_per_font: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42
    checkpoint_interval: int = 5
    log_interval: int = 100
    val_interval: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.glyphs_per_font < 2:
            raise ValueError(
                f"glyphs_per_font must be at least 2 for contrastive learning, "
                f"got {self.glyphs_per_font}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")


@dataclass
class PathConfig:
    """Configuration for file paths.

    Attributes:
        data_dir: Root directory for training data
        checkpoint_dir: Directory for saving model checkpoints
        log_dir: Directory for TensorBoard/WandB logs
        cache_dir: Directory for caching preprocessed data
    """

    data_dir: Path = field(default_factory=lambda: Path("data/style_dataset"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/style_encoder"))
    log_dir: Path = field(default_factory=lambda: Path("logs/style_encoder"))
    cache_dir: Path = field(default_factory=lambda: Path("cache/style_encoder"))

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class FullConfig:
    """Complete configuration combining all config sections.

    Attributes:
        model: Model architecture configuration
        training: Training hyperparameters
        loss: Loss function configuration
        augmentation: Data augmentation configuration
        paths: File path configuration
        wandb_project: WandB project name (None to disable)
        wandb_entity: WandB entity/username
        experiment_name: Name for this experiment run
    """

    model: StyleEncoderConfig = field(default_factory=StyleEncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    wandb_project: Optional[str] = "typogenesis-style-encoder"
    wandb_entity: Optional[str] = None
    experiment_name: str = "style_encoder"


def default_config() -> FullConfig:
    """Create default configuration for StyleEncoder training.

    Returns:
        FullConfig with default values optimized for quality.
    """
    return FullConfig()


def fast_config() -> FullConfig:
    """Create fast configuration for quick experimentation.

    Uses smaller batch size, fewer epochs, and simpler augmentation
    for faster iteration during development.

    Returns:
        FullConfig optimized for speed over quality.
    """
    return FullConfig(
        model=StyleEncoderConfig(
            backbone="resnet18",
            pretrained=True,
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=10,
            warmup_epochs=1,
            num_workers=4,
        ),
        augmentation=AugmentationConfig(
            rotation_range=5.0,
            scale_range=(0.95, 1.05),
            elastic_alpha=0.0,
        ),
    )


def efficientnet_config() -> FullConfig:
    """Create configuration using EfficientNet-B0 backbone.

    EfficientNet-B0 is more efficient than ResNet-18 while achieving
    similar or better performance.

    Returns:
        FullConfig with EfficientNet-B0 backbone.
    """
    return FullConfig(
        model=StyleEncoderConfig(
            backbone="efficientnet_b0",
            pretrained=True,
        ),
    )
