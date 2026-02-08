"""
Configuration for GlyphDiffusion model.

This module defines all hyperparameters for the flow-matching diffusion model
used to generate glyph images. Configuration is organized into dataclasses
for model architecture, training, and sampling.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Resolution(Enum):
    """Supported image resolutions for glyph generation."""

    LOW = 64
    HIGH = 128


@dataclass
class ModelConfig:
    """Configuration for the UNet model architecture.

    Attributes:
        resolution: Image resolution (64 or 128 pixels).
        in_channels: Number of input channels (1 for grayscale + optional mask).
        out_channels: Number of output channels (1 for grayscale).
        base_channels: Base number of channels in the UNet.
        channel_multipliers: Multipliers for each UNet level.
        num_res_blocks: Number of residual blocks per level.
        attention_resolutions: Resolutions at which to apply attention.
        dropout: Dropout rate in residual blocks.
        num_heads: Number of attention heads.
        num_characters: Number of distinct characters (a-z, A-Z, 0-9).
            The embedding table is sized num_characters + 1 to accommodate
            a null class for classifier-free guidance.
        char_embed_dim: Dimension of character embeddings.
        style_embed_dim: Dimension of style embeddings from StyleEncoder.
        time_embed_dim: Dimension of time embeddings.
        use_mask_conditioning: Whether to support partial outline masks.
    """

    resolution: Resolution = Resolution.LOW
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16, 8)
    dropout: float = 0.0
    num_heads: int = 4
    num_characters: int = 62  # a-z (26) + A-Z (26) + 0-9 (10)
    char_embed_dim: int = 64
    style_embed_dim: int = 128
    time_embed_dim: int = 256
    use_mask_conditioning: bool = True

    @property
    def image_size(self) -> int:
        """Get the image size in pixels."""
        return self.resolution.value


@dataclass
class TrainingConfig:
    """Configuration for training the model.

    Attributes:
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        lr_warmup_steps: Number of warmup steps for learning rate.
        lr_scheduler: Type of learning rate scheduler.
        num_epochs: Total number of training epochs.
        gradient_clip_norm: Maximum gradient norm for clipping.
        ema_decay: Exponential moving average decay for model weights.
        mixed_precision: Whether to use mixed precision (fp16) training.
        checkpoint_every_n_epochs: Save checkpoint every N epochs.
        validate_every_n_epochs: Run validation every N epochs.
        log_every_n_steps: Log metrics every N steps.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducibility.
        use_wandb: Whether to use Weights & Biases logging.
        use_tensorboard: Whether to use TensorBoard logging.
        project_name: Name of the wandb/tensorboard project.
        run_name: Name of this training run.
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    mixed_precision: bool = True
    checkpoint_every_n_epochs: int = 5
    validate_every_n_epochs: int = 1
    log_every_n_steps: int = 100
    num_workers: int = 4
    seed: int = 42
    use_wandb: bool = True
    use_tensorboard: bool = False
    project_name: str = "typogenesis-glyph-diffusion"
    run_name: Optional[str] = None


@dataclass
class FlowMatchingConfig:
    """Configuration for the flow-matching noise schedule.

    Flow matching uses optimal transport to learn a velocity field that
    transforms noise to data. This is faster than DDPM at inference time.

    Attributes:
        num_train_steps: Number of timesteps during training.
        num_inference_steps: Default number of inference steps.
        sigma_min: Minimum noise level (small positive value for stability).
        sigma_max: Maximum noise level (1.0 for full noise).
        prediction_type: What the model predicts ("velocity" for flow matching).
        scheduler_type: Type of scheduler ("linear" for optimal transport).
    """

    num_train_steps: int = 1000
    num_inference_steps: int = 50
    sigma_min: float = 1e-4
    sigma_max: float = 1.0
    prediction_type: str = "velocity"
    scheduler_type: str = "linear"


@dataclass
class SamplingConfig:
    """Configuration for inference and sampling.

    Attributes:
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale (1.0 = no guidance).
        batch_size: Batch size for generation.
        output_format: Output format ("png" or "numpy").
        save_intermediates: Whether to save intermediate denoising steps.
        seed: Random seed for reproducibility (None for random).
    """

    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    batch_size: int = 16
    output_format: str = "png"
    save_intermediates: bool = False
    seed: Optional[int] = None


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        data_dir: Directory containing the glyph dataset.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        augment: Whether to apply data augmentation.
        cache_in_memory: Whether to cache dataset in memory.
    """

    data_dir: Path = field(default_factory=lambda: Path("data/glyphs"))
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augment: bool = True
    cache_in_memory: bool = False


@dataclass
class Config:
    """Master configuration combining all sub-configurations.

    This is the main configuration class used throughout the codebase.
    It combines model, training, flow matching, sampling, and data configs.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration for 64x64 resolution."""
        return cls()

    @classmethod
    def high_resolution(cls) -> "Config":
        """Create configuration for 128x128 resolution."""
        config = cls()
        config.model.resolution = Resolution.HIGH
        # Adjust batch size for larger images
        config.training.batch_size = 32
        return config

    @classmethod
    def fast_dev(cls) -> "Config":
        """Create configuration for fast development/debugging."""
        config = cls()
        config.training.batch_size = 8
        config.training.num_epochs = 5
        config.training.log_every_n_steps = 10
        config.flow_matching.num_inference_steps = 10
        config.sampling.num_inference_steps = 10
        return config


# Character mapping utilities

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
ALL_CHARACTERS = LOWERCASE + UPPERCASE + DIGITS

CHAR_TO_IDX: dict[str, int] = {char: idx for idx, char in enumerate(ALL_CHARACTERS)}
IDX_TO_CHAR: dict[int, str] = {idx: char for idx, char in enumerate(ALL_CHARACTERS)}


def char_to_index(char: str) -> int:
    """Convert a character to its index in the embedding table.

    Args:
        char: A single character (a-z, A-Z, or 0-9).

    Returns:
        Index in range [0, 61].

    Raises:
        ValueError: If character is not in the supported set.
    """
    if char not in CHAR_TO_IDX:
        raise ValueError(
            f"Character '{char}' not in supported set. "
            f"Must be one of: {ALL_CHARACTERS}"
        )
    return CHAR_TO_IDX[char]


def index_to_char(idx: int) -> str:
    """Convert an index to its corresponding character.

    Args:
        idx: Index in range [0, 61].

    Returns:
        The corresponding character.

    Raises:
        ValueError: If index is out of range.
    """
    if idx not in IDX_TO_CHAR:
        raise ValueError(
            f"Index {idx} out of range. Must be in [0, {len(ALL_CHARACTERS) - 1}]"
        )
    return IDX_TO_CHAR[idx]


def get_character_set(set_name: str) -> list[str]:
    """Get a predefined character set by name.

    Args:
        set_name: One of "lowercase", "uppercase", "digits", "letters", "all".

    Returns:
        List of characters in the set.

    Raises:
        ValueError: If set_name is not recognized.
    """
    sets = {
        "lowercase": list(LOWERCASE),
        "uppercase": list(UPPERCASE),
        "digits": list(DIGITS),
        "letters": list(LOWERCASE + UPPERCASE),
        "all": list(ALL_CHARACTERS),
    }
    if set_name not in sets:
        raise ValueError(
            f"Unknown character set '{set_name}'. "
            f"Must be one of: {list(sets.keys())}"
        )
    return sets[set_name]
