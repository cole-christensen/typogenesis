"""
KerningNet Configuration

Configuration constants for the KerningNet Siamese CNN model, including
model hyperparameters, training settings, and critical kerning pairs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# =============================================================================
# Model Architecture Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for KerningNet model architecture."""

    # Input dimensions
    image_size: int = 64  # Input glyph image size (64x64)
    in_channels: int = 1  # Grayscale images

    # CNN Encoder configuration
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)  # Channel progression
    kernel_size: int = 3
    pool_size: int = 2

    # Embedding dimension from CNN encoder
    embedding_dim: int = 256

    # Font metrics input dimension
    metrics_dim: int = 4  # [left_advance, right_lsb, x_height_ratio, cap_height_ratio]

    # Regression head configuration
    fc_hidden_dims: Tuple[int, ...] = (256, 64)
    dropout_rate: float = 0.3

    # Output configuration
    output_dim: int = 1  # Single kerning value

    @property
    def combined_dim(self) -> int:
        """Dimension after concatenating left, right embeddings and metrics."""
        return 2 * self.embedding_dim + self.metrics_dim  # 256 + 256 + 4 = 516


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # Options: "cosine", "step", "plateau"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6

    # For step scheduler
    lr_step_size: int = 30
    lr_gamma: float = 0.1

    # For plateau scheduler
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Loss function
    loss_fn: str = "huber"  # Options: "mse", "huber", "smooth_l1"
    huber_delta: float = 1.0

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/kerning_net"))
    checkpoint_every: int = 5  # Save checkpoint every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints

    # Logging
    log_every: int = 100  # Log every N batches
    use_wandb: bool = False
    use_tensorboard: bool = True
    experiment_name: str = "kerning_net"

    # Validation
    val_split: float = 0.1
    val_every: int = 1  # Validate every N epochs

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Reproducibility
    seed: int = 42


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for kerning dataset."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data/kerning"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/kerning"))

    # Normalization parameters (based on typical font metrics)
    kerning_min: float = -200.0  # Minimum kerning value in UPM units
    kerning_max: float = 100.0   # Maximum kerning value in UPM units

    # Image preprocessing
    normalize_images: bool = True
    image_mean: float = 0.5
    image_std: float = 0.5

    # Data augmentation
    augment: bool = True
    rotation_range: float = 2.0  # Degrees
    scale_range: Tuple[float, float] = (0.95, 1.05)
    translate_range: float = 0.02  # Fraction of image size

    # Filtering
    min_kerning_magnitude: int = 0  # Include all pairs
    max_pairs_per_font: int = 0  # 0 = no limit

    def normalize_kerning(self, value: float) -> float:
        """Normalize kerning value to [-1, 1] range."""
        return (value - self.kerning_min) / (self.kerning_max - self.kerning_min) * 2 - 1

    def denormalize_kerning(self, normalized: float) -> float:
        """Convert normalized kerning back to UPM units."""
        return (normalized + 1) / 2 * (self.kerning_max - self.kerning_min) + self.kerning_min


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    # Model path
    model_path: Path = field(default_factory=lambda: Path("checkpoints/kerning_net/best_model.pt"))

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Batch inference
    batch_size: int = 256

    # Output format
    output_format: str = "table"  # Options: "table", "json", "csv"
    round_to_int: bool = True

    # Filtering
    min_kerning_to_include: int = 2  # Skip pairs with |kerning| < this value


# =============================================================================
# Critical Kerning Pairs
# =============================================================================

# These pairs commonly require kerning adjustments in professional fonts
# Organized by category for clarity

CRITICAL_PAIRS_CAPITALS_WITH_DIAGONALS: List[Tuple[str, str]] = [
    # A with following diagonals
    ("A", "V"), ("A", "W"), ("A", "Y"),
    # V, W, Y with following A
    ("V", "A"), ("W", "A"), ("Y", "A"),
    # A with T (horizontal vs diagonal)
    ("A", "T"), ("T", "A"),
]

CRITICAL_PAIRS_L_COMBINATIONS: List[Tuple[str, str]] = [
    # L with letters that have open left sides
    ("L", "T"), ("L", "V"), ("L", "W"), ("L", "Y"),
    ("L", "O"), ("L", "C"), ("L", "G"), ("L", "Q"),
    # Letters before L
    ("T", "L"), ("F", "L"),
]

CRITICAL_PAIRS_T_LOWERCASE: List[Tuple[str, str]] = [
    # T with round/open lowercase letters
    ("T", "a"), ("T", "e"), ("T", "i"), ("T", "o"), ("T", "r"), ("T", "u"), ("T", "y"),
    # Lowercase before T
    ("o", "T"), ("e", "T"),
]

CRITICAL_PAIRS_V_LOWERCASE: List[Tuple[str, str]] = [
    ("V", "a"), ("V", "e"), ("V", "i"), ("V", "o"), ("V", "u"),
    ("a", "V"), ("e", "V"), ("o", "V"),
]

CRITICAL_PAIRS_W_LOWERCASE: List[Tuple[str, str]] = [
    ("W", "a"), ("W", "e"), ("W", "i"), ("W", "o"), ("W", "u"),
    ("a", "W"), ("e", "W"), ("o", "W"),
]

CRITICAL_PAIRS_Y_LOWERCASE: List[Tuple[str, str]] = [
    ("Y", "a"), ("Y", "e"), ("Y", "i"), ("Y", "o"), ("Y", "u"),
    ("a", "Y"), ("e", "Y"), ("o", "Y"),
]

CRITICAL_PAIRS_F_COMBINATIONS: List[Tuple[str, str]] = [
    ("F", "a"), ("F", "e"), ("F", "i"), ("F", "o"), ("F", "r"),
    ("F", "A"), ("F", "J"),
]

CRITICAL_PAIRS_P_COMBINATIONS: List[Tuple[str, str]] = [
    ("P", "a"), ("P", "e"), ("P", "o"), ("P", "A"),
    ("P", "."), ("P", ","),
]

CRITICAL_PAIRS_R_COMBINATIONS: List[Tuple[str, str]] = [
    ("R", "T"), ("R", "V"), ("R", "W"), ("R", "Y"),
]

CRITICAL_PAIRS_LOWERCASE_F: List[Tuple[str, str]] = [
    # f-ligature candidates
    ("f", "f"), ("f", "i"), ("f", "l"), ("f", "t"),
    ("f", "a"), ("f", "e"), ("f", "o"),
]

CRITICAL_PAIRS_LOWERCASE_R: List[Tuple[str, str]] = [
    ("r", "a"), ("r", "e"), ("r", "o"), ("r", "n"), ("r", "m"),
    ("r", "."), ("r", ","), ("r", "'"),
]

CRITICAL_PAIRS_LOWERCASE_V_W_Y: List[Tuple[str, str]] = [
    ("v", "a"), ("v", "e"), ("v", "o"),
    ("w", "a"), ("w", "e"), ("w", "o"),
    ("y", "a"), ("y", "e"), ("y", "o"), ("y", "."), ("y", ","),
]

CRITICAL_PAIRS_QUOTES_PUNCTUATION: List[Tuple[str, str]] = [
    # Opening quotes
    ('"', "A"), ('"', "J"), ('"', "T"), ('"', "V"), ('"', "W"), ('"', "Y"),
    ("'", "A"), ("'", "J"), ("'", "T"), ("'", "V"), ("'", "W"), ("'", "Y"),
    # Closing quotes
    ("A", '"'), ("V", '"'), ("W", '"'), ("Y", '"'),
    ("A", "'"), ("V", "'"), ("W", "'"), ("Y", "'"),
    # Parentheses
    ("(", "A"), ("(", "J"), ("(", "C"), ("(", "G"), ("(", "O"), ("(", "Q"),
    # Periods/commas with capitals
    ("F", "."), ("T", "."), ("V", "."), ("W", "."), ("Y", "."),
    ("F", ","), ("T", ","), ("V", ","), ("W", ","), ("Y", ","),
]

CRITICAL_PAIRS_NUMBERS: List[Tuple[str, str]] = [
    ("1", "1"), ("1", "0"), ("1", "4"), ("1", "7"),
    ("7", "4"), ("7", "3"), ("7", "8"),
    ("4", "1"),
]

CRITICAL_PAIRS_ROUND_LETTERS: List[Tuple[str, str]] = [
    # O, C, G, Q combinations
    ("O", "A"), ("O", "T"), ("O", "V"), ("O", "W"), ("O", "X"), ("O", "Y"),
    ("C", "A"), ("C", "O"),
    ("Q", "u"),
]


def get_all_critical_pairs() -> List[Tuple[str, str]]:
    """
    Get all critical kerning pairs as a flat list.

    Returns:
        List of (left_char, right_char) tuples representing critical kerning pairs.
    """
    all_pairs: List[Tuple[str, str]] = []

    all_pairs.extend(CRITICAL_PAIRS_CAPITALS_WITH_DIAGONALS)
    all_pairs.extend(CRITICAL_PAIRS_L_COMBINATIONS)
    all_pairs.extend(CRITICAL_PAIRS_T_LOWERCASE)
    all_pairs.extend(CRITICAL_PAIRS_V_LOWERCASE)
    all_pairs.extend(CRITICAL_PAIRS_W_LOWERCASE)
    all_pairs.extend(CRITICAL_PAIRS_Y_LOWERCASE)
    all_pairs.extend(CRITICAL_PAIRS_F_COMBINATIONS)
    all_pairs.extend(CRITICAL_PAIRS_P_COMBINATIONS)
    all_pairs.extend(CRITICAL_PAIRS_R_COMBINATIONS)
    all_pairs.extend(CRITICAL_PAIRS_LOWERCASE_F)
    all_pairs.extend(CRITICAL_PAIRS_LOWERCASE_R)
    all_pairs.extend(CRITICAL_PAIRS_LOWERCASE_V_W_Y)
    all_pairs.extend(CRITICAL_PAIRS_QUOTES_PUNCTUATION)
    all_pairs.extend(CRITICAL_PAIRS_NUMBERS)
    all_pairs.extend(CRITICAL_PAIRS_ROUND_LETTERS)

    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    return unique_pairs


def get_negative_kerning_pairs() -> List[Tuple[str, str]]:
    """
    Get pairs that typically require negative kerning (tighter spacing).

    These pairs have letters with complementary shapes that create
    excessive visual whitespace when set at normal sidebearings.

    Returns:
        List of pairs expected to have negative kerning values.
    """
    return [
        # Classic kerning pairs with open space
        ("A", "V"), ("A", "W"), ("A", "Y"), ("A", "T"),
        ("V", "A"), ("W", "A"), ("Y", "A"), ("T", "A"),
        ("L", "T"), ("L", "V"), ("L", "W"), ("L", "Y"),
        ("P", "A"), ("P", "."), ("P", ","),
        ("F", "A"), ("F", "."), ("F", ","),
        ("T", "a"), ("T", "e"), ("T", "o"), ("T", "r"), ("T", "u"), ("T", "."), ("T", ","),
        ("V", "a"), ("V", "e"), ("V", "o"), ("V", "."), ("V", ","),
        ("W", "a"), ("W", "e"), ("W", "o"), ("W", "."), ("W", ","),
        ("Y", "a"), ("Y", "e"), ("Y", "o"), ("Y", "."), ("Y", ","),
        ("r", "a"), ("r", "e"), ("r", "o"), ("r", "."), ("r", ","),
        ("v", "a"), ("v", "e"), ("v", "o"),
        ("w", "a"), ("w", "e"), ("w", "o"),
        ("y", "a"), ("y", "e"), ("y", "o"), ("y", "."), ("y", ","),
    ]


def get_zero_kerning_pairs() -> List[Tuple[str, str]]:
    """
    Get pairs that typically require little to no kerning.

    These pairs have letters with parallel edges that naturally
    fit together without adjustment.

    Returns:
        List of pairs expected to have near-zero kerning values.
    """
    return [
        # Vertical stems
        ("H", "H"), ("H", "I"), ("H", "N"), ("H", "M"),
        ("I", "I"), ("I", "H"), ("I", "N"), ("I", "M"),
        ("N", "N"), ("N", "H"), ("N", "I"),
        ("M", "M"), ("M", "H"), ("M", "I"), ("M", "N"),
        # Lowercase verticals
        ("n", "n"), ("n", "m"), ("n", "h"), ("n", "i"),
        ("m", "m"), ("m", "n"), ("m", "h"), ("m", "i"),
        ("h", "h"), ("h", "n"), ("h", "m"), ("h", "i"),
        ("i", "i"), ("i", "n"), ("i", "m"), ("i", "h"),
        ("l", "l"), ("l", "i"),
        # Rounds that balance
        ("o", "o"), ("o", "c"), ("o", "e"),
        ("c", "o"), ("e", "o"),
    ]


# =============================================================================
# Default Configuration Instances
# =============================================================================

# Default configurations for easy import
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_config: DatasetConfig
) -> None:
    """
    Validate configuration settings for consistency.

    Args:
        model_config: Model architecture configuration.
        training_config: Training configuration.
        dataset_config: Dataset configuration.

    Raises:
        ValueError: If configuration is invalid.
    """
    # Validate image size
    if model_config.image_size < 16:
        raise ValueError(f"Image size must be at least 16, got {model_config.image_size}")

    # Validate encoder channels
    if len(model_config.encoder_channels) < 1:
        raise ValueError("Must have at least one encoder channel")

    # Validate batch size
    if training_config.batch_size < 1:
        raise ValueError(f"Batch size must be positive, got {training_config.batch_size}")

    # Validate learning rate
    if training_config.learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {training_config.learning_rate}")

    # Validate kerning range
    if dataset_config.kerning_min >= dataset_config.kerning_max:
        raise ValueError(
            f"kerning_min ({dataset_config.kerning_min}) must be less than "
            f"kerning_max ({dataset_config.kerning_max})"
        )

    # Validate validation split
    if not 0 < training_config.val_split < 1:
        raise ValueError(f"val_split must be in (0, 1), got {training_config.val_split}")


if __name__ == "__main__":
    # Print configuration summary when run directly
    print("KerningNet Configuration Summary")
    print("=" * 50)

    print(f"\nModel Configuration:")
    print(f"  Image size: {DEFAULT_MODEL_CONFIG.image_size}x{DEFAULT_MODEL_CONFIG.image_size}")
    print(f"  Encoder channels: {DEFAULT_MODEL_CONFIG.encoder_channels}")
    print(f"  Embedding dim: {DEFAULT_MODEL_CONFIG.embedding_dim}")
    print(f"  Combined dim: {DEFAULT_MODEL_CONFIG.combined_dim}")

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {DEFAULT_TRAINING_CONFIG.batch_size}")
    print(f"  Learning rate: {DEFAULT_TRAINING_CONFIG.learning_rate}")
    print(f"  Epochs: {DEFAULT_TRAINING_CONFIG.epochs}")
    print(f"  Loss function: {DEFAULT_TRAINING_CONFIG.loss_fn}")

    print(f"\nCritical Pairs:")
    all_pairs = get_all_critical_pairs()
    print(f"  Total critical pairs: {len(all_pairs)}")
    print(f"  Negative kerning pairs: {len(get_negative_kerning_pairs())}")
    print(f"  Zero kerning pairs: {len(get_zero_kerning_pairs())}")

    print(f"\nSample critical pairs:")
    for pair in all_pairs[:10]:
        print(f"    '{pair[0]}' + '{pair[1]}'")
    print("    ...")
