"""
GlyphDiffusion: Flow-Matching Diffusion Model for Glyph Generation.

This package implements a flow-matching diffusion model for generating
font glyphs conditioned on character identity and style embeddings.

Modules:
    config: Configuration dataclasses and character utilities
    model: UNet architecture with conditioning
    noise_schedule: Flow-matching schedule and training utilities
    train: Training loop with mixed precision, EMA, and logging
    sample: Inference and sampling utilities

Quick Start:
    # Training
    python -m glyph_diffusion.train --config default --epochs 100

    # Sampling
    python -m glyph_diffusion.sample --checkpoint checkpoints/best.pt --charset lowercase

Example Usage:
    from glyph_diffusion import GlyphDiffusionModel, Config, GlyphSampler

    # Create and train model
    config = Config.default()
    model = GlyphDiffusionModel(config.model)

    # Or load pretrained and sample
    sampler = GlyphSampler("checkpoints/best.pt")
    images, _ = sampler.generate("ABC")

Architecture:
    - UNet with skip connections for image-to-image generation
    - Sinusoidal time embedding for noise level conditioning
    - Learned character embeddings (62 chars: a-z, A-Z, 0-9)
    - AdaIN for style conditioning from 128-dim embeddings
    - Optional mask conditioning for partial completion

Training:
    - Flow-matching objective (velocity prediction)
    - Mixed precision (fp16) for efficiency
    - EMA model averaging for stable sampling
    - Gradient clipping and cosine LR schedule
    - Wandb/TensorBoard logging support

Performance Targets:
    - < 2s per glyph on Apple M1 (after CoreML conversion)
    - Recognizable glyphs after 50k training steps
    - Style consistency across character sets
"""

from .config import (
    Config,
    ModelConfig,
    TrainingConfig,
    FlowMatchingConfig,
    SamplingConfig,
    DataConfig,
    Resolution,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    ALL_CHARACTERS,
    char_to_index,
    index_to_char,
    get_character_set,
)

from .model import (
    GlyphDiffusionModel,
    UNet,
    create_model,
)

from .noise_schedule import (
    FlowMatchingSchedule,
    FlowMatchingScheduler,
    FlowMatchingLoss,
    prepare_training_batch,
    sample_euler,
)

from .sample import (
    GlyphSampler,
    tensor_to_image,
    save_image,
    save_numpy,
    create_grid,
)

__version__ = "0.1.0"
__author__ = "Typogenesis Team"

__all__ = [
    # Config
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "FlowMatchingConfig",
    "SamplingConfig",
    "DataConfig",
    "Resolution",
    "CHAR_TO_IDX",
    "IDX_TO_CHAR",
    "ALL_CHARACTERS",
    "char_to_index",
    "index_to_char",
    "get_character_set",
    # Model
    "GlyphDiffusionModel",
    "UNet",
    "create_model",
    # Noise Schedule
    "FlowMatchingSchedule",
    "FlowMatchingScheduler",
    "FlowMatchingLoss",
    "prepare_training_batch",
    "sample_euler",
    # Sampling
    "GlyphSampler",
    "tensor_to_image",
    "save_image",
    "save_numpy",
    "create_grid",
]
