"""
Sampling and inference for GlyphDiffusion model.

This module provides functions and CLI for generating glyphs:
- Load trained model checkpoint
- Generate single glyph given (character, style_embedding)
- Generate full alphabet with consistent style
- Support different sampling steps (10, 25, 50, 100)
- Output as PNG and numpy array
- Batch generation for efficiency

Usage:
    # Generate a single character
    python sample.py --checkpoint checkpoints/best.pt --char "A" --output output.png

    # Generate full lowercase alphabet
    python sample.py --checkpoint checkpoints/best.pt --charset lowercase --output-dir outputs/

    # Generate with custom style embedding
    python sample.py --checkpoint checkpoints/best.pt --char "B" --style-file style.npy --steps 100

    # Generate full alphabet with a reference font style
    python sample.py --checkpoint checkpoints/best.pt --charset all --style-font "path/to/font.ttf"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from config import (
    Config,
    ModelConfig,
    SamplingConfig,
    FlowMatchingConfig,
    CHAR_TO_IDX,
    ALL_CHARACTERS,
    get_character_set,
    char_to_index,
)
from model import GlyphDiffusionModel, create_model
from noise_schedule import FlowMatchingScheduler, sample_euler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GlyphSampler:
    """High-level interface for generating glyphs.

    This class handles model loading, style embedding, and batch generation.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[torch.device] = None,
        use_ema: bool = True,
    ):
        """Initialize sampler from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            device: Device to run inference on.
            use_ema: Whether to use EMA weights (recommended).
        """
        self.device = device or self._get_best_device()
        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Handle both checkpoint formats: full training checkpoint vs bare state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Full training checkpoint format
            config_dict = checkpoint.get("config", {})
            model_config = ModelConfig(**config_dict.get("model", {}))
            flow_config = FlowMatchingConfig(**config_dict.get("flow_matching", {}))

            # Create model
            self.model = create_model(model_config).to(self.device)

            # Load weights (EMA or regular)
            if use_ema and "ema_state_dict" in checkpoint:
                logger.info("Loading EMA weights")
                ema_state = checkpoint["ema_state_dict"]
                for name, param in self.model.named_parameters():
                    if name in ema_state["shadow"]:
                        param.data.copy_(ema_state["shadow"][name])
            else:
                logger.info("Loading regular weights")
                self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Bare state dict - use default config
            logger.info("Loading bare state dict (no config found, using defaults)")
            model_config = ModelConfig()
            flow_config = FlowMatchingConfig()
            self.model = create_model(model_config).to(self.device)
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)

        self.model.eval()

        # Create scheduler
        self.scheduler = FlowMatchingScheduler(flow_config)

        # Store config
        self.config = model_config
        self.flow_config = flow_config

        logger.info(f"Model loaded: {self.model.num_parameters():,} parameters")
        logger.info(f"Image size: {self.config.image_size}x{self.config.image_size}")

    def _get_best_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.no_grad()
    def generate(
        self,
        characters: Union[str, list[str]],
        style_embedding: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        save_intermediates: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """Generate glyph images for given characters.

        Args:
            characters: Single character string or list of characters.
            style_embedding: Style embedding of shape (style_dim,) or (batch, style_dim).
                If None, uses random style.
            num_steps: Number of inference steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            mask: Optional mask for partial completion.
            save_intermediates: Whether to return intermediate steps.

        Returns:
            Tuple of (generated images, intermediate steps if requested).
            Images are shape (batch, 1, height, width) in range [0, 1].
        """
        # Convert characters to indices
        if isinstance(characters, str):
            characters = list(characters)

        batch_size = len(characters)
        char_indices = torch.tensor(
            [char_to_index(c) for c in characters],
            dtype=torch.long,
            device=self.device,
        )

        # Handle style embedding
        if style_embedding is None:
            style_embedding = torch.randn(
                batch_size, self.config.style_embed_dim, device=self.device
            )
        else:
            style_embedding = style_embedding.to(self.device)
            if style_embedding.dim() == 1:
                # Broadcast single style to batch
                style_embedding = style_embedding.unsqueeze(0).expand(batch_size, -1)

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        # Generate initial noise
        noise = torch.randn(
            batch_size,
            1,
            self.config.image_size,
            self.config.image_size,
            device=self.device,
        )

        # Handle mask
        if mask is not None:
            mask = mask.to(self.device)
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)

        # Run sampling
        samples, intermediates = sample_euler(
            model=self.model,
            noise=noise,
            char_indices=char_indices,
            style_embed=style_embedding,
            scheduler=self.scheduler,
            mask=mask,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            save_intermediates=save_intermediates,
        )

        # Clamp to valid range
        samples = torch.clamp(samples, 0, 1)

        return samples, intermediates

    def generate_alphabet(
        self,
        charset: str = "lowercase",
        style_embedding: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        batch_size: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Generate a full alphabet with consistent style.

        Args:
            charset: Character set name ("lowercase", "uppercase", "digits", "all").
            style_embedding: Style embedding to use for all characters.
            num_steps: Number of inference steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed.
            batch_size: Batch size for generation.

        Returns:
            Dictionary mapping characters to generated images.
        """
        characters = get_character_set(charset)
        logger.info(f"Generating {len(characters)} characters from {charset} set")

        # Use consistent style for all characters
        if style_embedding is None:
            if seed is not None:
                torch.manual_seed(seed)
            style_embedding = torch.randn(
                1, self.config.style_embed_dim, device=self.device
            )

        results = {}

        # Generate in batches
        for i in range(0, len(characters), batch_size):
            batch_chars = characters[i : i + batch_size]
            logger.info(
                f"Generating batch {i // batch_size + 1}/{(len(characters) + batch_size - 1) // batch_size}: "
                f"{''.join(batch_chars)}"
            )

            samples, _ = self.generate(
                batch_chars,
                style_embedding=style_embedding,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,  # Use same seed for consistent noise pattern
            )

            for j, char in enumerate(batch_chars):
                results[char] = samples[j]

        return results


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.

    Args:
        tensor: Tensor of shape (1, H, W) or (H, W) in range [0, 1].

    Returns:
        PIL Image in grayscale mode.
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save tensor as PNG image.

    Args:
        tensor: Tensor of shape (1, H, W) or (H, W).
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_image(tensor)
    image.save(path)
    logger.info(f"Saved image to {path}")


def save_numpy(tensor: torch.Tensor, path: Path) -> None:
    """Save tensor as numpy array.

    Args:
        tensor: Tensor to save.
        path: Output path (.npy).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.cpu().numpy()
    np.save(path, array)
    logger.info(f"Saved numpy array to {path}")


def create_grid(images: dict[str, torch.Tensor], cols: int = 8) -> Image.Image:
    """Create a grid of character images.

    Args:
        images: Dictionary mapping characters to tensors.
        cols: Number of columns in grid.

    Returns:
        PIL Image with character grid.
    """
    # Get image size from first image
    first_img = next(iter(images.values()))
    img_size = first_img.shape[-1]

    # Calculate grid dimensions
    num_images = len(images)
    rows = (num_images + cols - 1) // cols

    # Create output image
    grid = Image.new("L", (cols * img_size, rows * img_size), color=255)

    # Place images
    for idx, (char, tensor) in enumerate(sorted(images.items())):
        row = idx // cols
        col = idx % cols
        img = tensor_to_image(tensor)
        grid.paste(img, (col * img_size, row * img_size))

    return grid


def load_style_from_font(font_path: Path, model_config: ModelConfig) -> torch.Tensor:
    """Extract style embedding from a font file.

    This is a placeholder that generates a random style.
    In a full implementation, this would:
    1. Render glyphs from the font
    2. Run them through a StyleEncoder model
    3. Return the style embedding

    Args:
        font_path: Path to TTF/OTF font file.
        model_config: Model configuration.

    Returns:
        Style embedding tensor.
    """
    logger.warning(
        f"Style extraction from font not yet implemented. "
        f"Using random style for {font_path}"
    )
    # Generate deterministic style based on font path hash
    seed = hash(str(font_path)) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(model_config.style_embed_dim)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate glyphs with GlyphDiffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )

    # Character specification (mutually exclusive)
    char_group = parser.add_mutually_exclusive_group(required=True)
    char_group.add_argument(
        "--char",
        type=str,
        help="Single character to generate",
    )
    char_group.add_argument(
        "--chars",
        type=str,
        help="Multiple characters to generate (e.g., 'ABC')",
    )
    char_group.add_argument(
        "--charset",
        type=str,
        choices=["lowercase", "uppercase", "digits", "letters", "all"],
        help="Predefined character set to generate",
    )

    # Style specification
    parser.add_argument(
        "--style-file",
        type=Path,
        help="Path to .npy file with style embedding",
    )
    parser.add_argument(
        "--style-font",
        type=Path,
        help="Path to TTF/OTF font to extract style from",
    )
    parser.add_argument(
        "--random-style",
        action="store_true",
        help="Use random style embedding",
    )

    # Sampling parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        choices=[10, 25, 50, 100],
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Output specification
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for single image (PNG)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for multiple images",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["png", "numpy", "both"],
        default="png",
        help="Output format",
    )
    parser.add_argument(
        "--save-grid",
        type=Path,
        help="Save character grid to this path",
    )

    # Advanced options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use regular weights instead of EMA",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate denoising steps",
    )

    # Partial completion
    parser.add_argument(
        "--mask",
        type=Path,
        help="Path to mask image for partial completion",
    )
    parser.add_argument(
        "--partial-image",
        type=Path,
        help="Path to partial image for completion",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Initialize sampler
    sampler = GlyphSampler(
        checkpoint_path=args.checkpoint,
        use_ema=not args.no_ema,
    )

    # Load style embedding
    style_embedding = None
    if args.style_file:
        logger.info(f"Loading style from {args.style_file}")
        style_embedding = torch.from_numpy(np.load(args.style_file)).float()
    elif args.style_font:
        style_embedding = load_style_from_font(args.style_font, sampler.config)

    # Load mask for partial completion
    mask = None
    if args.mask:
        mask_img = Image.open(args.mask).convert("L")
        mask_img = mask_img.resize(
            (sampler.config.image_size, sampler.config.image_size)
        )
        mask = torch.from_numpy(np.array(mask_img)).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0)

    # Determine what to generate
    if args.char:
        characters = [args.char]
    elif args.chars:
        characters = list(args.chars)
    else:
        characters = get_character_set(args.charset)

    logger.info(f"Generating {len(characters)} character(s)")

    # Generate
    if len(characters) == 1:
        # Single character
        samples, intermediates = sampler.generate(
            characters,
            style_embedding=style_embedding,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            mask=mask,
            save_intermediates=args.save_intermediates,
        )

        # Save output
        output_path = args.output or Path(f"glyph_{characters[0]}.png")

        if args.output_format in ["png", "both"]:
            save_image(samples[0], output_path.with_suffix(".png"))

        if args.output_format in ["numpy", "both"]:
            save_numpy(samples[0], output_path.with_suffix(".npy"))

        if args.save_intermediates and intermediates:
            inter_dir = output_path.parent / f"{output_path.stem}_intermediates"
            for i, inter in enumerate(intermediates):
                save_image(inter[0], inter_dir / f"step_{i:03d}.png")

    else:
        # Multiple characters or alphabet
        if args.charset:
            results = sampler.generate_alphabet(
                charset=args.charset,
                style_embedding=style_embedding,
                num_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                batch_size=args.batch_size,
            )
        else:
            samples, _ = sampler.generate(
                characters,
                style_embedding=style_embedding,
                num_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
            )
            results = {char: samples[i] for i, char in enumerate(characters)}

        # Save outputs
        output_dir = args.output_dir or Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        for char, tensor in results.items():
            # Use character name for special characters
            char_name = char if char.isalnum() else f"char_{ord(char):04x}"

            if args.output_format in ["png", "both"]:
                save_image(tensor, output_dir / f"{char_name}.png")

            if args.output_format in ["numpy", "both"]:
                save_numpy(tensor, output_dir / f"{char_name}.npy")

        # Save grid
        if args.save_grid:
            grid = create_grid(results)
            grid.save(args.save_grid)
            logger.info(f"Saved grid to {args.save_grid}")

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
