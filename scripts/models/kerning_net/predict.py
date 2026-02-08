#!/usr/bin/env python3
"""Inference script for KerningNet kerning prediction.

This module provides utilities for predicting kerning values for glyph pairs
using a trained KerningNet model. It supports:
    - Single pair prediction
    - Batch prediction for multiple pairs
    - Full font kerning table generation
    - CSV output of kerning values

Usage:
    # Predict kerning for a single pair
    python predict.py --left glyph_A.png --right glyph_V.png --checkpoint model.pt

    # Generate kerning table for a font
    python predict.py --font_dir MyFont/ --checkpoint model.pt --output kerning.csv

Example (programmatic):
    >>> from scripts.models.kerning_net.predict import KerningPredictor
    >>> predictor = KerningPredictor("checkpoints/kerning_net/best.pt")
    >>>
    >>> # Single pair
    >>> kerning = predictor.predict_pair("glyph_A.png", "glyph_V.png")
    >>> print(f"Kerning: {kerning:.1f} units")
    >>>
    >>> # Full font
    >>> table = predictor.predict_font("MyFont/")
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import KerningNet, KerningNetConfig, create_kerning_net

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Type aliases
Tensor = torch.Tensor
NDArray = np.ndarray

# Critical kerning pairs that most benefit from adjustment
CRITICAL_PAIRS = [
    ("A", "V"), ("A", "W"), ("A", "Y"), ("A", "T"),
    ("T", "a"), ("T", "e"), ("T", "o"), ("T", "r"),
    ("V", "a"), ("V", "e"), ("V", "o"),
    ("W", "a"), ("W", "e"), ("W", "o"),
    ("Y", "a"), ("Y", "e"), ("Y", "o"),
    ("L", "T"), ("L", "V"), ("L", "W"), ("L", "Y"),
    ("P", "a"), ("P", "e"), ("P", "o"),
    ("F", "a"), ("F", "e"), ("F", "o"),
    ("r", "a"), ("r", "e"), ("r", "o"),
    ("f", "a"), ("f", "e"), ("f", "o"),
]


class KerningPredictor:
    """High-level interface for predicting kerning values.

    Provides methods for predicting kerning for single pairs, batches,
    and full font kerning tables.

    Attributes:
        model: Loaded KerningNet model.
        device: Computation device (CPU/GPU).
        transform: Image preprocessing pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[KerningNetConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize KerningPredictor.

        Args:
            checkpoint_path: Path to trained model checkpoint.
                           If None, creates model with random weights.
            config: Model configuration. If None, uses default.
            device: Computation device. If None, auto-detect.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if checkpoint_path is not None:
            self.model = create_kerning_net(
                config=config,
                checkpoint_path=str(checkpoint_path),
                device=self.device,
            )
        else:
            self.model = create_kerning_net(
                config=config or KerningNetConfig(),
                device=self.device,
            )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        logger.info(f"KerningPredictor initialized on {self.device}")

    def _load_image(self, image: Union[str, Path, Image.Image, NDArray]) -> Tensor:
        """Load and preprocess a glyph image.

        Args:
            image: Input image as path, PIL Image, or numpy array.

        Returns:
            Preprocessed tensor of shape (1, 1, 64, 64).
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("L")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img = Image.fromarray(image.astype(np.uint8), mode="L")
            else:
                img = Image.fromarray(image.astype(np.uint8)).convert("L")
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        tensor = self.transform(img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_pair(
        self,
        left: Union[str, Path, Image.Image, NDArray],
        right: Union[str, Path, Image.Image, NDArray],
    ) -> float:
        """Predict kerning value for a single glyph pair.

        Args:
            left: Left glyph image (path, PIL Image, or numpy array).
            right: Right glyph image (path, PIL Image, or numpy array).

        Returns:
            Predicted kerning value in units per em.
        """
        left_tensor = self._load_image(left)
        right_tensor = self._load_image(right)

        output = self.model(left_tensor, right_tensor)
        return float(output.item())

    @torch.no_grad()
    def predict_batch(
        self,
        pairs: list[tuple[Union[str, Path], Union[str, Path]]],
        batch_size: int = 64,
    ) -> list[float]:
        """Predict kerning values for multiple glyph pairs.

        Args:
            pairs: List of (left_path, right_path) tuples.
            batch_size: Batch size for inference.

        Returns:
            List of predicted kerning values.
        """
        results = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]

            left_batch = []
            right_batch = []
            for left_path, right_path in batch_pairs:
                left_batch.append(self._load_image(left_path).squeeze(0))
                right_batch.append(self._load_image(right_path).squeeze(0))

            left_tensor = torch.stack(left_batch)
            right_tensor = torch.stack(right_batch)

            output = self.model(left_tensor, right_tensor)
            results.extend(output.squeeze(1).cpu().tolist())

        return results

    @torch.no_grad()
    def predict_font(
        self,
        font_dir: Union[str, Path],
        critical_only: bool = True,
    ) -> dict[tuple[str, str], float]:
        """Generate kerning table for a font directory.

        Args:
            font_dir: Path to directory containing glyph images.
                     Images should be named by character (e.g., A.png, V.png).
            critical_only: If True, only predict critical kerning pairs.

        Returns:
            Dictionary mapping (left_char, right_char) to kerning value.
        """
        font_dir = Path(font_dir)
        if not font_dir.exists():
            raise ValueError(f"Font directory does not exist: {font_dir}")

        # Find available glyphs
        glyph_map = {}
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for path in font_dir.glob(ext):
                char = path.stem
                if len(char) == 1:  # Single character filename
                    glyph_map[char] = path

        if not glyph_map:
            raise ValueError(f"No single-character glyph images found in {font_dir}")

        logger.info(f"Found {len(glyph_map)} glyphs in {font_dir}")

        # Determine which pairs to predict
        if critical_only:
            pairs_to_predict = [
                (l, r) for l, r in CRITICAL_PAIRS
                if l in glyph_map and r in glyph_map
            ]
        else:
            chars = sorted(glyph_map.keys())
            pairs_to_predict = [
                (l, r) for l in chars for r in chars
            ]

        # Predict kerning for each pair
        kerning_table = {}
        path_pairs = [
            (glyph_map[l], glyph_map[r]) for l, r in pairs_to_predict
        ]

        if path_pairs:
            values = self.predict_batch(path_pairs)
            for (l, r), val in zip(pairs_to_predict, values):
                kerning_table[(l, r)] = val

        logger.info(f"Predicted kerning for {len(kerning_table)} pairs")
        return kerning_table


def save_kerning_table(
    table: dict[tuple[str, str], float],
    output_path: Union[str, Path],
) -> None:
    """Save kerning table to CSV file.

    Args:
        table: Dictionary mapping (left, right) to kerning value.
        output_path: Path to output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left", "right", "kerning"])
        for (left, right), value in sorted(table.items()):
            writer.writerow([left, right, f"{value:.1f}"])

    logger.info(f"Saved kerning table ({len(table)} pairs) to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict kerning values using KerningNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pair",
        nargs=2,
        type=Path,
        metavar=("LEFT", "RIGHT"),
        help="Predict kerning for a single pair of glyph images",
    )
    input_group.add_argument(
        "--font_dir",
        type=Path,
        help="Directory containing glyph images for kerning table generation",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file for kerning table",
    )
    parser.add_argument(
        "--all_pairs",
        action="store_true",
        help="Predict all pairs, not just critical ones",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for kerning prediction."""
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    predictor = KerningPredictor(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    if args.pair:
        left_path, right_path = args.pair
        kerning = predictor.predict_pair(left_path, right_path)
        print(f"Kerning: {kerning:.1f} units per em")

    elif args.font_dir:
        table = predictor.predict_font(
            args.font_dir,
            critical_only=not args.all_pairs,
        )

        # Print results
        print(f"\nKerning Table ({len(table)} pairs):")
        print("-" * 40)
        for (left, right), value in sorted(table.items()):
            print(f"  {left}{right}: {value:>8.1f}")

        # Save if output specified
        if args.output:
            save_kerning_table(table, args.output)


if __name__ == "__main__":
    main()
