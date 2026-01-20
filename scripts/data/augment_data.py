#!/usr/bin/env python3
"""
Data augmentation for glyph training images.

This script applies various augmentations to glyph images including:
- Rotation (small angles)
- Scale variation
- Stroke weight simulation
- Noise addition
- Elastic deformation for handwriting variation

Usage:
    python augment_data.py --input-dir ./glyphs/images_64 --output-dir ./glyphs_augmented
    python augment_data.py --input-dir ./glyphs/images_64 --output-dir ./glyphs_augmented --factor 5
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GlyphAugmenter:
    """Augment glyph images with various transformations."""

    def __init__(
        self,
        rotation_range: tuple[float, float] = (-5.0, 5.0),
        scale_range: tuple[float, float] = (0.9, 1.1),
        stroke_weight_range: tuple[float, float] = (-1.0, 1.0),
        noise_level: float = 0.02,
        elastic_alpha: float = 20.0,
        elastic_sigma: float = 4.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the augmenter.

        Args:
            rotation_range: Min/max rotation in degrees
            scale_range: Min/max scale factor
            stroke_weight_range: Min/max stroke weight adjustment (pixels)
            noise_level: Gaussian noise standard deviation (0-1)
            elastic_alpha: Elastic deformation intensity
            elastic_sigma: Elastic deformation smoothness
            random_seed: Random seed for reproducibility
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.stroke_weight_range = stroke_weight_range
        self.noise_level = noise_level
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def rotate(
        self,
        image: Image.Image,
        angle: Optional[float] = None
    ) -> Image.Image:
        """
        Rotate image by a small angle.

        Args:
            image: PIL Image
            angle: Rotation angle in degrees (random if None)

        Returns:
            Rotated image
        """
        if angle is None:
            angle = random.uniform(*self.rotation_range)

        # Rotate with white background fill
        rotated = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=255
        )
        return rotated

    def scale(
        self,
        image: Image.Image,
        factor: Optional[float] = None
    ) -> Image.Image:
        """
        Scale image while maintaining canvas size.

        Args:
            image: PIL Image
            factor: Scale factor (random if None)

        Returns:
            Scaled image
        """
        if factor is None:
            factor = random.uniform(*self.scale_range)

        width, height = image.size
        new_width = int(width * factor)
        new_height = int(height * factor)

        # Scale the image
        scaled = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.BICUBIC
        )

        # Create new image at original size
        result = Image.new('L', (width, height), color=255)

        # Center the scaled image
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2

        # Handle scaling up (crop) vs scaling down (pad)
        if factor > 1:
            # Crop center of scaled image
            crop_x = (new_width - width) // 2
            crop_y = (new_height - height) // 2
            scaled = scaled.crop((crop_x, crop_y, crop_x + width, crop_y + height))
            result.paste(scaled, (0, 0))
        else:
            # Paste scaled image in center
            result.paste(scaled, (paste_x, paste_y))

        return result

    def adjust_stroke_weight(
        self,
        image: Image.Image,
        amount: Optional[float] = None
    ) -> Image.Image:
        """
        Simulate stroke weight variation by erosion/dilation.

        Args:
            image: PIL Image
            amount: Positive = thicker strokes, negative = thinner

        Returns:
            Image with adjusted stroke weight
        """
        if amount is None:
            amount = random.uniform(*self.stroke_weight_range)

        # Convert to numpy for morphological operations
        arr = np.array(image)

        # Invert (black becomes white for morphology)
        arr = 255 - arr

        if amount > 0:
            # Dilate (thicken strokes)
            iterations = int(abs(amount))
            if iterations > 0:
                from scipy.ndimage import binary_dilation
                binary = arr > 127
                for _ in range(iterations):
                    binary = binary_dilation(binary)
                arr = binary.astype(np.uint8) * 255
        elif amount < 0:
            # Erode (thin strokes)
            iterations = int(abs(amount))
            if iterations > 0:
                from scipy.ndimage import binary_erosion
                binary = arr > 127
                for _ in range(iterations):
                    binary = binary_erosion(binary)
                arr = binary.astype(np.uint8) * 255

        # Fractional adjustment with blur
        frac = abs(amount) - int(abs(amount))
        if frac > 0:
            blur_radius = frac * 1.5
            blurred = gaussian_filter(arr.astype(float), sigma=blur_radius)

            if amount > 0:
                # Thicken: lower threshold
                threshold = 127 - (frac * 50)
            else:
                # Thin: higher threshold
                threshold = 127 + (frac * 50)

            arr = (blurred > threshold).astype(np.uint8) * 255

        # Invert back
        arr = 255 - arr

        return Image.fromarray(arr)

    def add_noise(
        self,
        image: Image.Image,
        level: Optional[float] = None
    ) -> Image.Image:
        """
        Add Gaussian noise to image.

        Args:
            image: PIL Image
            level: Noise standard deviation (0-1)

        Returns:
            Noisy image
        """
        if level is None:
            level = self.noise_level

        arr = np.array(image).astype(float) / 255.0

        # Add Gaussian noise
        noise = np.random.normal(0, level, arr.shape)
        noisy = arr + noise

        # Clip to valid range
        noisy = np.clip(noisy, 0, 1)

        return Image.fromarray((noisy * 255).astype(np.uint8))

    def elastic_deform(
        self,
        image: Image.Image,
        alpha: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> Image.Image:
        """
        Apply elastic deformation for handwriting-like variation.

        This creates natural-looking distortions similar to human
        handwriting variation.

        Args:
            image: PIL Image
            alpha: Deformation intensity
            sigma: Deformation smoothness

        Returns:
            Deformed image
        """
        if alpha is None:
            alpha = self.elastic_alpha
        if sigma is None:
            sigma = self.elastic_sigma

        arr = np.array(image)
        shape = arr.shape

        # Create random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma
        ) * alpha

        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma
        ) * alpha

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Apply displacement
        indices = [
            np.clip(y + dy, 0, shape[0] - 1),
            np.clip(x + dx, 0, shape[1] - 1)
        ]

        # Interpolate
        deformed = map_coordinates(
            arr,
            indices,
            order=1,
            mode='constant',
            cval=255
        )

        return Image.fromarray(deformed.astype(np.uint8))

    def shear(
        self,
        image: Image.Image,
        shear_x: Optional[float] = None,
        shear_y: Optional[float] = None
    ) -> Image.Image:
        """
        Apply shear transformation.

        Args:
            image: PIL Image
            shear_x: Horizontal shear factor
            shear_y: Vertical shear factor

        Returns:
            Sheared image
        """
        if shear_x is None:
            shear_x = random.uniform(-0.1, 0.1)
        if shear_y is None:
            shear_y = random.uniform(-0.05, 0.05)

        width, height = image.size

        # Affine transformation matrix for shear
        # [1, shear_x, -shear_x*height/2]
        # [shear_y, 1, -shear_y*width/2]
        coeffs = (
            1, shear_x, -shear_x * height / 2,
            shear_y, 1, -shear_y * width / 2
        )

        return image.transform(
            (width, height),
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BICUBIC,
            fillcolor=255
        )

    def translate(
        self,
        image: Image.Image,
        dx: Optional[int] = None,
        dy: Optional[int] = None,
        max_shift: int = 5
    ) -> Image.Image:
        """
        Translate image by a small amount.

        Args:
            image: PIL Image
            dx: Horizontal shift (random if None)
            dy: Vertical shift (random if None)
            max_shift: Maximum shift in pixels

        Returns:
            Translated image
        """
        if dx is None:
            dx = random.randint(-max_shift, max_shift)
        if dy is None:
            dy = random.randint(-max_shift, max_shift)

        return ImageOps.expand(
            image.crop((
                max(-dx, 0),
                max(-dy, 0),
                image.width + min(-dx, 0),
                image.height + min(-dy, 0)
            )),
            border=(
                max(dx, 0),
                max(dy, 0),
                max(-dx, 0),
                max(-dy, 0)
            ),
            fill=255
        )

    def augment(
        self,
        image: Image.Image,
        augmentations: Optional[list[str]] = None
    ) -> Image.Image:
        """
        Apply a random set of augmentations.

        Args:
            image: PIL Image
            augmentations: List of augmentation names to apply

        Returns:
            Augmented image
        """
        available = {
            "rotate": self.rotate,
            "scale": self.scale,
            "stroke": self.adjust_stroke_weight,
            "noise": self.add_noise,
            "elastic": self.elastic_deform,
            "shear": self.shear,
            "translate": self.translate
        }

        if augmentations is None:
            # Default: apply a random subset
            augmentations = random.sample(
                list(available.keys()),
                k=random.randint(1, 4)
            )

        result = image.copy()
        for aug_name in augmentations:
            if aug_name in available:
                result = available[aug_name](result)

        return result

    def generate_variants(
        self,
        image: Image.Image,
        count: int = 5,
        ensure_diverse: bool = True
    ) -> list[Image.Image]:
        """
        Generate multiple augmented variants of an image.

        Args:
            image: PIL Image
            count: Number of variants to generate
            ensure_diverse: Ensure different augmentation combinations

        Returns:
            List of augmented images
        """
        variants = []

        if ensure_diverse:
            # Predefined augmentation combinations for diversity
            combinations = [
                ["rotate"],
                ["scale"],
                ["elastic"],
                ["stroke", "noise"],
                ["rotate", "scale"],
                ["shear", "translate"],
                ["elastic", "noise"],
                ["rotate", "elastic"],
                ["scale", "stroke"],
                ["rotate", "scale", "noise"],
            ]

            for i in range(count):
                augs = combinations[i % len(combinations)]
                variants.append(self.augment(image, augs))
        else:
            for _ in range(count):
                variants.append(self.augment(image))

        return variants


class DatasetAugmenter:
    """Augment an entire dataset of glyph images."""

    def __init__(
        self,
        output_dir: Path,
        augmenter: GlyphAugmenter,
        augmentation_factor: int = 5
    ):
        """
        Initialize the dataset augmenter.

        Args:
            output_dir: Directory to save augmented images
            augmenter: GlyphAugmenter instance
            augmentation_factor: Number of variants per original image
        """
        self.output_dir = Path(output_dir)
        self.augmenter = augmenter
        self.augmentation_factor = augmentation_factor

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def augment_image(
        self,
        image_path: Path,
        skip_existing: bool = True
    ) -> int:
        """
        Augment a single image.

        Args:
            image_path: Path to image file
            skip_existing: Skip if augmented images exist

        Returns:
            Number of variants generated
        """
        # Generate output filename pattern
        stem = image_path.stem
        suffix = image_path.suffix

        # Check if augmented versions exist
        if skip_existing:
            existing = list(self.output_dir.glob(f"{stem}_aug_*.{suffix}"))
            if len(existing) >= self.augmentation_factor:
                return 0

        # Load image
        try:
            image = Image.open(image_path).convert('L')
        except Exception as e:
            logger.warning(f"Could not open {image_path}: {e}")
            return 0

        # Copy original
        original_path = self.output_dir / f"{stem}_orig{suffix}"
        if not original_path.exists():
            image.save(original_path)

        # Generate augmented variants
        variants = self.augmenter.generate_variants(
            image,
            count=self.augmentation_factor,
            ensure_diverse=True
        )

        # Save variants
        saved = 0
        for i, variant in enumerate(variants):
            variant_path = self.output_dir / f"{stem}_aug_{i:02d}{suffix}"
            if not skip_existing or not variant_path.exists():
                variant.save(variant_path)
                saved += 1

        return saved

    def augment_directory(
        self,
        input_dir: Path,
        skip_existing: bool = True
    ) -> tuple[int, int]:
        """
        Augment all images in a directory.

        Args:
            input_dir: Directory containing images
            skip_existing: Skip existing augmented images

        Returns:
            Tuple of (images_processed, variants_generated)
        """
        input_dir = Path(input_dir)

        # Find all image files
        image_files = (
            list(input_dir.glob("*.png")) +
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.jpeg"))
        )

        # Filter out already augmented images
        image_files = [
            f for f in image_files
            if "_aug_" not in f.stem and "_orig" not in f.stem
        ]

        logger.info(f"Found {len(image_files)} images to augment")
        logger.info(f"Augmentation factor: {self.augmentation_factor}")

        images_processed = 0
        total_variants = 0

        for image_path in tqdm(image_files, desc="Augmenting images"):
            variants = self.augment_image(image_path, skip_existing=skip_existing)
            if variants > 0:
                images_processed += 1
                total_variants += variants

        return images_processed, total_variants

    def get_statistics(self) -> dict:
        """Get statistics about augmented dataset."""
        all_images = list(self.output_dir.glob("*.png"))

        stats = {
            "total_images": len(all_images),
            "original_images": len([f for f in all_images if "_orig" in f.stem]),
            "augmented_images": len([f for f in all_images if "_aug_" in f.stem])
        }

        return stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Augment glyph training images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic augmentation with 5 variants per image
    python augment_data.py --input-dir ./glyphs/images_64 --output-dir ./augmented

    # Higher augmentation factor
    python augment_data.py --input-dir ./glyphs/images_64 --output-dir ./augmented --factor 10

    # Custom augmentation parameters
    python augment_data.py --input-dir ./glyphs/images_64 --output-dir ./augmented \\
        --rotation 10 --scale-range 0.85,1.15 --noise 0.05
        """
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing glyph images"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save augmented images"
    )

    parser.add_argument(
        "--factor",
        type=int,
        default=5,
        help="Number of augmented variants per image (default: 5)"
    )

    parser.add_argument(
        "--rotation",
        type=float,
        default=5.0,
        help="Maximum rotation angle in degrees (default: 5)"
    )

    parser.add_argument(
        "--scale-range",
        type=str,
        default="0.9,1.1",
        help="Scale range as min,max (default: 0.9,1.1)"
    )

    parser.add_argument(
        "--stroke-range",
        type=str,
        default="-1,1",
        help="Stroke weight adjustment range (default: -1,1)"
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="Noise level 0-1 (default: 0.02)"
    )

    parser.add_argument(
        "--elastic-alpha",
        type=float,
        default=20.0,
        help="Elastic deformation intensity (default: 20)"
    )

    parser.add_argument(
        "--elastic-sigma",
        type=float,
        default=4.0,
        help="Elastic deformation smoothness (default: 4)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-generate existing augmented images"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse ranges
    scale_range = tuple(float(x) for x in args.scale_range.split(","))
    stroke_range = tuple(float(x) for x in args.stroke_range.split(","))

    # Create augmenter
    augmenter = GlyphAugmenter(
        rotation_range=(-args.rotation, args.rotation),
        scale_range=scale_range,
        stroke_weight_range=stroke_range,
        noise_level=args.noise,
        elastic_alpha=args.elastic_alpha,
        elastic_sigma=args.elastic_sigma,
        random_seed=args.seed
    )

    dataset_augmenter = DatasetAugmenter(
        output_dir=args.output_dir,
        augmenter=augmenter,
        augmentation_factor=args.factor
    )

    if args.stats_only:
        stats = dataset_augmenter.get_statistics()
        print("\nAugmentation Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Original images: {stats['original_images']}")
        print(f"  Augmented images: {stats['augmented_images']}")
        return 0

    try:
        images_processed, total_variants = dataset_augmenter.augment_directory(
            input_dir=args.input_dir,
            skip_existing=not args.no_skip_existing
        )

        logger.info(f"\nAugmentation complete!")
        logger.info(f"Images processed: {images_processed}")
        logger.info(f"Variants generated: {total_variants}")

        stats = dataset_augmenter.get_statistics()
        logger.info(f"Total images in output: {stats['total_images']}")

        return 0

    except Exception as e:
        logger.error(f"Augmentation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
