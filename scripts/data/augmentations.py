"""Data augmentation transforms for glyph images.

Provides training-time augmentations that simulate natural variation in
handwritten and rendered glyphs without destroying character identity.

Usage:
    from data.augmentations import get_train_transforms, get_eval_transforms

    train_transform = get_train_transforms(image_size=64)
    eval_transform = get_eval_transforms(image_size=64)
"""

import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

try:
    from scipy.ndimage import gaussian_filter, map_coordinates
except ImportError:
    gaussian_filter = None
    map_coordinates = None


class ElasticDeformation:
    """Apply small elastic deformation to simulate handwriting variation.

    Uses a random displacement field smoothed with a Gaussian to produce
    natural-looking warping.
    """

    def __init__(self, alpha: float = 8.0, sigma: float = 3.0):
        """Initialize elastic deformation.

        Args:
            alpha: Displacement magnitude.
            sigma: Gaussian smoothing sigma for the displacement field.
        """
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        if gaussian_filter is None or map_coordinates is None:
            return img  # scipy not available, skip elastic deformation

        w, h = img.size
        rng = np.random.default_rng()

        # Random displacement fields
        dx = rng.standard_normal((h, w)).astype(np.float32)
        dy = rng.standard_normal((h, w)).astype(np.float32)

        # Smooth with Gaussian
        dx = gaussian_filter(dx, self.sigma) * self.alpha
        dy = gaussian_filter(dy, self.sigma) * self.alpha

        # Create coordinate grid
        y, x = np.mgrid[0:h, 0:w]
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)

        # Apply displacement via map_coordinates
        arr = np.array(img, dtype=np.float32)
        result = map_coordinates(arr, [y_new, x_new], order=1, mode="constant", cval=0)

        return Image.fromarray(result.astype(np.uint8), mode=img.mode)


class MorphologyTransform:
    """Apply morphological erosion or dilation to simulate weight variation.

    Erosion thins strokes (lighter weight), dilation thickens them (heavier weight).
    """

    def __init__(self, erosion_prob: float = 0.2, dilation_prob: float = 0.2, kernel_size: int = 3):
        """Initialize morphology transform.

        Args:
            erosion_prob: Probability of applying erosion.
            dilation_prob: Probability of applying dilation.
            kernel_size: Size of the morphological kernel.
        """
        self.erosion_prob = erosion_prob
        self.dilation_prob = dilation_prob
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.random()
        if r < self.erosion_prob:
            return img.filter(ImageFilter.MinFilter(self.kernel_size))
        elif r < self.erosion_prob + self.dilation_prob:
            return img.filter(ImageFilter.MaxFilter(self.kernel_size))
        return img


class GaussianNoise:
    """Add Gaussian noise to a tensor."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, -1.0, 1.0)
        return tensor


class RandomInvert:
    """Randomly invert a grayscale PIL Image."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            from PIL import ImageOps
            return ImageOps.invert(img)
        return img


def get_train_transforms(
    image_size: int = 64,
    rotation: float = 5.0,
    scale: tuple[float, float] = (0.9, 1.1),
    translate: float = 0.05,
    noise_std: float = 0.02,
    elastic: bool = True,
    morphology: bool = True,
) -> transforms.Compose:
    """Build training augmentation pipeline.

    Args:
        image_size: Target image size.
        rotation: Max rotation in degrees.
        scale: Scale range (min, max).
        translate: Translation range as fraction of image size.
        noise_std: Gaussian noise standard deviation.
        elastic: Whether to apply elastic deformation.
        morphology: Whether to apply erosion/dilation.

    Returns:
        Composed transform pipeline.
    """
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]

    # Morphological augmentation (before geometric transforms)
    if morphology:
        transform_list.append(MorphologyTransform(erosion_prob=0.15, dilation_prob=0.15))

    # Elastic deformation
    if elastic:
        transform_list.append(
            transforms.RandomApply([ElasticDeformation(alpha=6.0, sigma=3.0)], p=0.3)
        )

    # Geometric augmentation
    if rotation > 0 or scale != (1.0, 1.0) or translate > 0:
        transform_list.append(
            transforms.RandomAffine(
                degrees=rotation,
                translate=(translate, translate),
                scale=scale,
                fill=0,
            )
        )

    # Brightness/contrast jitter
    transform_list.append(
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2)],
            p=0.3,
        )
    )

    # To tensor
    transform_list.append(transforms.ToTensor())

    # Gaussian noise (applied to tensor)
    if noise_std > 0:
        transform_list.append(GaussianNoise(std=noise_std))

    # Normalize to [-1, 1]
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_list)


def get_eval_transforms(image_size: int = 64) -> transforms.Compose:
    """Build evaluation/inference transform pipeline (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
