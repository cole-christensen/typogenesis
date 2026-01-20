#!/usr/bin/env python3
"""
PyTorch Dataset for glyph images.

This dataset loads glyph images and metadata for training diffusion
and other generative models. Supports character conditioning and
style labels.

Usage:
    from datasets.glyph_dataset import GlyphDataset

    dataset = GlyphDataset(
        data_dir="./glyphs",
        image_size=64,
        split="train"
    )

    for image, metadata in dataset:
        # image: torch.Tensor [1, 64, 64]
        # metadata: dict with character, font, etc.
        pass
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

# Standard ASCII characters for conditioning
ASCII_CHARS = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)

# Character to index mapping
CHAR_TO_IDX = {c: i for i, c in enumerate(ASCII_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ASCII_CHARS)}
NUM_CHARS = len(ASCII_CHARS)


def char_to_onehot(char: str, num_classes: int = NUM_CHARS) -> torch.Tensor:
    """Convert character to one-hot encoding."""
    idx = CHAR_TO_IDX.get(char, 0)
    onehot = torch.zeros(num_classes)
    onehot[idx] = 1.0
    return onehot


def char_to_embedding(char: str) -> torch.Tensor:
    """Convert character to learnable embedding index."""
    return torch.tensor(CHAR_TO_IDX.get(char, 0), dtype=torch.long)


class GlyphDataset(Dataset):
    """
    PyTorch Dataset for glyph images.

    Loads glyph images from a directory structure like:
        data_dir/
            images_64/
                font1_U0041.png
                font1_U0042.png
                ...
            outlines/
                font1_U0041.json
                font1_U0042.json
                ...

    Returns:
        image: torch.Tensor of shape [1, H, W] (grayscale)
        metadata: dict containing:
            - character: str
            - unicode: int
            - font_family: str
            - font_style: str
            - char_idx: int (for conditioning)
            - char_onehot: torch.Tensor [NUM_CHARS]
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        image_size: int = 64,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform: Optional[Callable] = None,
        include_augmented: bool = True,
        filter_chars: Optional[str] = None,
        filter_fonts: Optional[list[str]] = None,
        random_seed: int = 42
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory containing images and outlines
            image_size: Size of images to load (64 or 128)
            split: One of "train", "val", "test", or "all"
            split_ratio: Ratio for train/val/test splits
            transform: Optional transform to apply to images
            include_augmented: Include augmented images (with _aug_ in name)
            filter_chars: Only include these characters
            filter_fonts: Only include these font families
            random_seed: Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.split_ratio = split_ratio
        self.transform = transform
        self.include_augmented = include_augmented
        self.filter_chars = set(filter_chars) if filter_chars else None
        self.filter_fonts = set(filter_fonts) if filter_fonts else None

        # Find image directory
        self.image_dir = self.data_dir / f"images_{image_size}"
        if not self.image_dir.exists():
            # Try without size suffix
            self.image_dir = self.data_dir / "images"
            if not self.image_dir.exists():
                raise ValueError(f"Image directory not found: {self.image_dir}")

        # Find outline directory
        self.outline_dir = self.data_dir / "outlines"

        # Load samples
        self.samples = self._load_samples()

        # Apply split
        if split != "all":
            random.seed(random_seed)
            indices = list(range(len(self.samples)))
            random.shuffle(indices)

            n_train = int(len(indices) * split_ratio[0])
            n_val = int(len(indices) * split_ratio[1])

            if split == "train":
                indices = indices[:n_train]
            elif split == "val":
                indices = indices[n_train:n_train + n_val]
            elif split == "test":
                indices = indices[n_train + n_val:]

            self.samples = [self.samples[i] for i in indices]

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_samples(self) -> list[dict]:
        """Load all valid samples from the dataset."""
        samples = []

        # Find all image files
        image_files = list(self.image_dir.glob("*.png"))

        # Filter augmented if needed
        if not self.include_augmented:
            image_files = [
                f for f in image_files
                if "_aug_" not in f.stem and "_orig" not in f.stem
            ]

        for image_path in image_files:
            # Try to load metadata
            stem = image_path.stem

            # Handle augmented images - get original stem
            original_stem = stem
            if "_aug_" in stem:
                original_stem = stem.split("_aug_")[0]
            elif "_orig" in stem:
                original_stem = stem.replace("_orig", "")

            # Load outline/metadata JSON if available
            outline_path = self.outline_dir / f"{original_stem}.json"
            metadata = {}

            if outline_path.exists():
                try:
                    with open(outline_path) as f:
                        metadata = json.load(f)
                except Exception:
                    pass

            # Extract character info from metadata or filename
            character = metadata.get("character", "")
            unicode_val = metadata.get("unicode", 0)
            font_family = metadata.get("font_family", "")
            font_style = metadata.get("font_style", "")

            # Try to extract from filename if not in metadata
            # Format: fontkey_U0041.png
            if not character and "_U" in original_stem:
                try:
                    unicode_hex = original_stem.split("_U")[-1][:4]
                    unicode_val = int(unicode_hex, 16)
                    character = chr(unicode_val)
                except:
                    pass

            # Apply filters
            if self.filter_chars and character not in self.filter_chars:
                continue
            if self.filter_fonts and font_family not in self.filter_fonts:
                continue

            sample = {
                "image_path": str(image_path),
                "outline_path": str(outline_path) if outline_path.exists() else None,
                "character": character,
                "unicode": unicode_val,
                "font_family": font_family,
                "font_style": font_style,
                "is_augmented": "_aug_" in stem or "_orig" in stem
            }

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, metadata_dict)
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert('L')

        # Resize if needed
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BICUBIC
            )

        # Convert to tensor
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # [1, H, W]

        # Apply transform if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Prepare metadata
        char = sample["character"]
        metadata = {
            "character": char,
            "unicode": sample["unicode"],
            "font_family": sample["font_family"],
            "font_style": sample["font_style"],
            "is_augmented": sample["is_augmented"],
            "char_idx": CHAR_TO_IDX.get(char, 0),
            "char_onehot": char_to_onehot(char)
        }

        return image_tensor, metadata

    def get_char_distribution(self) -> dict[str, int]:
        """Get distribution of characters in dataset."""
        dist = {}
        for sample in self.samples:
            char = sample["character"]
            dist[char] = dist.get(char, 0) + 1
        return dict(sorted(dist.items()))

    def get_font_distribution(self) -> dict[str, int]:
        """Get distribution of fonts in dataset."""
        dist = {}
        for sample in self.samples:
            font = sample["font_family"]
            dist[font] = dist.get(font, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, dict]:
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of (image, metadata) tuples

        Returns:
            Tuple of (batched_images, batched_metadata)
        """
        images = torch.stack([item[0] for item in batch])

        # Batch metadata
        metadata = {
            "character": [item[1]["character"] for item in batch],
            "unicode": torch.tensor([item[1]["unicode"] for item in batch]),
            "font_family": [item[1]["font_family"] for item in batch],
            "char_idx": torch.tensor([item[1]["char_idx"] for item in batch]),
            "char_onehot": torch.stack([item[1]["char_onehot"] for item in batch])
        }

        return images, metadata


class GlyphConditionedDataset(GlyphDataset):
    """
    Dataset that returns (image, condition, label) for conditional generation.

    The condition is the character identity (for character-conditioned generation).
    The label is the image itself (for diffusion/autoencoder training).
    """

    def __init__(
        self,
        *args,
        condition_type: str = "onehot",
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize the conditioned dataset.

        Args:
            condition_type: "onehot", "embedding", or "text"
            normalize: Normalize images to [-1, 1] for diffusion models
            *args, **kwargs: Passed to parent GlyphDataset
        """
        super().__init__(*args, **kwargs)
        self.condition_type = condition_type
        self.normalize = normalize

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get a conditioned sample.

        Returns:
            Tuple of (image, condition, metadata)
        """
        image_tensor, metadata = super().__getitem__(idx)

        # Normalize for diffusion models
        if self.normalize:
            image_tensor = image_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # Get condition
        if self.condition_type == "onehot":
            condition = metadata["char_onehot"]
        elif self.condition_type == "embedding":
            condition = torch.tensor(metadata["char_idx"], dtype=torch.long)
        else:  # text
            condition = metadata["character"]

        return image_tensor, condition, metadata


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    image_size: int = 64,
    num_workers: int = 4,
    include_augmented: bool = True,
    **kwargs
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        image_size: Image size
        num_workers: DataLoader workers
        include_augmented: Include augmented images
        **kwargs: Additional arguments for GlyphDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = GlyphDataset(
        data_dir=data_dir,
        image_size=image_size,
        split="train",
        include_augmented=include_augmented,
        **kwargs
    )

    val_dataset = GlyphDataset(
        data_dir=data_dir,
        image_size=image_size,
        split="val",
        include_augmented=False,  # Don't use augmented for validation
        **kwargs
    )

    test_dataset = GlyphDataset(
        data_dir=data_dir,
        image_size=image_size,
        split="test",
        include_augmented=False,  # Don't use augmented for testing
        **kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=GlyphDataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=GlyphDataset.collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=GlyphDataset.collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GlyphDataset")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--image-size", type=int, default=64, help="Image size")
    parser.add_argument("--split", type=str, default="train", help="Split to load")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Test dataset
    dataset = GlyphDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        split=args.split
    )

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")

    char_dist = dataset.get_char_distribution()
    print(f"  Unique characters: {len(char_dist)}")

    font_dist = dataset.get_font_distribution()
    print(f"  Unique fonts: {len(font_dist)}")

    # Test loading a sample
    if len(dataset) > 0:
        image, metadata = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {image.shape}")
        print(f"  Character: {metadata['character']}")
        print(f"  Font: {metadata['font_family']}")
        print(f"  Char index: {metadata['char_idx']}")
