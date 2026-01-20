#!/usr/bin/env python3
"""
PyTorch Dataset for style pairs (contrastive learning).

This dataset loads pairs of glyphs from the same or different fonts
for contrastive learning of style embeddings.

Usage:
    from datasets.style_dataset import StylePairDataset

    dataset = StylePairDataset(
        data_dir="./glyphs",
        image_size=64,
        split="train"
    )

    for glyph1, glyph2, same_font in dataset:
        # glyph1: torch.Tensor [1, 64, 64]
        # glyph2: torch.Tensor [1, 64, 64]
        # same_font: torch.Tensor [1] (1 if same font, 0 otherwise)
        pass
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class StylePairDataset(Dataset):
    """
    PyTorch Dataset for style pairs (contrastive learning).

    For each sample, returns two glyphs and a label indicating
    whether they're from the same font (style).

    This is designed for training style encoders using contrastive
    loss (e.g., triplet loss, contrastive loss, or InfoNCE).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        image_size: int = 64,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform: Optional[Callable] = None,
        positive_ratio: float = 0.5,
        same_char_for_negative: bool = True,
        num_pairs: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize the style pair dataset.

        Args:
            data_dir: Root directory containing images and outlines
            image_size: Size of images to load
            split: One of "train", "val", "test", or "all"
            split_ratio: Ratio for train/val/test splits
            transform: Optional transform to apply to images
            positive_ratio: Ratio of same-font pairs vs different-font pairs
            same_char_for_negative: For negative pairs, use same character
            num_pairs: Total number of pairs to generate (None = auto)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.split_ratio = split_ratio
        self.transform = transform
        self.positive_ratio = positive_ratio
        self.same_char_for_negative = same_char_for_negative

        # Find image directory
        self.image_dir = self.data_dir / f"images_{image_size}"
        if not self.image_dir.exists():
            self.image_dir = self.data_dir / "images"
            if not self.image_dir.exists():
                raise ValueError(f"Image directory not found: {self.image_dir}")

        # Find outline directory
        self.outline_dir = self.data_dir / "outlines"

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load and organize samples by font and character
        self._load_samples()

        # Generate pairs
        self.pairs = self._generate_pairs(num_pairs)

        # Apply split
        if split != "all":
            indices = list(range(len(self.pairs)))
            random.shuffle(indices)

            n_train = int(len(indices) * split_ratio[0])
            n_val = int(len(indices) * split_ratio[1])

            if split == "train":
                indices = indices[:n_train]
            elif split == "val":
                indices = indices[n_train:n_train + n_val]
            elif split == "test":
                indices = indices[n_train + n_val:]

            self.pairs = [self.pairs[i] for i in indices]

        logger.info(f"Generated {len(self.pairs)} style pairs for {split} split")

    def _load_samples(self):
        """Load and organize samples by font and character."""
        # samples_by_font[font_family][character] = [image_paths]
        self.samples_by_font = defaultdict(lambda: defaultdict(list))

        # samples_by_char[character][font_family] = [image_paths]
        self.samples_by_char = defaultdict(lambda: defaultdict(list))

        # Find all image files (excluding augmented)
        image_files = [
            f for f in self.image_dir.glob("*.png")
            if "_aug_" not in f.stem and "_orig" not in f.stem
        ]

        for image_path in image_files:
            # Try to get metadata
            stem = image_path.stem
            outline_path = self.outline_dir / f"{stem}.json"

            character = ""
            font_family = ""

            if outline_path.exists():
                try:
                    with open(outline_path) as f:
                        metadata = json.load(f)
                        character = metadata.get("character", "")
                        font_family = metadata.get("font_family", "")
                except:
                    pass

            # Try to extract from filename if not in metadata
            if not character and "_U" in stem:
                try:
                    unicode_hex = stem.split("_U")[-1][:4]
                    unicode_val = int(unicode_hex, 16)
                    character = chr(unicode_val)
                except:
                    pass

            if not font_family:
                # Use part of filename before _U as font key
                if "_U" in stem:
                    font_family = stem.split("_U")[0]
                else:
                    font_family = "unknown"

            if character and font_family:
                self.samples_by_font[font_family][character].append(str(image_path))
                self.samples_by_char[character][font_family].append(str(image_path))

        self.font_families = list(self.samples_by_font.keys())
        self.characters = list(self.samples_by_char.keys())

        logger.info(f"Found {len(self.font_families)} fonts with {len(self.characters)} characters")

    def _generate_pairs(self, num_pairs: Optional[int] = None) -> list[dict]:
        """Generate pairs of glyphs."""
        pairs = []

        # Default number of pairs
        if num_pairs is None:
            # Approximately 2x the number of individual samples
            total_samples = sum(
                len(chars)
                for chars in self.samples_by_font.values()
            )
            num_pairs = total_samples * 2

        num_positive = int(num_pairs * self.positive_ratio)
        num_negative = num_pairs - num_positive

        # Generate positive pairs (same font, different characters)
        logger.info(f"Generating {num_positive} positive pairs...")
        positive_count = 0
        attempts = 0
        max_attempts = num_positive * 10

        while positive_count < num_positive and attempts < max_attempts:
            attempts += 1

            # Pick a random font with multiple characters
            font = random.choice(self.font_families)
            chars = list(self.samples_by_font[font].keys())

            if len(chars) < 2:
                continue

            # Pick two different characters
            char1, char2 = random.sample(chars, 2)

            # Pick images
            img1 = random.choice(self.samples_by_font[font][char1])
            img2 = random.choice(self.samples_by_font[font][char2])

            pairs.append({
                "image1_path": img1,
                "image2_path": img2,
                "font1": font,
                "font2": font,
                "char1": char1,
                "char2": char2,
                "same_font": True
            })
            positive_count += 1

        # Generate negative pairs (different fonts)
        logger.info(f"Generating {num_negative} negative pairs...")
        negative_count = 0
        attempts = 0
        max_attempts = num_negative * 10

        while negative_count < num_negative and attempts < max_attempts:
            attempts += 1

            if len(self.font_families) < 2:
                break

            # Pick two different fonts
            font1, font2 = random.sample(self.font_families, 2)

            if self.same_char_for_negative:
                # Find a character that exists in both fonts
                chars1 = set(self.samples_by_font[font1].keys())
                chars2 = set(self.samples_by_font[font2].keys())
                common_chars = list(chars1 & chars2)

                if not common_chars:
                    continue

                char = random.choice(common_chars)
                char1, char2 = char, char
            else:
                # Pick any characters
                char1 = random.choice(list(self.samples_by_font[font1].keys()))
                char2 = random.choice(list(self.samples_by_font[font2].keys()))

            # Pick images
            img1 = random.choice(self.samples_by_font[font1][char1])
            img2 = random.choice(self.samples_by_font[font2][char2])

            pairs.append({
                "image1_path": img1,
                "image2_path": img2,
                "font1": font1,
                "font2": font2,
                "char1": char1,
                "char2": char2,
                "same_font": False
            })
            negative_count += 1

        # Shuffle pairs
        random.shuffle(pairs)

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        image = Image.open(path).convert('L')

        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BICUBIC
            )

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Get a pair sample.

        Returns:
            Tuple of (image1, image2, same_font_label, metadata)
        """
        pair = self.pairs[idx]

        image1 = self._load_image(pair["image1_path"])
        image2 = self._load_image(pair["image2_path"])

        same_font = torch.tensor([1.0 if pair["same_font"] else 0.0])

        metadata = {
            "font1": pair["font1"],
            "font2": pair["font2"],
            "char1": pair["char1"],
            "char2": pair["char2"],
            "same_font": pair["same_font"]
        }

        return image1, image2, same_font, metadata

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        images1 = torch.stack([item[0] for item in batch])
        images2 = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])

        metadata = {
            "font1": [item[3]["font1"] for item in batch],
            "font2": [item[3]["font2"] for item in batch],
            "char1": [item[3]["char1"] for item in batch],
            "char2": [item[3]["char2"] for item in batch],
            "same_font": [item[3]["same_font"] for item in batch]
        }

        return images1, images2, labels, metadata


class StyleTripletDataset(Dataset):
    """
    Dataset returning (anchor, positive, negative) triplets for triplet loss.

    - Anchor: A glyph from font A
    - Positive: Different glyph from font A (same style)
    - Negative: Glyph from font B (different style)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        image_size: int = 64,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform: Optional[Callable] = None,
        num_triplets: Optional[int] = None,
        hard_negative: bool = False,
        random_seed: int = 42
    ):
        """
        Initialize the triplet dataset.

        Args:
            data_dir: Root directory containing images
            image_size: Size of images to load
            split: One of "train", "val", "test", or "all"
            split_ratio: Ratio for train/val/test splits
            transform: Optional transform to apply to images
            num_triplets: Number of triplets to generate
            hard_negative: Use hard negative mining (same character)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.transform = transform
        self.hard_negative = hard_negative

        # Find image directory
        self.image_dir = self.data_dir / f"images_{image_size}"
        if not self.image_dir.exists():
            self.image_dir = self.data_dir / "images"

        self.outline_dir = self.data_dir / "outlines"

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load samples
        self._load_samples()

        # Generate triplets
        self.triplets = self._generate_triplets(num_triplets)

        # Apply split
        if split != "all":
            indices = list(range(len(self.triplets)))
            random.shuffle(indices)

            n_train = int(len(indices) * split_ratio[0])
            n_val = int(len(indices) * split_ratio[1])

            if split == "train":
                indices = indices[:n_train]
            elif split == "val":
                indices = indices[n_train:n_train + n_val]
            elif split == "test":
                indices = indices[n_train + n_val:]

            self.triplets = [self.triplets[i] for i in indices]

        logger.info(f"Generated {len(self.triplets)} triplets for {split} split")

    def _load_samples(self):
        """Load and organize samples by font and character."""
        self.samples_by_font = defaultdict(lambda: defaultdict(list))
        self.samples_by_char = defaultdict(lambda: defaultdict(list))

        image_files = [
            f for f in self.image_dir.glob("*.png")
            if "_aug_" not in f.stem and "_orig" not in f.stem
        ]

        for image_path in image_files:
            stem = image_path.stem
            outline_path = self.outline_dir / f"{stem}.json"

            character = ""
            font_family = ""

            if outline_path.exists():
                try:
                    with open(outline_path) as f:
                        metadata = json.load(f)
                        character = metadata.get("character", "")
                        font_family = metadata.get("font_family", "")
                except:
                    pass

            if not character and "_U" in stem:
                try:
                    unicode_hex = stem.split("_U")[-1][:4]
                    unicode_val = int(unicode_hex, 16)
                    character = chr(unicode_val)
                except:
                    pass

            if not font_family and "_U" in stem:
                font_family = stem.split("_U")[0]

            if character and font_family:
                self.samples_by_font[font_family][character].append(str(image_path))
                self.samples_by_char[character][font_family].append(str(image_path))

        self.font_families = list(self.samples_by_font.keys())
        self.characters = list(self.samples_by_char.keys())

    def _generate_triplets(self, num_triplets: Optional[int] = None) -> list[dict]:
        """Generate (anchor, positive, negative) triplets."""
        triplets = []

        if num_triplets is None:
            total_samples = sum(
                len(chars)
                for chars in self.samples_by_font.values()
            )
            num_triplets = total_samples

        count = 0
        attempts = 0
        max_attempts = num_triplets * 20

        while count < num_triplets and attempts < max_attempts:
            attempts += 1

            if len(self.font_families) < 2:
                break

            # Pick anchor font
            anchor_font = random.choice(self.font_families)
            chars = list(self.samples_by_font[anchor_font].keys())

            if len(chars) < 2:
                continue

            # Pick anchor and positive (same font, different chars)
            anchor_char, positive_char = random.sample(chars, 2)

            anchor_path = random.choice(self.samples_by_font[anchor_font][anchor_char])
            positive_path = random.choice(self.samples_by_font[anchor_font][positive_char])

            # Pick negative (different font)
            negative_font = random.choice([f for f in self.font_families if f != anchor_font])

            if self.hard_negative:
                # Hard negative: same character, different font
                if anchor_char in self.samples_by_font[negative_font]:
                    negative_char = anchor_char
                else:
                    negative_char = random.choice(list(self.samples_by_font[negative_font].keys()))
            else:
                negative_char = random.choice(list(self.samples_by_font[negative_font].keys()))

            negative_path = random.choice(self.samples_by_font[negative_font][negative_char])

            triplets.append({
                "anchor_path": anchor_path,
                "positive_path": positive_path,
                "negative_path": negative_path,
                "anchor_font": anchor_font,
                "negative_font": negative_font,
                "anchor_char": anchor_char,
                "positive_char": positive_char,
                "negative_char": negative_char
            })
            count += 1

        random.shuffle(triplets)
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        image = Image.open(path).convert('L')

        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BICUBIC
            )

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Get a triplet sample.

        Returns:
            Tuple of (anchor, positive, negative, metadata)
        """
        triplet = self.triplets[idx]

        anchor = self._load_image(triplet["anchor_path"])
        positive = self._load_image(triplet["positive_path"])
        negative = self._load_image(triplet["negative_path"])

        metadata = {
            "anchor_font": triplet["anchor_font"],
            "negative_font": triplet["negative_font"],
            "anchor_char": triplet["anchor_char"],
            "positive_char": triplet["positive_char"],
            "negative_char": triplet["negative_char"]
        }

        return anchor, positive, negative, metadata

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        anchors = torch.stack([item[0] for item in batch])
        positives = torch.stack([item[1] for item in batch])
        negatives = torch.stack([item[2] for item in batch])

        metadata = {
            "anchor_font": [item[3]["anchor_font"] for item in batch],
            "negative_font": [item[3]["negative_font"] for item in batch],
            "anchor_char": [item[3]["anchor_char"] for item in batch],
            "positive_char": [item[3]["positive_char"] for item in batch],
            "negative_char": [item[3]["negative_char"] for item in batch]
        }

        return anchors, positives, negatives, metadata


def create_style_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    image_size: int = 64,
    num_workers: int = 4,
    triplet_loss: bool = False,
    **kwargs
):
    """
    Create train/val/test dataloaders for style learning.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        image_size: Image size
        num_workers: DataLoader workers
        triplet_loss: Use triplet dataset instead of pair dataset
        **kwargs: Additional arguments for dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    DatasetClass = StyleTripletDataset if triplet_loss else StylePairDataset

    train_dataset = DatasetClass(
        data_dir=data_dir,
        image_size=image_size,
        split="train",
        **kwargs
    )

    val_dataset = DatasetClass(
        data_dir=data_dir,
        image_size=image_size,
        split="val",
        **kwargs
    )

    test_dataset = DatasetClass(
        data_dir=data_dir,
        image_size=image_size,
        split="test",
        **kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DatasetClass.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DatasetClass.collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DatasetClass.collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test StylePairDataset")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--image-size", type=int, default=64, help="Image size")
    parser.add_argument("--split", type=str, default="train", help="Split to load")
    parser.add_argument("--triplet", action="store_true", help="Test triplet dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.triplet:
        dataset = StyleTripletDataset(
            data_dir=args.data_dir,
            image_size=args.image_size,
            split=args.split
        )

        print(f"\nStyleTripletDataset Statistics:")
        print(f"  Total triplets: {len(dataset)}")

        if len(dataset) > 0:
            anchor, positive, negative, metadata = dataset[0]
            print(f"\nSample 0:")
            print(f"  Anchor shape: {anchor.shape}")
            print(f"  Anchor font: {metadata['anchor_font']}")
            print(f"  Negative font: {metadata['negative_font']}")
    else:
        dataset = StylePairDataset(
            data_dir=args.data_dir,
            image_size=args.image_size,
            split=args.split
        )

        print(f"\nStylePairDataset Statistics:")
        print(f"  Total pairs: {len(dataset)}")

        # Count positive/negative
        positive_count = sum(1 for p in dataset.pairs if p["same_font"])
        print(f"  Positive pairs (same font): {positive_count}")
        print(f"  Negative pairs (different font): {len(dataset) - positive_count}")

        if len(dataset) > 0:
            img1, img2, label, metadata = dataset[0]
            print(f"\nSample 0:")
            print(f"  Image1 shape: {img1.shape}")
            print(f"  Image2 shape: {img2.shape}")
            print(f"  Same font: {label.item()}")
            print(f"  Font1: {metadata['font1']}")
            print(f"  Font2: {metadata['font2']}")
