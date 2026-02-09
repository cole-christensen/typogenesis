"""KerningDataset for kerning prediction training.

Loads glyph pair images and kerning values from a JSONL manifest.
Kerning values are normalized to [-1, 1] by dividing by the font's UPM.

This dataset replaces the inline KerningDataset in models/kerning_net/train.py.
"""

import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KerningDataset(Dataset):
    """Kerning prediction dataset.

    Each sample returns a pair of glyph images and a normalized kerning value.

    Returns:
        Tuple of (left_glyph, right_glyph, kerning_value):
            - left_glyph: Tensor[1, 64, 64]
            - right_glyph: Tensor[1, 64, 64]
            - kerning_value: scalar Tensor (normalized to [-1, 1])
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 64,
        augment: bool = True,
        max_kerning: float = 200.0,
    ) -> None:
        """Initialize KerningDataset.

        Args:
            data_dir: Directory containing JSONL manifests and extracted data.
            split: Dataset split ("train", "val", or "test").
            image_size: Expected image size.
            augment: Whether to apply data augmentation.
            max_kerning: Maximum kerning value for normalization. Kerning values
                        are divided by this to produce [-1, 1] range.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.max_kerning = max_kerning

        # Load manifest
        manifest_path = self.data_dir / f"kerning_{split}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run prepare_datasets.py first."
            )

        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest_path}")

        logger.info(f"Loaded {len(self.entries)} kerning pairs for split '{split}'")

        # Validate kerning range
        if self.entries:
            actual_max = max(abs(e["kerning"]) for e in self.entries)
            if actual_max > self.max_kerning:
                logger.warning(
                    f"Found kerning values up to {actual_max:.1f}, exceeding max_kerning={self.max_kerning}. "
                    f"Values will be clipped. Consider increasing max_kerning."
                )

        # Count nonzero vs zero pairs
        nonzero = sum(1 for e in self.entries if e["kerning"] != 0)
        logger.info(f"  Non-zero kerning: {nonzero}, Zero kerning: {len(self.entries) - nonzero}")

        # Set up transforms
        if self.augment:
            from data.augmentations import get_train_transforms
            self.transform = get_train_transforms(
                image_size=image_size,
                rotation=3.0,       # Less rotation for kerning (alignment matters)
                scale=(0.95, 1.05),  # Less scale variation
                translate=0.02,
                elastic=False,       # No elastic for kerning
                morphology=True,     # Weight variation is relevant
            )
        else:
            from data.augmentations import get_eval_transforms
            self.transform = get_eval_transforms(image_size=image_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]

        # Load images
        left_path = self.data_dir / entry["left_image"]
        right_path = self.data_dir / entry["right_image"]

        try:
            left_img = Image.open(left_path).convert("L")
            right_img = Image.open(right_path).convert("L")
        except Exception as e:
            logger.warning(f"Failed to load kerning images: {e}, using blank")
            left_img = Image.new("L", (self.image_size, self.image_size), 0)
            right_img = Image.new("L", (self.image_size, self.image_size), 0)

        left_tensor = self.transform(left_img)
        right_tensor = self.transform(right_img)

        # Normalize kerning value
        raw_kerning = entry["kerning"]
        # Normalize: divide by max_kerning to get roughly [-1, 1]
        normalized_kerning = raw_kerning / self.max_kerning
        # Clamp to prevent outliers
        normalized_kerning = max(-1.0, min(1.0, normalized_kerning))

        kerning_tensor = torch.tensor(normalized_kerning, dtype=torch.float32)

        return left_tensor, right_tensor, kerning_tensor
