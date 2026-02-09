"""StyleDataset for contrastive learning.

Loads font glyph collections from a JSONL manifest. Each sample returns
N glyphs from the same font for forming contrastive pairs.

This dataset replaces the inline StyleDataset in models/style_encoder/train.py.
"""

import json
import logging
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StyleDataset(Dataset):
    """Contrastive learning dataset for StyleEncoder.

    Each sample returns N glyph images from the same font. During training,
    the collate function flattens these into a batch where same-font glyphs
    form positive pairs and cross-font glyphs form negatives.

    Returns:
        Tuple of (glyphs, font_idx):
            - glyphs: Tensor[N, 1, 64, 64]
            - font_idx: int (index into the font list)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        glyphs_per_font: int = 4,
        image_size: int = 64,
        augment: bool = True,
        min_glyphs: int = 10,
    ) -> None:
        """Initialize StyleDataset.

        Args:
            data_dir: Directory containing JSONL manifests and extracted data.
            split: Dataset split ("train", "val", or "test").
            glyphs_per_font: Number of glyphs to sample per font per batch item.
            image_size: Expected image size.
            augment: Whether to apply data augmentation.
            min_glyphs: Minimum number of glyphs required per font.
        """
        self.data_dir = Path(data_dir)
        self.glyphs_per_font = glyphs_per_font
        self.image_size = image_size
        self.augment = augment and split == "train"

        # Load manifest
        manifest_path = self.data_dir / f"style_{split}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run prepare_datasets.py first."
            )

        self.fonts = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("num_glyphs", 0) >= min_glyphs:
                        self.fonts.append(entry)

        if not self.fonts:
            raise ValueError(f"No fonts with >= {min_glyphs} glyphs in {manifest_path}")

        logger.info(f"Loaded {len(self.fonts)} fonts for split '{split}'")

        # Set up transforms
        if self.augment:
            from data.augmentations import get_train_transforms
            self.transform = get_train_transforms(image_size=image_size)
        else:
            from data.augmentations import get_eval_transforms
            self.transform = get_eval_transforms(image_size=image_size)

    def __len__(self) -> int:
        return len(self.fonts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get N glyphs from the same font.

        Args:
            idx: Font index.

        Returns:
            Tuple of (glyphs, font_idx) where glyphs has shape
            (glyphs_per_font, 1, image_size, image_size).
        """
        font_entry = self.fonts[idx]
        glyph_paths = font_entry["glyphs"]

        # Sample random glyphs from this font
        n = min(self.glyphs_per_font, len(glyph_paths))
        selected = random.sample(glyph_paths, n)

        glyphs = []
        for rel_path in selected:
            img_path = self.data_dir / rel_path
            try:
                img = Image.open(img_path).convert("L")
            except Exception as e:
                logger.warning(f"Failed to load style image {img_path}: {e}, using blank")
                img = Image.new("L", (self.image_size, self.image_size), 0)
            tensor = self.transform(img)
            glyphs.append(tensor)

        # Pad if needed (random sampling to avoid duplicate pairs)
        while len(glyphs) < self.glyphs_per_font:
            pad_idx = random.randint(0, len(glyphs) - 1)
            glyphs.append(glyphs[pad_idx].clone())

        return torch.stack(glyphs), idx


def style_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function that flattens glyph batches for contrastive learning.

    Takes a batch of (glyphs, font_idx) tuples and produces flat tensors
    where the font label tracks which glyphs belong to the same font.

    Args:
        batch: List of (glyphs, font_idx) tuples from StyleDataset.__getitem__.

    Returns:
        Tuple of (all_glyphs, all_labels):
            - all_glyphs: Shape (batch_size * glyphs_per_font, 1, H, W)
            - all_labels: Shape (batch_size * glyphs_per_font,) with font indices
    """
    glyphs, labels = zip(*batch, strict=True)

    # glyphs: list of (N, 1, H, W) tensors -> cat to (B*N, 1, H, W)
    all_glyphs = torch.cat(glyphs, dim=0)

    # labels: expand each font_idx to match its N glyphs
    glyphs_per_font = glyphs[0].shape[0]
    all_labels = torch.tensor(labels).repeat_interleave(glyphs_per_font)

    return all_glyphs, all_labels
