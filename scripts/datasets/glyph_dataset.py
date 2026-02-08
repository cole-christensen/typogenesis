"""GlyphDataset for diffusion model training.

Loads glyph images from a JSONL manifest file produced by prepare_datasets.py.
Each sample returns an image tensor, character index, and style embedding.

This dataset is imported by models/glyph_diffusion/train.py.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlyphDataset(Dataset):
    """Real glyph dataset for diffusion training.

    Loads from JSONL manifests produced by data.prepare_datasets.

    Each sample returns a dict with:
        - "image": Tensor[1, image_size, image_size] in [-1, 1]
        - "char_index": int in [0, 61]
        - "style_embed": Tensor[128] (precomputed or statistical fallback)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 64,
        augment: bool = True,
        style_embed_dir: str | Path | None = None,
    ) -> None:
        """Initialize GlyphDataset.

        Args:
            data_dir: Directory containing JSONL manifests and extracted data.
            split: Dataset split ("train", "val", or "test").
            image_size: Expected image size.
            augment: Whether to apply data augmentation.
            style_embed_dir: Optional directory with precomputed style embeddings.
                           If None, uses statistical fallback (pixel statistics).
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment and split == "train"

        # Load manifest
        manifest_path = self.data_dir / f"glyph_{split}.jsonl"
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

        logger.info(f"Loaded {len(self.entries)} glyph entries for split '{split}'")

        # Set up transforms
        if self.augment:
            from data.augmentations import get_train_transforms
            self.transform = get_train_transforms(image_size=image_size)
        else:
            from data.augmentations import get_eval_transforms
            self.transform = get_eval_transforms(image_size=image_size)

        # Load precomputed style embeddings if available
        self.style_embeds = {}
        if style_embed_dir:
            self._load_style_embeddings(Path(style_embed_dir))

    def _load_style_embeddings(self, embed_dir: Path) -> None:
        """Load precomputed style embeddings from .npy files."""
        for npy_file in embed_dir.glob("*.npy"):
            font_id = npy_file.stem
            self.style_embeds[font_id] = torch.from_numpy(
                np.load(npy_file)
            ).float()
        logger.info(f"Loaded style embeddings for {len(self.style_embeds)} fonts")

    def _compute_statistical_style(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Compute a statistical style embedding from pixel values.

        A simple fallback when no trained StyleEncoder is available.
        Uses image statistics (mean, std, histogram features) padded to 128 dims.

        Args:
            img_tensor: Image tensor of shape (1, H, W).

        Returns:
            128-dim style embedding tensor.
        """
        flat = img_tensor.flatten()
        features = []

        # Basic statistics
        features.extend([flat.mean(), flat.std(), flat.min(), flat.max()])

        # Histogram (32 bins)
        hist = torch.histc(flat, bins=32, min=-1.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)  # Normalize
        features.extend(hist.tolist())

        # Spatial statistics (quadrant means)
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        mid_h, mid_w = h // 2, w // 2
        features.append(img_tensor[0, :mid_h, :mid_w].mean())
        features.append(img_tensor[0, :mid_h, mid_w:].mean())
        features.append(img_tensor[0, mid_h:, :mid_w].mean())
        features.append(img_tensor[0, mid_h:, mid_w:].mean())

        # Row and column density profiles (downsampled)
        row_density = img_tensor[0].mean(dim=1)  # (H,)
        col_density = img_tensor[0].mean(dim=0)  # (W,)
        # Downsample to 16 each via averaging
        row_ds = torch.nn.functional.adaptive_avg_pool1d(
            row_density.unsqueeze(0).unsqueeze(0), 16
        ).squeeze()
        col_ds = torch.nn.functional.adaptive_avg_pool1d(
            col_density.unsqueeze(0).unsqueeze(0), 16
        ).squeeze()
        features.extend(row_ds.tolist())
        features.extend(col_ds.tolist())

        # Pad or truncate to 128
        embed = torch.tensor(features, dtype=torch.float32)
        if len(embed) < 128:
            embed = torch.nn.functional.pad(embed, (0, 128 - len(embed)))
        else:
            embed = embed[:128]

        # L2 normalize
        embed = embed / (embed.norm() + 1e-8)
        return embed

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        # Load image
        img_path = self.data_dir / entry["image"]
        img = Image.open(img_path)
        img_tensor = self.transform(img)

        # Character index
        char_index = entry["char_index"]

        # Style embedding
        font_id = entry["font_id"]
        if font_id in self.style_embeds:
            style_embed = self.style_embeds[font_id]
        else:
            style_embed = self._compute_statistical_style(img_tensor)

        return {
            "image": img_tensor,
            "char_index": char_index,
            "style_embed": style_embed,
        }
