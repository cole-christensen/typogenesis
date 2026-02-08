#!/usr/bin/env python3
"""Training script for KerningNet model.

This script trains the KerningNet using a dataset of glyph pair images
with ground-truth kerning values. The model learns to predict optimal
kerning adjustments in units per em.

Training uses L1 (smooth) loss for regression, with mixed precision
training, gradient clipping, and cosine LR schedule.

Usage:
    # Basic training with defaults
    python train.py --data_dir data/kerning_dataset

    # Custom configuration
    python train.py --data_dir data/kerning_dataset \\
        --batch_size 128 \\
        --epochs 50 \\
        --learning_rate 1e-4

    # Resume from checkpoint
    python train.py --data_dir data/kerning_dataset \\
        --resume checkpoints/kerning_net/latest.pt
"""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from .model import KerningNet, KerningNetConfig, create_kerning_net

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class KerningDataset(Dataset):
    """Dataset for kerning prediction training.

    Each sample returns a pair of glyph images (left, right) and
    the ground-truth kerning value in units per em.

    The dataset expects a directory structure with:
        - left_glyphs/: Directory of left glyph images
        - right_glyphs/: Directory of right glyph images
        - kerning_values.csv: CSV with columns (left, right, kerning)

    Or a flat structure with pairs:
        data_dir/
            pairs.csv  (columns: left_path, right_path, kerning_value)

    Attributes:
        pairs: List of (left_path, right_path, kerning_value) tuples.
        transform: Image transformation pipeline.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[object] = None,
    ) -> None:
        """Initialize KerningDataset.

        Args:
            data_dir: Root directory containing the kerning dataset.
            split: Dataset split ("train", "val", or "test").
            transform: Image transformations. If None, uses default transforms.

        Raises:
            ValueError: If data_dir doesn't exist or contains no valid pairs.
        """
        from torchvision import transforms

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.pairs = self._load_pairs(split)

        if not self.pairs:
            raise ValueError(
                f"No valid kerning pairs found in {data_dir} for split '{split}'."
            )

        logger.info(f"Loaded {len(self.pairs)} kerning pairs for split '{split}'")

    def _load_pairs(self, split: str) -> list[tuple[Path, Path, float]]:
        """Load kerning pairs from CSV or directory structure.

        Args:
            split: Dataset split name.

        Returns:
            List of (left_path, right_path, kerning_value) tuples.
        """
        import csv

        pairs = []
        csv_path = self.data_dir / f"{split}_pairs.csv"

        if not csv_path.exists():
            csv_path = self.data_dir / "pairs.csv"

        if csv_path.exists():
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    left_path = self.data_dir / row["left_path"]
                    right_path = self.data_dir / row["right_path"]
                    kerning = float(row["kerning_value"])

                    if left_path.exists() and right_path.exists():
                        pairs.append((left_path, right_path, kerning))

        return pairs

    def __len__(self) -> int:
        """Return number of kerning pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a kerning pair sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (left_glyph, right_glyph, kerning_value):
                - left_glyph: Shape (1, 64, 64)
                - right_glyph: Shape (1, 64, 64)
                - kerning_value: Scalar tensor
        """
        from PIL import Image

        left_path, right_path, kerning = self.pairs[idx]

        left_img = Image.open(left_path).convert("L")
        right_img = Image.open(right_path).convert("L")

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        kerning_tensor = torch.tensor(kerning, dtype=torch.float32)

        return left_img, right_img, kerning_tensor


class DummyKerningDataset(Dataset):
    """Dummy dataset for testing when real data is not available.

    Generates synthetic glyph pairs with random kerning values.
    """

    def __init__(
        self,
        size: int = 10000,
        image_size: int = 64,
    ) -> None:
        """Initialize dummy dataset.

        Args:
            size: Number of samples.
            image_size: Image size in pixels.
        """
        self.size = size
        self.image_size = image_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a dummy kerning pair.

        Returns:
            Tuple of (left_glyph, right_glyph, kerning_value).
        """
        left = torch.randn(1, self.image_size, self.image_size).clamp(0, 1)
        right = torch.randn(1, self.image_size, self.image_size).clamp(0, 1)
        # Kerning values typically in range [-200, 50] for 1000 upm
        kerning = torch.tensor(random.uniform(-200.0, 50.0), dtype=torch.float32)
        return left, right, kerning


class Trainer:
    """Trainer class for KerningNet.

    Handles training loop, validation, checkpointing, and logging.

    Attributes:
        model: KerningNet model.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        scaler: Gradient scaler for mixed precision.
        criterion: Loss function (Smooth L1).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Training device (CPU/GPU).
    """

    def __init__(
        self,
        model: KerningNet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        gradient_clip_norm: float = 1.0,
        mixed_precision: bool = True,
        checkpoint_dir: Path = Path("checkpoints/kerning_net"),
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            model: KerningNet model to train.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization weight decay.
            epochs: Number of training epochs.
            warmup_epochs: Number of warmup epochs.
            min_lr: Minimum learning rate.
            gradient_clip_norm: Maximum gradient norm.
            mixed_precision: Whether to use AMP.
            checkpoint_dir: Directory for saving checkpoints.
            device: Training device. If None, auto-detect.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        # Mixed precision scaler
        self.scaler = GradScaler(self.device.type, enabled=mixed_precision)

        # Loss function: Smooth L1 is more robust to outliers than MSE
        self.criterion = nn.SmoothL1Loss()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (left, right, target) in enumerate(self.train_loader):
            left = left.to(self.device)
            right = right.to(self.device)
            target = target.to(self.device).unsqueeze(1)  # (B,) -> (B, 1)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.mixed_precision):
                predicted = self.model(left, right)
                loss = self.criterion(predicted, target)

            self.scaler.scale(loss).backward()

            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} LR: {lr:.2e}"
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation.

        Returns:
            Dictionary with validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for left, right, target in self.val_loader:
            left = left.to(self.device)
            right = right.to(self.device)
            target = target.to(self.device).unsqueeze(1)

            predicted = self.model(left, right)
            loss = self.criterion(predicted, target)
            mae = torch.abs(predicted - target).mean()

            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

        metrics = {
            "val/loss": total_loss / max(1, num_batches),
            "val/mae": total_mae / max(1, num_batches),
        }

        logger.info(
            f"Validation - Loss: {metrics['val/loss']:.4f} "
            f"MAE: {metrics['val/mae']:.2f} units"
        )

        return metrics

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
            is_best: If True, also saves as 'best.pt'.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": vars(self.model.config),
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self) -> None:
        """Run full training loop."""
        logger.info(f"Training KerningNet on {self.device}")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")
        logger.info(f"Total epochs: {self.epochs}")

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                f"Train loss: {train_loss:.4f}"
            )

            self.scheduler.step()

            # Validate
            if self.val_loader and epoch % 1 == 0:
                val_metrics = self.validate()
                val_loss = val_metrics.get("val/loss", float("inf"))

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                if epoch % 5 == 0:
                    self.save_checkpoint(f"epoch_{epoch:04d}.pt", is_best=is_best)
            else:
                if epoch % 5 == 0:
                    self.save_checkpoint(f"epoch_{epoch:04d}.pt")

        self.save_checkpoint("final.pt")
        logger.info("Training completed!")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train KerningNet for kerning prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to kerning dataset directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/kerning_net"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA training",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Create model
    config = KerningNetConfig()
    model = create_kerning_net(config, device=device)

    # Create datasets
    try:
        train_dataset = KerningDataset(args.data_dir, split="train")
        val_dataset = KerningDataset(args.data_dir, split="val")
    except (ValueError, FileNotFoundError):
        logger.warning("Real dataset not found, using dummy dataset for development")
        train_dataset = DummyKerningDataset(size=10000)
        val_dataset = DummyKerningDataset(size=1000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Train set: {len(train_dataset)} pairs")
    logger.info(f"Val set: {len(val_dataset)} pairs")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
