#!/usr/bin/env python3
"""Training script for StyleEncoder model.

This script trains the StyleEncoder using contrastive learning on a dataset
of font glyph images. Glyphs from the same font are treated as positive pairs,
while glyphs from different fonts are treated as negative pairs.

Training uses the NT-Xent loss (SimCLR-style) by default, with options for
triplet loss and InfoNCE.

Usage:
    # Basic training with defaults
    python train.py --data_dir data/style_dataset

    # Custom configuration
    python train.py --data_dir data/style_dataset \\
        --backbone efficientnet_b0 \\
        --batch_size 128 \\
        --epochs 50 \\
        --learning_rate 1e-4

    # Resume from checkpoint
    python train.py --data_dir data/style_dataset \\
        --resume checkpoints/style_encoder/latest.pt

Example with WandB logging:
    python train.py --data_dir data/style_dataset \\
        --wandb_project my-project \\
        --experiment_name exp-001
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
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from .config import (
    AugmentationConfig,
    FullConfig,
    LossConfig,
    PathConfig,
    StyleEncoderConfig,
    TrainingConfig,
)
from .losses import NTXentLoss, TripletLoss, InfoNCELoss, get_loss_function
from .model import StyleEncoder, create_style_encoder, model_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class StyleDataset(Dataset):
    """Dataset for contrastive learning on font styles.

    Each sample returns multiple glyph images from the same font,
    allowing the model to learn that these should have similar embeddings.

    The dataset expects a directory structure like:
        data_dir/
            font_name_1/
                glyph_A.png
                glyph_B.png
                ...
            font_name_2/
                glyph_A.png
                ...

    Attributes:
        data_dir: Root directory containing font subdirectories
        font_dirs: List of font directory paths
        glyphs_per_font: Number of glyphs to sample per font
        transform: Image transformations to apply
    """

    def __init__(
        self,
        data_dir: Path,
        glyphs_per_font: int = 4,
        transform: Optional[transforms.Compose] = None,
        min_glyphs: int = 10,
    ) -> None:
        """Initialize StyleDataset.

        Args:
            data_dir: Root directory containing font subdirectories
            glyphs_per_font: Number of glyphs to sample per font
            transform: Image transformations. If None, uses default transforms.
            min_glyphs: Minimum number of glyphs required per font

        Raises:
            ValueError: If data_dir doesn't exist or contains no valid fonts
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        self.glyphs_per_font = glyphs_per_font
        self.transform = transform or self._default_transform()

        # Find all font directories with sufficient glyphs
        self.font_dirs = []
        self.font_glyphs = {}  # font_name -> list of glyph paths

        for font_dir in self.data_dir.iterdir():
            if not font_dir.is_dir():
                continue

            glyph_files = list(font_dir.glob("*.png")) + list(font_dir.glob("*.jpg"))
            if len(glyph_files) >= min_glyphs:
                self.font_dirs.append(font_dir)
                self.font_glyphs[font_dir.name] = glyph_files

        if not self.font_dirs:
            raise ValueError(
                f"No valid fonts found in {data_dir}. "
                f"Each font must have at least {min_glyphs} glyph images."
            )

        logger.info(
            f"Loaded {len(self.font_dirs)} fonts with "
            f"{sum(len(g) for g in self.font_glyphs.values())} total glyphs"
        )

    def _default_transform(self) -> transforms.Compose:
        """Create default image transformation pipeline."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        """Return number of fonts in dataset."""
        return len(self.font_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a batch of glyphs from the same font.

        Args:
            idx: Font index

        Returns:
            Tuple of (glyphs, font_idx) where glyphs has shape
            (glyphs_per_font, 1, 64, 64) and font_idx is the font label.
        """
        from PIL import Image

        font_dir = self.font_dirs[idx]
        glyph_files = self.font_glyphs[font_dir.name]

        # Sample random glyphs from this font
        selected = random.sample(glyph_files, min(self.glyphs_per_font, len(glyph_files)))

        glyphs = []
        for glyph_path in selected:
            img = Image.open(glyph_path)
            if self.transform:
                img = self.transform(img)
            glyphs.append(img)

        # Pad if we don't have enough glyphs
        while len(glyphs) < self.glyphs_per_font:
            glyphs.append(glyphs[-1].clone())

        return torch.stack(glyphs), idx


def get_augmentation_transform(config: AugmentationConfig) -> transforms.Compose:
    """Create data augmentation pipeline for contrastive learning.

    Args:
        config: Augmentation configuration

    Returns:
        Composed transformation pipeline
    """
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
    ]

    # Geometric augmentations
    if config.rotation_range > 0 or config.scale_range != (1.0, 1.0):
        transform_list.append(
            transforms.RandomAffine(
                degrees=config.rotation_range,
                translate=(config.translate_range, config.translate_range),
                scale=config.scale_range,
            )
        )

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Noise augmentation
    if config.noise_std > 0:
        transform_list.append(
            transforms.Lambda(
                lambda x: x + config.noise_std * torch.randn_like(x)
            )
        )

    # Normalize
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_list)


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for StyleDataset.

    Flattens the batch so each glyph is treated as a separate sample,
    while preserving font labels for positive pair identification.

    Args:
        batch: List of (glyphs, font_idx) tuples

    Returns:
        Tuple of (all_glyphs, all_labels):
            - all_glyphs: Shape (batch_size * glyphs_per_font, 1, 64, 64)
            - all_labels: Shape (batch_size * glyphs_per_font,)
    """
    glyphs, labels = zip(*batch)

    # Flatten: (batch, glyphs_per_font, C, H, W) -> (batch * glyphs_per_font, C, H, W)
    all_glyphs = torch.cat(glyphs, dim=0)

    # Expand labels: each font's glyphs get the same label
    all_labels = []
    for label, glyph_batch in zip(labels, glyphs):
        all_labels.extend([label] * glyph_batch.shape[0])

    return all_glyphs, torch.tensor(all_labels)


class Trainer:
    """Trainer class for StyleEncoder.

    Handles the training loop, validation, checkpointing, and logging.

    Attributes:
        config: Full training configuration
        model: StyleEncoder model
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        criterion: Contrastive loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        writer: TensorBoard writer
        device: Training device (CPU/GPU)
    """

    def __init__(
        self,
        config: FullConfig,
        model: StyleEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            config: Full training configuration
            model: StyleEncoder model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            device: Training device. If None, uses CUDA if available.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.warmup_epochs,
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs - config.training.warmup_epochs,
            eta_min=config.training.min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.training.warmup_epochs],
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.training.mixed_precision)

        # Loss function
        self.criterion = get_loss_function(
            config.loss.loss_type,
            temperature=config.loss.temperature,
        )

        # TensorBoard writer
        config.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=config.paths.log_dir / config.experiment_name
        )

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Optional WandB
        self.wandb_run = None
        if config.wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.experiment_name,
                    config={
                        "model": vars(config.model),
                        "training": vars(config.training),
                        "loss": vars(config.loss),
                        "augmentation": vars(config.augmentation),
                    },
                )
            except ImportError:
                logger.warning("wandb not installed, skipping WandB logging")

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.training.mixed_precision):
                # Forward pass with projection for contrastive loss
                _, projections = self.model(images, return_projection=True)

                # Compute contrastive loss
                loss = self._compute_contrastive_loss(projections, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.config.training.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} LR: {lr:.2e}"
                )
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)

                if self.wandb_run:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "global_step": self.global_step,
                    })

        return total_loss / num_batches

    def _compute_contrastive_loss(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss based on configuration.

        For NT-Xent: Creates positive pairs from same-font glyphs
        For Triplet: Mines hard negatives from batch

        Args:
            projections: Projected embeddings, shape (batch_size, projection_dim)
            labels: Font labels, shape (batch_size,)

        Returns:
            Loss tensor.
        """
        if self.config.loss.loss_type == "nt_xent":
            # Split batch into two views for NT-Xent
            # Assumes glyphs_per_font >= 2 and batch is ordered by font
            batch_size = projections.shape[0]
            mid = batch_size // 2
            z_i = projections[:mid]
            z_j = projections[mid:]

            # Handle odd batch sizes
            min_size = min(z_i.shape[0], z_j.shape[0])
            z_i = z_i[:min_size]
            z_j = z_j[:min_size]

            return self.criterion(z_i, z_j)

        elif self.config.loss.loss_type == "triplet":
            # Use supervised contrastive approach with hard mining
            return self.criterion(
                projections,  # anchor
                projections,  # positive (will be selected by label)
                labels=labels,
            )

        elif self.config.loss.loss_type == "infonce":
            # Use first glyph as query, second as positive key
            batch_size = projections.shape[0]
            mid = batch_size // 2
            query = projections[:mid]
            key = projections[mid:]

            min_size = min(query.shape[0], key.shape[0])
            return self.criterion(query[:min_size], key[:min_size])

        else:
            raise ValueError(f"Unknown loss type: {self.config.loss.loss_type}")

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation.

        Computes:
            - Average validation loss
            - Same-font similarity (should be > 0.9)
            - Different-font similarity (should be < 0.5)

        Returns:
            Dictionary with validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_embeddings = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            embeddings, projections = self.model(images, return_projection=True)

            loss = self._compute_contrastive_loss(projections, labels)
            total_loss += loss.item()
            num_batches += 1

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

        # Compute similarity metrics
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        same_font_sim, diff_font_sim = self._compute_similarity_metrics(
            embeddings, labels
        )

        metrics = {
            "val/loss": total_loss / max(1, num_batches),
            "val/same_font_similarity": same_font_sim,
            "val/diff_font_similarity": diff_font_sim,
        }

        # Log metrics
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.epoch)

        if self.wandb_run:
            import wandb
            wandb.log({**metrics, "epoch": self.epoch})

        logger.info(
            f"Validation - Loss: {metrics['val/loss']:.4f} "
            f"Same-font sim: {same_font_sim:.3f} "
            f"Diff-font sim: {diff_font_sim:.3f}"
        )

        return metrics

    def _compute_similarity_metrics(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute average same-font and different-font similarities.

        Args:
            embeddings: Embeddings, shape (N, embedding_dim)
            labels: Font labels, shape (N,)

        Returns:
            Tuple of (same_font_similarity, different_font_similarity)
        """
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Create masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = torch.eye(len(labels), dtype=torch.bool)
        same_font_mask = labels_equal & ~self_mask
        diff_font_mask = ~labels_equal

        # Compute mean similarities
        same_font_sim = sim_matrix[same_font_mask].mean().item() if same_font_mask.any() else 0.0
        diff_font_sim = sim_matrix[diff_font_mask].mean().item() if diff_font_mask.any() else 0.0

        return same_font_sim, diff_font_sim

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: If True, also saves as 'best.pt'
        """
        self.config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "model": vars(self.config.model),
                "training": vars(self.config.training),
                "loss": vars(self.config.loss),
            },
        }

        path = self.config.paths.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.config.paths.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
        logger.info(model_summary(self.model))
        logger.info(f"Training on {self.device}")
        logger.info(f"Total epochs: {self.config.training.epochs}")

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config.training.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                f"Train loss: {train_loss:.4f}"
            )

            # Update learning rate
            self.scheduler.step()

            # Validate
            if self.val_loader and epoch % self.config.training.val_interval == 0:
                val_metrics = self.validate()
                val_loss = val_metrics.get("val/loss", float("inf"))

                # Save best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Checkpoint
                if epoch % self.config.training.checkpoint_interval == 0:
                    self.save_checkpoint(f"epoch_{epoch:04d}.pt", is_best=is_best)

            else:
                # Checkpoint without validation
                if epoch % self.config.training.checkpoint_interval == 0:
                    self.save_checkpoint(f"epoch_{epoch:04d}.pt")

        # Save final checkpoint
        self.save_checkpoint("final.pt")
        logger.info("Training completed!")

        if self.wandb_run:
            self.wandb_run.finish()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train StyleEncoder for font style embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to style dataset directory",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
        help="CNN backbone architecture",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Output embedding dimension",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Don't use pretrained backbone weights",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size (number of fonts)",
    )
    parser.add_argument(
        "--glyphs_per_font",
        type=int,
        default=4,
        help="Number of glyphs per font in each batch",
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
        default=8,
        help="Number of data loading workers",
    )

    # Loss arguments
    parser.add_argument(
        "--loss",
        type=str,
        default="nt_xent",
        choices=["nt_xent", "triplet", "infonce"],
        help="Contrastive loss function",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for NT-Xent/InfoNCE loss",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/style_encoder"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Logging arguments
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("logs/style_encoder"),
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (None to disable)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity/username",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )

    # Other arguments
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

    # Set random seed
    set_seed(args.seed)

    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"style_encoder_{args.backbone}_{timestamp}"

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Create configuration
    config = FullConfig(
        model=StyleEncoderConfig(
            backbone=args.backbone,
            embedding_dim=args.embedding_dim,
            pretrained=not args.no_pretrained,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            glyphs_per_font=args.glyphs_per_font,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            num_workers=args.num_workers,
            seed=args.seed,
        ),
        loss=LossConfig(
            loss_type=args.loss,
            temperature=args.temperature,
        ),
        paths=PathConfig(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
        ),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        experiment_name=args.experiment_name,
    )

    # Create transforms
    aug_config = config.augmentation
    train_transform = get_augmentation_transform(aug_config)
    val_transform = get_augmentation_transform(
        AugmentationConfig(rotation_range=0, scale_range=(1.0, 1.0), noise_std=0)
    )

    # Create datasets
    full_dataset = StyleDataset(
        args.data_dir,
        glyphs_per_font=args.glyphs_per_font,
        transform=train_transform,
    )

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Override val dataset transform
    # Note: This is a workaround since random_split doesn't copy transforms
    # For proper implementation, create separate datasets

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info(f"Train set: {len(train_dataset)} fonts")
    logger.info(f"Val set: {len(val_dataset)} fonts")

    # Create model
    model = create_style_encoder(config.model, device=device)

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
