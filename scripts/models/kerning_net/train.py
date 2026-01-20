#!/usr/bin/env python3
"""
KerningNet Training Script

Training loop for the KerningNet Siamese CNN model with:
- Configurable loss functions (MSE, Huber, SmoothL1)
- Learning rate scheduling (cosine, step, plateau)
- Gradient clipping
- Checkpointing
- Optional Weights & Biases / TensorBoard logging
- Early stopping
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.kerning_net.config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    validate_config,
)
from models.kerning_net.model import KerningNet, KerningNetWithAuxiliary, create_model


# =============================================================================
# Dummy Dataset (for testing without actual data)
# =============================================================================

class DummyKerningDataset(Dataset):
    """
    Dummy dataset for testing the training pipeline.

    Generates random glyph images with synthetic kerning values.
    Replace with actual KerningDataset from Workstream A.
    """

    def __init__(
        self,
        size: int = 10000,
        image_size: int = 64,
        config: Optional[DatasetConfig] = None,
    ) -> None:
        """
        Initialize dummy dataset.

        Args:
            size: Number of samples to generate.
            image_size: Size of glyph images.
            config: Dataset configuration.
        """
        self.size = size
        self.image_size = image_size
        self.config = config or DEFAULT_DATASET_CONFIG

        # Pre-generate random data for consistency
        self._seed = 42
        rng = np.random.default_rng(self._seed)

        # Generate synthetic kerning values
        # Mix of negative (60%), zero (30%), and positive (10%)
        self.kerning_values = []
        for _ in range(size):
            r = rng.random()
            if r < 0.6:
                # Negative kerning (typical)
                self.kerning_values.append(rng.uniform(-150, -10))
            elif r < 0.9:
                # Near-zero kerning
                self.kerning_values.append(rng.uniform(-5, 5))
            else:
                # Positive kerning (rare)
                self.kerning_values.append(rng.uniform(5, 50))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary with:
                - left_glyph: (1, H, W) tensor
                - right_glyph: (1, H, W) tensor
                - metrics: (4,) tensor
                - kerning: (1,) tensor (normalized)
        """
        # Generate random glyph-like images
        rng = np.random.default_rng(self._seed + idx)

        left = self._generate_glyph_image(rng)
        right = self._generate_glyph_image(rng)

        # Normalize kerning to [-1, 1]
        raw_kerning = self.kerning_values[idx]
        normalized_kerning = self.config.normalize_kerning(raw_kerning)

        # Generate random metrics
        metrics = torch.tensor([
            rng.uniform(0.3, 0.8),   # left advance width / UPM
            rng.uniform(0.0, 0.15),  # right LSB / UPM
            rng.uniform(0.4, 0.6),   # x-height ratio
            rng.uniform(0.65, 0.75), # cap height ratio
        ], dtype=torch.float32)

        return {
            "left_glyph": torch.from_numpy(left).unsqueeze(0).float(),
            "right_glyph": torch.from_numpy(right).unsqueeze(0).float(),
            "metrics": metrics,
            "kerning": torch.tensor([normalized_kerning], dtype=torch.float32),
        }

    def _generate_glyph_image(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a synthetic glyph-like image."""
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Add some random shapes to simulate a glyph
        for _ in range(rng.integers(2, 6)):
            # Random rectangle or ellipse
            x1 = rng.integers(10, self.image_size - 20)
            y1 = rng.integers(10, self.image_size - 20)
            x2 = x1 + rng.integers(10, 30)
            y2 = y1 + rng.integers(15, 40)
            x2 = min(x2, self.image_size - 1)
            y2 = min(y2, self.image_size - 1)
            img[y1:y2, x1:x2] = 1.0

        # Normalize to [0, 1]
        if img.max() > 0:
            img = img / img.max()

        return img


# =============================================================================
# Loss Functions
# =============================================================================

def get_loss_function(config: TrainingConfig) -> nn.Module:
    """
    Get loss function based on configuration.

    Args:
        config: Training configuration.

    Returns:
        Loss function module.
    """
    if config.loss_fn == "mse":
        return nn.MSELoss()
    elif config.loss_fn == "huber":
        return nn.HuberLoss(delta=config.huber_delta)
    elif config.loss_fn == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss_fn}")


class AuxiliaryLoss(nn.Module):
    """
    Combined loss for main kerning prediction and auxiliary tasks.
    """

    def __init__(
        self,
        config: TrainingConfig,
        kerning_weight: float = 1.0,
        needs_kerning_weight: float = 0.3,
        direction_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.main_loss = get_loss_function(config)
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.direction_loss = nn.CrossEntropyLoss()

        self.kerning_weight = kerning_weight
        self.needs_kerning_weight = needs_kerning_weight
        self.direction_weight = direction_weight

    def forward(
        self,
        kerning_pred: torch.Tensor,
        needs_kerning_pred: torch.Tensor,
        direction_pred: torch.Tensor,
        kerning_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            kerning_pred: Predicted kerning values.
            needs_kerning_pred: Predicted needs_kerning logits.
            direction_pred: Predicted direction logits.
            kerning_target: Target kerning values.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        # Main kerning loss
        loss_kerning = self.main_loss(kerning_pred, kerning_target)

        # Needs kerning: |target| > 0.1 (in normalized space)
        needs_kerning_target = (kerning_target.abs() > 0.1).float()
        loss_needs_kerning = self.classification_loss(needs_kerning_pred, needs_kerning_target)

        # Direction: 0=negative, 1=zero, 2=positive
        direction_target = torch.zeros(kerning_target.size(0), dtype=torch.long, device=kerning_target.device)
        direction_target[kerning_target.squeeze() < -0.1] = 0
        direction_target[(kerning_target.squeeze() >= -0.1) & (kerning_target.squeeze() <= 0.1)] = 1
        direction_target[kerning_target.squeeze() > 0.1] = 2
        loss_direction = self.direction_loss(direction_pred, direction_target)

        # Combined loss
        total_loss = (
            self.kerning_weight * loss_kerning +
            self.needs_kerning_weight * loss_needs_kerning +
            self.direction_weight * loss_direction
        )

        components = {
            "loss_kerning": loss_kerning.item(),
            "loss_needs_kerning": loss_needs_kerning.item(),
            "loss_direction": loss_direction.item(),
        }

        return total_loss, components


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

def get_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> Optional[Any]:
    """
    Get learning rate scheduler based on configuration.

    Args:
        optimizer: Optimizer to schedule.
        config: Training configuration.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        Scheduler or None.
    """
    if config.lr_scheduler == "cosine":
        # Cosine annealing with warm restarts
        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.lr_warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return config.lr_min / config.learning_rate + (1 - config.lr_min / config.learning_rate) * (
                1 + np.cos(np.pi * progress)
            ) / 2

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.lr_scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )

    elif config.lr_scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.lr_min,
        )

    else:
        return None


# =============================================================================
# Training Functions
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the best available device.

    Args:
        preference: Device preference ("auto", "cuda", "mps", "cpu").

    Returns:
        PyTorch device.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    scheduler: Optional[Any] = None,
    use_auxiliary: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Device to train on.
        config: Training configuration.
        scheduler: Optional LR scheduler (if step-based).
        use_auxiliary: Whether model has auxiliary outputs.

    Returns:
        Dictionary of average metrics for the epoch.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0
    loss_components: Dict[str, float] = {}

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        left_glyph = batch["left_glyph"].to(device)
        right_glyph = batch["right_glyph"].to(device)
        metrics = batch["metrics"].to(device)
        kerning_target = batch["kerning"].to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_auxiliary:
            kerning_pred, needs_kerning_pred, direction_pred = model(left_glyph, right_glyph, metrics)
            loss, components = loss_fn(kerning_pred, needs_kerning_pred, direction_pred, kerning_target)

            # Accumulate component losses
            for key, value in components.items():
                loss_components[key] = loss_components.get(key, 0.0) + value * len(kerning_target)
        else:
            kerning_pred = model(left_glyph, right_glyph, metrics)
            loss = loss_fn(kerning_pred, kerning_target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()

        # Update step-based scheduler
        if scheduler is not None and config.lr_scheduler == "cosine":
            scheduler.step()

        total_loss += loss.item() * len(kerning_target)
        total_samples += len(kerning_target)

        # Logging
        if (batch_idx + 1) % config.log_every == 0:
            avg_loss = total_loss / total_samples
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={avg_loss:.4f}, lr={lr:.6f}")

    metrics_dict = {
        "train_loss": total_loss / total_samples,
    }

    # Add component losses if using auxiliary
    if loss_components:
        for key, value in loss_components.items():
            metrics_dict[f"train_{key}"] = value / total_samples

    return metrics_dict


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: DatasetConfig,
    use_auxiliary: bool = False,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: Model to validate.
        dataloader: Validation data loader.
        loss_fn: Loss function.
        device: Device.
        config: Dataset configuration.
        use_auxiliary: Whether model has auxiliary outputs.

    Returns:
        Dictionary of validation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds: List[float] = []
    all_targets: List[float] = []

    for batch in dataloader:
        left_glyph = batch["left_glyph"].to(device)
        right_glyph = batch["right_glyph"].to(device)
        metrics = batch["metrics"].to(device)
        kerning_target = batch["kerning"].to(device)

        if use_auxiliary:
            kerning_pred, needs_kerning_pred, direction_pred = model(left_glyph, right_glyph, metrics)
            # Use only main kerning loss for validation metric
            if isinstance(loss_fn, AuxiliaryLoss):
                loss = loss_fn.main_loss(kerning_pred, kerning_target)
            else:
                loss, _ = loss_fn(kerning_pred, needs_kerning_pred, direction_pred, kerning_target)
        else:
            kerning_pred = model(left_glyph, right_glyph, metrics)
            loss = loss_fn(kerning_pred, kerning_target)

        total_loss += loss.item() * len(kerning_target)
        total_samples += len(kerning_target)

        # Collect predictions for additional metrics
        all_preds.extend(kerning_pred.cpu().squeeze().tolist())
        all_targets.extend(kerning_target.cpu().squeeze().tolist())

    # Convert to numpy for metric calculation
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Calculate metrics
    val_loss = total_loss / total_samples

    # Mean Absolute Error (in normalized units)
    mae = np.mean(np.abs(preds - targets))

    # Correlation coefficient
    if np.std(preds) > 0 and np.std(targets) > 0:
        correlation = np.corrcoef(preds, targets)[0, 1]
    else:
        correlation = 0.0

    # Denormalize for interpretable metrics
    preds_upm = np.array([config.denormalize_kerning(p) for p in preds])
    targets_upm = np.array([config.denormalize_kerning(t) for t in targets])
    mae_upm = np.mean(np.abs(preds_upm - targets_upm))

    return {
        "val_loss": val_loss,
        "val_mae": mae,
        "val_mae_upm": mae_upm,
        "val_correlation": correlation,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    config: ModelConfig,
    training_config: TrainingConfig,
    path: Path,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "training_config": training_config,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Load training checkpoint.

    Returns:
        Tuple of (epoch, metrics).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def cleanup_checkpoints(checkpoint_dir: Path, keep_last_n: int, best_path: Path) -> None:
    """Remove old checkpoints, keeping the N most recent and the best."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)

    # Keep the best checkpoint and the last N
    to_keep = {best_path}
    for ckpt in checkpoints[-keep_last_n:]:
        to_keep.add(ckpt)

    for ckpt in checkpoints:
        if ckpt not in to_keep and ckpt.exists():
            ckpt.unlink()


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    resume_from: Optional[Path] = None,
    use_auxiliary: bool = False,
) -> nn.Module:
    """
    Main training function.

    Args:
        model_config: Model architecture configuration.
        training_config: Training configuration.
        dataset_config: Dataset configuration.
        train_dataset: Training dataset. If None, uses dummy dataset.
        val_dataset: Validation dataset. If None, splits from train.
        resume_from: Path to checkpoint to resume from.
        use_auxiliary: Whether to use auxiliary outputs.

    Returns:
        Trained model.
    """
    # Use defaults if not provided
    model_config = model_config or DEFAULT_MODEL_CONFIG
    training_config = training_config or DEFAULT_TRAINING_CONFIG
    dataset_config = dataset_config or DEFAULT_DATASET_CONFIG

    # Validate configuration
    validate_config(model_config, training_config, dataset_config)

    # Set seed for reproducibility
    set_seed(training_config.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create model
    model = create_model(model_config, use_auxiliary)
    model.to(device)
    print(f"\n{model.summary()}\n")

    # Create or use provided datasets
    if train_dataset is None:
        print("WARNING: Using dummy dataset for training. Replace with actual dataset.")
        full_dataset = DummyKerningDataset(
            size=10000,
            image_size=model_config.image_size,
            config=dataset_config,
        )

        # Split into train/val
        val_size = int(len(full_dataset) * training_config.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(training_config.seed),
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # Create scheduler
    scheduler = get_scheduler(optimizer, training_config, len(train_loader))

    # Create loss function
    if use_auxiliary:
        loss_fn = AuxiliaryLoss(training_config)
    else:
        loss_fn = get_loss_function(training_config)

    # Setup logging
    writer = None
    if training_config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = training_config.checkpoint_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logs: {log_dir}")
        except ImportError:
            print("TensorBoard not available. Skipping.")

    if training_config.use_wandb:
        try:
            import wandb
            wandb.init(
                project="kerning-net",
                name=training_config.experiment_name,
                config={
                    "model": model_config.__dict__,
                    "training": training_config.__dict__,
                    "dataset": dataset_config.__dict__,
                },
            )
        except ImportError:
            print("Weights & Biases not available. Skipping.")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from is not None and resume_from.exists():
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch, metrics = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        best_val_loss = metrics.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.4f}")

    # Training loop
    checkpoint_dir = training_config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"

    epochs_without_improvement = 0

    print(f"\nStarting training for {training_config.epochs} epochs...")
    print("=" * 60)

    for epoch in range(start_epoch, training_config.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{training_config.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, training_config, scheduler, use_auxiliary
        )

        # Validate
        if (epoch + 1) % training_config.val_every == 0:
            if use_auxiliary:
                val_metrics = validate(model, val_loader, loss_fn, device, dataset_config, use_auxiliary)
            else:
                val_metrics = validate(model, val_loader, loss_fn, device, dataset_config, use_auxiliary)

            all_metrics = {**train_metrics, **val_metrics}

            # Update plateau scheduler
            if scheduler is not None and training_config.lr_scheduler == "plateau":
                scheduler.step(val_metrics["val_loss"])

            # Check for improvement
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                epochs_without_improvement = 0

                # Save best model
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, all_metrics,
                    model_config, training_config, best_model_path
                )
                print(f"  New best model saved! val_loss={best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
        else:
            all_metrics = train_metrics

        # Update step scheduler (per epoch)
        if scheduler is not None and training_config.lr_scheduler == "step":
            scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Time: {epoch_time:.1f}s, LR: {lr:.6f}")
        for key, value in all_metrics.items():
            print(f"  {key}: {value:.4f}")

        if writer is not None:
            for key, value in all_metrics.items():
                writer.add_scalar(key, value, epoch + 1)
            writer.add_scalar("learning_rate", lr, epoch + 1)

        if training_config.use_wandb:
            try:
                import wandb
                wandb.log({**all_metrics, "learning_rate": lr, "epoch": epoch + 1})
            except Exception:
                pass

        # Save checkpoint
        if (epoch + 1) % training_config.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, all_metrics,
                model_config, training_config, ckpt_path
            )
            cleanup_checkpoints(checkpoint_dir, training_config.keep_last_n, best_model_path)

        # Early stopping
        if training_config.early_stopping and epochs_without_improvement >= training_config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement.")
            break

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

    if writer is not None:
        writer.close()

    if training_config.use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    # Load best model for return
    model.load_state_dict(torch.load(best_model_path, weights_only=False)["model_state_dict"])

    return model


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train KerningNet model for kerning prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--image-size", type=int, default=64,
        help="Input image size"
    )
    model_group.add_argument(
        "--embedding-dim", type=int, default=256,
        help="Embedding dimension from encoder"
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate in regression head"
    )
    model_group.add_argument(
        "--auxiliary", action="store_true",
        help="Use auxiliary outputs for multi-task learning"
    )

    # Training configuration
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=128,
        help="Training batch size"
    )
    train_group.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=1e-5,
        help="Weight decay for AdamW"
    )
    train_group.add_argument(
        "--loss", type=str, choices=["mse", "huber", "smooth_l1"], default="huber",
        help="Loss function"
    )
    train_group.add_argument(
        "--scheduler", type=str, choices=["cosine", "step", "plateau", "none"], default="cosine",
        help="Learning rate scheduler"
    )
    train_group.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Gradient clipping max norm (0 to disable)"
    )

    # Data configuration
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-dir", type=Path, default=Path("data/kerning"),
        help="Path to kerning dataset"
    )
    data_group.add_argument(
        "--val-split", type=float, default=0.1,
        help="Validation split ratio"
    )
    data_group.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers"
    )

    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/kerning_net"),
        help="Directory for saving checkpoints"
    )
    ckpt_group.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume from"
    )

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--wandb", action="store_true",
        help="Use Weights & Biases for logging"
    )
    log_group.add_argument(
        "--tensorboard", action="store_true", default=True,
        help="Use TensorBoard for logging"
    )
    log_group.add_argument(
        "--experiment-name", type=str, default="kerning_net",
        help="Experiment name for logging"
    )

    # Misc
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    misc_group.add_argument(
        "--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help="Device to train on"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for training script."""
    args = parse_args()

    # Build configurations from arguments
    model_config = ModelConfig(
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_fn=args.loss,
        lr_scheduler=args.scheduler if args.scheduler != "none" else "step",
        max_grad_norm=args.grad_clip,
        val_split=args.val_split,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )

    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
    )

    # Train the model
    print("=" * 60)
    print("KerningNet Training")
    print("=" * 60)

    train(
        model_config=model_config,
        training_config=training_config,
        dataset_config=dataset_config,
        resume_from=args.resume,
        use_auxiliary=args.auxiliary,
    )


if __name__ == "__main__":
    main()
