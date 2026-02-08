"""
Training script for GlyphDiffusion model.

This module implements the training loop for flow-matching diffusion,
including:
- Data loading from glyph_dataset
- Mixed precision training
- Gradient clipping
- EMA model averaging
- Checkpointing
- Wandb/TensorBoard logging

Usage:
    python train.py --config default
    python train.py --batch-size 32 --lr 1e-4 --epochs 100
    python train.py --resume checkpoints/latest.pt
"""

import argparse
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import (
    Config,
    ModelConfig,
    TrainingConfig,
    FlowMatchingConfig,
    DataConfig,
)
from model import GlyphDiffusionModel, create_model
from noise_schedule import (
    FlowMatchingSchedule,
    FlowMatchingLoss,
    prepare_training_batch,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model weights.

    EMA helps stabilize training and often produces better samples
    than the raw model weights.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        """Initialize EMA model.

        Args:
            model: Model to track.
            decay: EMA decay factor.
            device: Device to store EMA weights on.
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Create shadow copy of parameters
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights.

        Args:
            model: Model with updated weights.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.to(self.device), alpha=1 - self.decay
                )

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights to model.

        Args:
            model: Model to copy weights to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        """Get EMA state dict for saving."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state dict."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


class DummyGlyphDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing when real data isn't available.

    This creates synthetic glyph-like images for development/testing.
    Replace with actual glyph_dataset when Workstream A is complete.
    """

    def __init__(
        self,
        size: int = 10000,
        image_size: int = 64,
        num_characters: int = 62,
        style_dim: int = 128,
    ):
        """Initialize dummy dataset.

        Args:
            size: Number of samples.
            image_size: Image size in pixels.
            num_characters: Number of character classes.
            style_dim: Dimension of style embeddings.
        """
        self.size = size
        self.image_size = image_size
        self.num_characters = num_characters
        self.style_dim = style_dim

        # Generate random style embeddings (simulate different fonts)
        self.num_styles = 100
        self.styles = torch.randn(self.num_styles, style_dim)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample.

        Returns:
            Dictionary with:
                - image: Grayscale glyph image (1, H, W)
                - char_index: Character index
                - style_embed: Style embedding
        """
        # Random character
        char_index = torch.randint(0, self.num_characters, (1,)).item()

        # Random style
        style_idx = idx % self.num_styles
        style_embed = self.styles[style_idx]

        # Generate a simple synthetic glyph (circle/rectangle based on char)
        image = self._generate_synthetic_glyph(char_index)

        return {
            "image": image,
            "char_index": char_index,
            "style_embed": style_embed,
        }

    def _generate_synthetic_glyph(self, char_index: int) -> torch.Tensor:
        """Generate a simple synthetic glyph image.

        This creates basic shapes that vary by character index.
        Real glyphs will come from the dataset.
        """
        image = torch.zeros(1, self.image_size, self.image_size)

        # Use character index to vary the shape
        center = self.image_size // 2
        radius = self.image_size // 4

        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(self.image_size),
            torch.arange(self.image_size),
            indexing="ij",
        )

        # Different shapes for different character groups
        if char_index < 26:  # lowercase a-z: circles
            dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
            image[0] = (dist < radius).float()
        elif char_index < 52:  # uppercase A-Z: rectangles
            hw = radius * 0.8
            image[0] = (
                (torch.abs(x - center) < hw) & (torch.abs(y - center) < hw)
            ).float()
        else:  # digits 0-9: diamonds
            dist = torch.abs(x - center) + torch.abs(y - center)
            image[0] = (dist < radius).float()

        # Add some noise
        image = image + 0.1 * torch.randn_like(image)
        image = torch.clamp(image, 0, 1)

        return image


def create_dataloaders(
    config: Config,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders.

    Args:
        config: Full configuration.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Try to import the real dataset
    try:
        # This will be available when Workstream A is complete
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "datasets"))
        from glyph_dataset import GlyphDataset

        logger.info("Using real GlyphDataset")
        train_dataset = GlyphDataset(
            data_dir=config.data.data_dir,
            split="train",
            image_size=config.model.image_size,
            augment=config.data.augment,
        )
        val_dataset = GlyphDataset(
            data_dir=config.data.data_dir,
            split="val",
            image_size=config.model.image_size,
            augment=False,
        )
    except ImportError:
        logger.warning("GlyphDataset not found, using DummyGlyphDataset")
        train_dataset = DummyGlyphDataset(
            size=10000,
            image_size=config.model.image_size,
            num_characters=config.model.num_characters,
            style_dim=config.model.style_embed_dim,
        )
        val_dataset = DummyGlyphDataset(
            size=1000,
            image_size=config.model.image_size,
            num_characters=config.model.num_characters,
            style_dim=config.model.style_embed_dim,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    num_training_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize.
        config: Training configuration.
        num_training_steps: Total number of training steps.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.lr_warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - config.lr_warmup_steps,
        eta_min=config.learning_rate * 0.01,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.lr_warmup_steps],
    )

    return optimizer, scheduler


def save_checkpoint(
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    step: int,
    loss: float,
    config: Config,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save.
        ema: EMA model.
        optimizer: Optimizer state.
        scheduler: Scheduler state.
        scaler: Gradient scaler for mixed precision.
        epoch: Current epoch.
        step: Current global step.
        loss: Current loss value.
        config: Training configuration.
        checkpoint_dir: Directory to save checkpoints.
        is_best: Whether this is the best model so far.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "config": {
            "model": config.model.__dict__,
            "training": config.training.__dict__,
            "flow_matching": config.flow_matching.__dict__,
        },
    }

    # Save latest
    torch.save(checkpoint, checkpoint_dir / "latest.pt")

    # Save epoch checkpoint
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt")

    # Save best if applicable
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best.pt")

    logger.info(f"Saved checkpoint at epoch {epoch}, step {step}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        ema: EMA model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        scaler: Gradient scaler to load state into.
        device: Device to load tensors to.

    Returns:
        Tuple of (epoch, step, loss).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle both full training checkpoint and bare state dict formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Bare state dict - load directly into model
        model.load_state_dict(checkpoint)
        logger.warning("Loaded bare state dict; optimizer/scheduler/EMA state not available")
        return 0, 0, float("inf")

    ema.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    loss_fn: FlowMatchingLoss,
    schedule: FlowMatchingSchedule,
    ema: EMAModel,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    device: torch.device,
    writer=None,
) -> tuple[float, int]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        scaler: Gradient scaler for mixed precision.
        loss_fn: Loss function.
        schedule: Flow matching schedule.
        ema: EMA model.
        config: Training configuration.
        epoch: Current epoch number.
        global_step: Current global step.
        device: Device to train on.
        writer: TensorBoard or wandb writer.

    Returns:
        Tuple of (average loss, updated global step).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    use_amp = config.mixed_precision and device.type == "cuda"
    amp_context = autocast(device_type=device.type) if use_amp else nullcontext()

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch["image"].to(device)
        char_indices = torch.tensor(batch["char_index"]).to(device)
        style_embed = batch["style_embed"].to(device)

        # Prepare training batch (add noise, compute targets)
        noisy_images, timesteps, noise, target_velocity = prepare_training_batch(
            images, schedule, device
        )

        # Forward pass with mixed precision
        with amp_context:
            predicted_velocity = model(
                noisy_images,
                timesteps,
                char_indices,
                style_embed,
                mask=None,  # TODO: Add mask conditioning
            )
            loss = loss_fn(predicted_velocity, target_velocity)

        # Backward pass
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()

        scheduler.step()

        # Update EMA
        ema.update(model)

        # Logging
        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if global_step % config.log_every_n_steps == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch} | Step {global_step} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e}"
            )

            # Log to tensorboard/wandb
            if writer is not None:
                log_metrics(
                    writer,
                    {
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/lr": lr,
                    },
                    global_step,
                )

    return total_loss / num_batches, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: FlowMatchingLoss,
    schedule: FlowMatchingSchedule,
    device: torch.device,
) -> float:
    """Run validation.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        loss_fn: Loss function.
        schedule: Flow matching schedule.
        device: Device to run on.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        char_indices = torch.tensor(batch["char_index"]).to(device)
        style_embed = batch["style_embed"].to(device)

        noisy_images, timesteps, noise, target_velocity = prepare_training_batch(
            images, schedule, device
        )

        predicted_velocity = model(
            noisy_images,
            timesteps,
            char_indices,
            style_embed,
        )
        loss = loss_fn(predicted_velocity, target_velocity)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def log_metrics(writer, metrics: dict, step: int) -> None:
    """Log metrics to tensorboard or wandb.

    Args:
        writer: TensorBoard SummaryWriter or wandb.
        metrics: Dictionary of metric names to values.
        step: Global step.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
            return
    except ImportError:
        pass

    # Fall back to tensorboard
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(name, value, step)


def setup_logging(config: TrainingConfig, output_dir: Path):
    """Set up logging backends.

    Args:
        config: Training configuration.
        output_dir: Output directory for logs.

    Returns:
        Writer object (tensorboard or wandb).
    """
    writer = None

    if config.use_wandb:
        try:
            import wandb

            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config={
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "epochs": config.num_epochs,
                },
            )
            logger.info("Initialized wandb logging")
        except ImportError:
            logger.warning("wandb not installed, falling back to tensorboard")
            config.use_wandb = False
            config.use_tensorboard = True

    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = output_dir / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir)
            logger.info(f"Initialized tensorboard logging at {log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed")

    return writer


def train(config: Config, resume_path: Optional[Path] = None) -> None:
    """Main training function.

    Args:
        config: Full configuration.
        resume_path: Optional path to checkpoint to resume from.
    """
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Set random seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    writer = setup_logging(config.training, config.output_dir)

    # Create model
    logger.info("Creating model...")
    model = create_model(config.model).to(device)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Create EMA model
    ema = EMAModel(model, decay=config.training.ema_decay, device=device)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Calculate total training steps
    num_training_steps = len(train_loader) * config.training.num_epochs

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config.training, num_training_steps
    )

    # Create gradient scaler for mixed precision
    scaler = GradScaler(device.type, enabled=config.training.mixed_precision)

    # Create loss function and schedule
    loss_fn = FlowMatchingLoss()
    schedule = FlowMatchingSchedule(config.flow_matching)

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_path is not None:
        start_epoch, global_step, _ = load_checkpoint(
            resume_path, model, ema, optimizer, scheduler, scaler, device
        )
        start_epoch += 1  # Start from next epoch
        logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_fn=loss_fn,
            schedule=schedule,
            ema=ema,
            config=config.training,
            epoch=epoch,
            global_step=global_step,
            device=device,
            writer=writer,
        )

        epoch_time = time.time() - epoch_start

        # Validate
        val_loss = None
        if (
            val_loader is not None
            and (epoch + 1) % config.training.validate_every_n_epochs == 0
        ):
            val_loss = validate(model, val_loader, loss_fn, schedule, device)
            logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s"
            )

            if writer is not None:
                log_metrics(
                    writer,
                    {
                        "val/loss": val_loss,
                        "epoch": epoch,
                    },
                    global_step,
                )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            is_best = False

        # Save checkpoint
        if (epoch + 1) % config.training.checkpoint_every_n_epochs == 0:
            save_checkpoint(
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                loss=train_loss,
                config=config,
                checkpoint_dir=config.checkpoint_dir,
                is_best=is_best,
            )

    # Save final model
    save_checkpoint(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=config.training.num_epochs - 1,
        step=global_step,
        loss=train_loss,
        config=config,
        checkpoint_dir=config.checkpoint_dir,
        is_best=False,
    )

    logger.info("Training complete!")

    # Clean up
    if writer is not None:
        writer.close()
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GlyphDiffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config presets
    parser.add_argument(
        "--config",
        type=str,
        choices=["default", "high_res", "fast_dev"],
        default="default",
        help="Configuration preset",
    )

    # Model config
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[64, 128],
        default=64,
        help="Image resolution",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base number of UNet channels",
    )

    # Training config
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay factor",
    )

    # Flow matching config
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )

    # Data config
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/glyphs"),
        help="Path to glyph dataset",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        help="Use TensorBoard logging",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="typogenesis-glyph-diffusion",
        help="Project name for logging",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logging",
    )

    # Checkpointing
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create config from preset
    if args.config == "high_res":
        config = Config.high_resolution()
    elif args.config == "fast_dev":
        config = Config.fast_dev()
    else:
        config = Config.default()

    # Override with command line arguments
    from config import Resolution

    if args.resolution == 128:
        config.model.resolution = Resolution.HIGH
    config.model.base_channels = args.base_channels

    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.num_epochs = args.epochs
    config.training.mixed_precision = not args.no_mixed_precision
    config.training.gradient_clip_norm = args.gradient_clip
    config.training.ema_decay = args.ema_decay
    config.training.num_workers = args.num_workers
    config.training.use_wandb = args.use_wandb
    config.training.use_tensorboard = args.use_tensorboard
    config.training.project_name = args.project_name
    config.training.run_name = args.run_name
    config.training.seed = args.seed

    config.flow_matching.num_inference_steps = args.inference_steps

    config.data.data_dir = args.data_dir

    config.output_dir = args.output_dir
    config.checkpoint_dir = args.checkpoint_dir

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Resolution: {config.model.image_size}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Epochs: {config.training.num_epochs}")
    logger.info(f"  Mixed precision: {config.training.mixed_precision}")
    logger.info(f"  Data directory: {config.data.data_dir}")

    # Start training
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
