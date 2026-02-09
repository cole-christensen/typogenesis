#!/usr/bin/env python3
"""End-to-end training pipeline for all Typogenesis ML models.

Trains StyleEncoder, KerningNet, and GlyphDiffusion on extracted font data,
evaluates them, converts to CoreML, and generates sample outputs.

Usage:
    python train_all.py --data-dir data/extracted --output-dir outputs
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── StyleEncoder Training ───────────────────────────────────────────


def train_style_encoder(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    glyphs_per_font: int = 4,
) -> Path:
    """Train StyleEncoder with contrastive learning on real font data."""
    from datasets.style_dataset import StyleDataset, style_collate_fn
    from models.style_encoder.losses import NTXentLoss
    from models.style_encoder.model import StyleEncoder, StyleEncoderConfig

    logger.info("=" * 60)
    logger.info("TRAINING STYLE ENCODER")
    logger.info("=" * 60)

    # Create datasets
    train_ds = StyleDataset(
        data_dir=data_dir,
        split="train",
        image_size=64,
        glyphs_per_font=glyphs_per_font,
        min_glyphs=10,
    )
    val_ds = StyleDataset(
        data_dir=data_dir,
        split="val",
        image_size=64,
        glyphs_per_font=glyphs_per_font,
        min_glyphs=10,
    )

    if len(train_ds) == 0:
        logger.error("No training data for StyleEncoder! Check style_train.jsonl")
        raise RuntimeError("Empty training dataset")

    logger.info(f"Train: {len(train_ds)} fonts, Val: {len(val_ds)} fonts")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=style_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, len(val_ds)), shuffle=False,
        collate_fn=style_collate_fn,
    ) if len(val_ds) > 0 else None

    # Create model
    config = StyleEncoderConfig(pretrained=False)
    model = StyleEncoder(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"StyleEncoder: {param_count:,} parameters on {device}")

    # Loss and optimizer - NTXent with proper positive pair construction
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    checkpoint_path = output_dir / "style_encoder.pt"
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for glyphs, labels in train_loader:
            glyphs = glyphs.to(device)
            labels = labels.to(device)

            # Get projections for contrastive loss
            _, projections = model(glyphs, return_projection=True)

            # Reshape to (batch_size, glyphs_per_font, proj_dim)
            # collate_fn stacks font-by-font: [f0g0,f0g1,..,f1g0,f1g1,..]
            n_total = projections.shape[0]
            n_fonts = n_total // glyphs_per_font
            if n_fonts < 2:
                continue
            proj_reshaped = projections[:n_fonts * glyphs_per_font].view(
                n_fonts, glyphs_per_font, -1
            )
            # z_i = first glyph per font, z_j = second glyph per font
            z_i = proj_reshaped[:, 0]  # (n_fonts, proj_dim)
            z_j = proj_reshaped[:, 1]  # (n_fonts, proj_dim)

            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validate
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for glyphs, _labels in val_loader:
                    glyphs = glyphs.to(device)
                    _, projections = model(glyphs, return_projection=True)
                    n_total = projections.shape[0]
                    n_fonts = n_total // glyphs_per_font
                    if n_fonts < 2:
                        continue
                    proj_r = projections[:n_fonts * glyphs_per_font].view(
                        n_fonts, glyphs_per_font, -1
                    )
                    val_loss = criterion(proj_r[:, 0], proj_r[:, 1]).item()
            history["val_loss"].append(val_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "config": {"embedding_dim": 128, "projection_dim": 256},
            }, checkpoint_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}: train_loss={avg_loss:.4f}, "
                f"val_loss={val_loss:.4f}, best={best_loss:.4f}"
            )

    logger.info(f"StyleEncoder training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Save training history
    (output_dir / "style_encoder_history.json").write_text(json.dumps(history))

    return checkpoint_path


# ─── Style Embedding Pre-computation ─────────────────────────────────


def precompute_style_embeddings(
    checkpoint_path: Path,
    data_dir: Path,
    device: torch.device,
) -> Path:
    """Pre-compute style embeddings for all fonts using trained StyleEncoder."""
    from data.augmentations import get_eval_transforms
    from models.style_encoder.model import StyleEncoder, StyleEncoderConfig

    logger.info("Pre-computing style embeddings...")

    config = StyleEncoderConfig(pretrained=False)
    model = StyleEncoder(config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    transform = get_eval_transforms(image_size=64)
    embeddings_dir = data_dir / "style_embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # For each font directory, compute mean embedding from all glyphs
    font_dirs = [d for d in data_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]

    for font_dir in font_dirs:
        glyph_dir = font_dir / "glyphs"
        if not glyph_dir.exists():
            continue

        glyph_images = list(glyph_dir.glob("*_64.png"))
        if not glyph_images:
            continue

        # Load and encode all glyphs
        tensors = []
        for img_path in glyph_images:
            img = Image.open(img_path).convert("L")
            tensors.append(transform(img))

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            embeddings = model(batch)  # (N, 128)

        # Save mean embedding for this font
        mean_embed = embeddings.mean(dim=0).cpu().numpy()
        np.save(embeddings_dir / f"{font_dir.name}.npy", mean_embed)

    n_fonts = len(list(embeddings_dir.glob("*.npy")))
    logger.info(f"Saved {n_fonts} style embeddings to {embeddings_dir}")
    return embeddings_dir


# ─── KerningNet Training ─────────────────────────────────────────────


def train_kerning_net(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Path:
    """Train KerningNet on real kerning pair data."""
    from datasets.kerning_dataset import KerningDataset
    from models.kerning_net.model import KerningNet

    logger.info("=" * 60)
    logger.info("TRAINING KERNING NET")
    logger.info("=" * 60)

    train_ds = KerningDataset(data_dir=data_dir, split="train", image_size=64)
    val_ds = KerningDataset(data_dir=data_dir, split="val", image_size=64)

    if len(train_ds) == 0:
        logger.error("No training data for KerningNet! Check kerning_train.jsonl")
        raise RuntimeError("Empty training dataset")

    logger.info(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = KerningNet().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"KerningNet: {param_count:,} parameters on {device}")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    checkpoint_path = output_dir / "kerning_net.pt"
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for left, right, target in train_loader:
            left, right, target = left.to(device), right.to(device), target.to(device)
            pred = model(left, right).squeeze(-1)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        with torch.no_grad():
            for left, right, target in val_loader:
                left, right, target = left.to(device), right.to(device), target.to(device)
                pred = model(left, right).squeeze(-1)
                val_loss += criterion(pred, target).item()
                val_mae += (pred - target).abs().mean().item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        val_mae /= max(val_batches, 1)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae * 200.0)  # Denormalize to font units

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, checkpoint_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}: train_loss={avg_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_mae={val_mae * 200:.1f} units"
            )

    logger.info(f"KerningNet training complete. Best val loss: {best_loss:.4f}")
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    (output_dir / "kerning_net_history.json").write_text(json.dumps(history))
    return checkpoint_path


# ─── GlyphDiffusion Training ─────────────────────────────────────────


def train_glyph_diffusion(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
) -> Path:
    """Train GlyphDiffusion model with flow matching on real glyph data."""
    from datasets.glyph_dataset import GlyphDataset
    from models.glyph_diffusion import (
        FlowMatchingLoss,
        FlowMatchingSchedule,
        GlyphDiffusionModel,
        ModelConfig,
        prepare_training_batch,
    )

    logger.info("=" * 60)
    logger.info("TRAINING GLYPH DIFFUSION")
    logger.info("=" * 60)

    train_ds = GlyphDataset(
        data_dir=data_dir, split="train", image_size=64,
        augment=True, style_embed_dir=data_dir / "style_embeddings",
    )
    val_ds = GlyphDataset(
        data_dir=data_dir, split="val", image_size=64,
        augment=False, style_embed_dir=data_dir / "style_embeddings",
    )

    if len(train_ds) == 0:
        logger.error("No training data for GlyphDiffusion!")
        raise RuntimeError("Empty training dataset")

    logger.info(f"Train: {len(train_ds)} glyphs, Val: {len(val_ds)} glyphs")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Create model
    model_config = ModelConfig()
    model = GlyphDiffusionModel(model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GlyphDiffusion: {param_count:,} parameters on {device}")

    # Training setup
    schedule = FlowMatchingSchedule()
    loss_fn = FlowMatchingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    checkpoint_path = output_dir / "glyph_diffusion.pt"
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            char_indices = batch["char_index"].to(device)
            style_embed = batch["style_embed"].to(device)

            # Prepare flow matching batch
            x_t, timesteps, _noise, target_velocity = prepare_training_batch(
                images, schedule, device
            )

            # Forward pass
            predicted_velocity = model(x_t, timesteps, char_indices, style_embed)
            loss = loss_fn(predicted_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                char_indices = batch["char_index"].to(device)
                style_embed = batch["style_embed"].to(device)

                x_t, timesteps, _noise, target_velocity = prepare_training_batch(
                    images, schedule, device
                )
                predicted_velocity = model(x_t, timesteps, char_indices, style_embed)
                val_loss += loss_fn(predicted_velocity, target_velocity).item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "config": {
                    "model": {
                        "resolution": 64,
                        "base_channels": model_config.base_channels,
                        "num_characters": model_config.num_characters,
                        "style_embed_dim": model_config.style_embed_dim,
                    },
                },
            }, checkpoint_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}: train_loss={avg_loss:.4f}, "
                f"val_loss={val_loss:.4f}, best={best_loss:.4f}"
            )

    logger.info(f"GlyphDiffusion training complete. Best val loss: {best_loss:.4f}")
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    (output_dir / "glyph_diffusion_history.json").write_text(json.dumps(history))
    return checkpoint_path


# ─── Evaluation ───────────────────────────────────────────────────────


def evaluate_all(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run evaluation metrics on all trained models."""
    from datasets.kerning_dataset import KerningDataset
    from datasets.style_dataset import StyleDataset, style_collate_fn
    from evaluation.metrics import (
        cross_font_similarity,
        kerning_direction_accuracy,
        kerning_mae,
        retrieval_accuracy,
        same_font_similarity,
    )
    from models.kerning_net.model import KerningNet
    from models.style_encoder.model import StyleEncoder, StyleEncoderConfig

    logger.info("=" * 60)
    logger.info("EVALUATING ALL MODELS")
    logger.info("=" * 60)

    results = {}

    # ── StyleEncoder evaluation ──
    se_ckpt = output_dir / "style_encoder.pt"
    if se_ckpt.exists():
        logger.info("Evaluating StyleEncoder...")
        config = StyleEncoderConfig(pretrained=False)
        model = StyleEncoder(config).to(device)
        state = torch.load(se_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        # Load val data
        val_ds = StyleDataset(data_dir=data_dir, split="val", image_size=64,
                              glyphs_per_font=8, min_glyphs=10)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=len(val_ds),
                                    collate_fn=style_collate_fn)
            glyphs, labels = next(iter(val_loader))
            glyphs = glyphs.to(device)

            with torch.no_grad():
                embeddings = model(glyphs).cpu()

            same_sim = same_font_similarity(embeddings, labels)
            cross_sim = cross_font_similarity(embeddings, labels)
            retrieval = retrieval_accuracy(embeddings, labels, top_k=(1, 5))

            results["style_encoder"] = {
                "same_font_similarity": round(same_sim, 4),
                "cross_font_similarity": round(cross_sim, 4),
                **{f"retrieval_{k}": round(v, 4) for k, v in retrieval.items()},
                "num_embeddings": len(embeddings),
                "num_fonts": len(labels.unique()),
            }
            logger.info(f"  Same-font sim: {same_sim:.4f}, Cross-font sim: {cross_sim:.4f}")
            logger.info(f"  Retrieval: {retrieval}")

    # ── KerningNet evaluation ──
    kn_ckpt = output_dir / "kerning_net.pt"
    if kn_ckpt.exists():
        logger.info("Evaluating KerningNet...")
        model = KerningNet().to(device)
        state = torch.load(kn_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        val_ds = KerningDataset(data_dir=data_dir, split="val", image_size=64)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=128)
            all_preds, all_targets = [], []
            with torch.no_grad():
                for left, right, target in val_loader:
                    left, right = left.to(device), right.to(device)
                    pred = model(left, right).cpu().squeeze(-1)
                    all_preds.append(pred)
                    all_targets.append(target)

            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            mae = kerning_mae(preds, targets, max_kerning=200.0)
            dir_acc = kerning_direction_accuracy(preds, targets)

            results["kerning_net"] = {
                "mae_font_units": round(mae, 2),
                "direction_accuracy": round(dir_acc, 4),
                "num_pairs": len(preds),
            }
            logger.info(f"  MAE: {mae:.2f} font units, Direction acc: {dir_acc:.4f}")

    return results


# ─── Sample Generation ────────────────────────────────────────────────


def generate_samples(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Generate sample glyph images using trained GlyphDiffusion model."""
    from models.glyph_diffusion import (
        FlowMatchingScheduler,
        GlyphDiffusionModel,
        ModelConfig,
        sample_euler,
    )
    from models.glyph_diffusion.config import CHAR_TO_IDX

    logger.info("=" * 60)
    logger.info("GENERATING SAMPLES")
    logger.info("=" * 60)

    ckpt_path = output_dir / "glyph_diffusion.pt"
    if not ckpt_path.exists():
        logger.warning("No GlyphDiffusion checkpoint found, skipping sample generation")
        return

    # Load model
    model_config = ModelConfig()
    model = GlyphDiffusionModel(model_config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Load a style embedding
    embed_dir = data_dir / "style_embeddings"
    embed_files = list(embed_dir.glob("*.npy")) if embed_dir.exists() else []
    if not embed_files:
        logger.warning("No style embeddings found, using random")
        torch.manual_seed(42)
        style_embed = torch.randn(1, 128)
    else:
        style_embed = torch.tensor(np.load(embed_files[0]), dtype=torch.float32).unsqueeze(0)

    # Generate A-Z + a-z + 0-9
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    char_indices = torch.tensor([CHAR_TO_IDX[c] for c in chars if c in CHAR_TO_IDX], device=device)
    n = len(char_indices)
    style = style_embed.expand(n, -1).to(device)

    scheduler = FlowMatchingScheduler()

    # Generate with fixed seed for reproducibility
    torch.manual_seed(42)
    noise = torch.randn(n, 1, 64, 64, device=device)

    logger.info(f"Generating {n} glyphs...")
    t_start = time.time()
    with torch.no_grad():
        samples, _ = sample_euler(model, noise, char_indices, style, scheduler, num_steps=50)
    elapsed = time.time() - t_start
    logger.info(f"Generated {n} glyphs in {elapsed:.1f}s ({elapsed / n:.2f}s/glyph)")

    # Save individual samples
    samples_dir = output_dir / "generated_samples"
    samples_dir.mkdir(exist_ok=True)

    for i, c in enumerate(chars[:n]):
        arr = ((samples[i, 0].cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
        img = Image.fromarray(arr, mode="L")
        img.save(samples_dir / f"{c}.png")

    # Create a grid image
    cols = 10
    rows = (n + cols - 1) // cols
    cell = 64
    pad = 2
    grid_w = cols * (cell + pad) + pad
    grid_h = rows * (cell + pad) + pad + 20

    grid = Image.new("L", (grid_w, grid_h), 32)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(grid)
    draw.text((grid_w // 2 - 60, 2), "Generated Glyphs", fill=200)

    for i in range(n):
        row, col = divmod(i, cols)
        x = col * (cell + pad) + pad
        y = row * (cell + pad) + pad + 18
        arr = ((samples[i, 0].cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
        glyph = Image.fromarray(arr, mode="L")
        grid.paste(glyph, (x, y))

    grid_path = output_dir / "generated_grid.png"
    grid.save(grid_path)
    logger.info(f"Saved glyph grid to {grid_path}")

    # Also generate with multiple styles
    if len(embed_files) >= 3:
        logger.info("Generating multi-style comparison...")
        test_chars = "ABCabc123"
        test_indices = torch.tensor(
            [CHAR_TO_IDX[c] for c in test_chars], device=device
        )
        n_test = len(test_indices)

        style_grid = Image.new("L", (n_test * (cell + pad) + 80, 3 * (cell + pad) + 40), 32)
        style_draw = ImageDraw.Draw(style_grid)
        style_draw.text((10, 2), "Multi-Style Generation", fill=200)

        # Column headers
        for j, c in enumerate(test_chars):
            style_draw.text((80 + j * (cell + pad) + cell // 2 - 3, 18), c, fill=180)

        for s_idx, embed_file in enumerate(embed_files[:3]):
            se = torch.tensor(np.load(embed_file), dtype=torch.float32).unsqueeze(0)
            se = se.expand(n_test, -1).to(device)

            torch.manual_seed(42)
            noise = torch.randn(n_test, 1, 64, 64, device=device)

            with torch.no_grad():
                out, _ = sample_euler(model, noise, test_indices, se, scheduler, num_steps=50)

            # Style label
            font_name = embed_file.stem[:10]
            style_draw.text((2, 32 + s_idx * (cell + pad) + cell // 2 - 6), font_name, fill=160)

            for j in range(n_test):
                x = 80 + j * (cell + pad)
                y = 32 + s_idx * (cell + pad)
                arr = ((out[j, 0].cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
                glyph = Image.fromarray(arr, mode="L")
                style_grid.paste(glyph, (x, y))

        style_path = output_dir / "multi_style_grid.png"
        style_grid.save(style_path)
        logger.info(f"Saved multi-style grid to {style_path}")


# ─── CoreML Conversion ───────────────────────────────────────────────


def convert_to_coreml(output_dir: Path) -> None:
    """Convert trained PyTorch models to CoreML format."""
    logger.info("=" * 60)
    logger.info("CONVERTING TO COREML")
    logger.info("=" * 60)

    try:
        import coremltools as ct
    except ImportError:
        logger.warning("coremltools not installed, skipping CoreML conversion")
        return

    coreml_dir = output_dir / "coreml"
    coreml_dir.mkdir(exist_ok=True)

    # ── StyleEncoder ──
    se_ckpt = output_dir / "style_encoder.pt"
    if se_ckpt.exists():
        logger.info("Converting StyleEncoder to CoreML...")
        try:
            from models.style_encoder.model import StyleEncoder, StyleEncoderConfig

            config = StyleEncoderConfig(pretrained=False)
            model = StyleEncoder(config)
            state = torch.load(se_ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            model.eval()

            dummy = torch.randn(1, 1, 64, 64)
            traced = torch.jit.trace(model, dummy)

            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="image", shape=(1, 1, 64, 64))],
                outputs=[ct.TensorType(name="embedding")],
                minimum_deployment_target=ct.target.macOS14,
            )
            mlmodel.save(str(coreml_dir / "StyleEncoder.mlpackage"))
            logger.info("  Saved StyleEncoder.mlpackage")
        except (RuntimeError, Exception) as e:
            logger.warning(f"  StyleEncoder conversion failed: {e}")
            logger.warning("  (coremltools may require Python 3.11-3.12)")

    # ── KerningNet ──
    kn_ckpt = output_dir / "kerning_net.pt"
    if kn_ckpt.exists():
        logger.info("Converting KerningNet to CoreML...")
        try:
            from models.kerning_net.model import KerningNet

            model = KerningNet()
            state = torch.load(kn_ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            model.eval()

            dummy_left = torch.randn(1, 1, 64, 64)
            dummy_right = torch.randn(1, 1, 64, 64)
            traced = torch.jit.trace(model, (dummy_left, dummy_right))

            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="left_glyph", shape=(1, 1, 64, 64)),
                    ct.TensorType(name="right_glyph", shape=(1, 1, 64, 64)),
                ],
                outputs=[ct.TensorType(name="kerning")],
                minimum_deployment_target=ct.target.macOS14,
            )
            mlmodel.save(str(coreml_dir / "KerningNet.mlpackage"))
            logger.info("  Saved KerningNet.mlpackage")
        except (RuntimeError, Exception) as e:
            logger.warning(f"  KerningNet conversion failed: {e}")
            logger.warning("  (coremltools may require Python 3.11-3.12)")

    # ── GlyphDiffusion UNet ──
    gd_ckpt = output_dir / "glyph_diffusion.pt"
    if gd_ckpt.exists():
        logger.info("Converting GlyphDiffusion UNet to CoreML...")
        try:
            from models.glyph_diffusion.config import ModelConfig
            from models.glyph_diffusion.model import GlyphDiffusionModel

            config = ModelConfig()
            model = GlyphDiffusionModel(config)
            state = torch.load(gd_ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            model.eval()

            # Replace einops.rearrange in AttentionBlock with trace-friendly ops
            for module in model.modules():
                if hasattr(module, "attention") and hasattr(module, "norm"):
                    # Monkey-patch AttentionBlock.forward for tracing
                    import types

                    def _trace_friendly_forward(self, x: torch.Tensor) -> torch.Tensor:
                        batch, channels, height, width = x.shape
                        h = self.norm(x)
                        h = h.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
                        h, _ = self.attention(h, h, h)
                        h = h.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
                        return x + h

                    module.forward = types.MethodType(_trace_friendly_forward, module)

            # Wrapper to make mask non-optional (required for export)
            class UNetExportWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x, timesteps, char_indices, style_embed, mask):
                    return self.model(x, timesteps, char_indices, style_embed, mask)

            wrapper = UNetExportWrapper(model)
            wrapper.eval()

            # Create dummy inputs (char_indices as int32 for CoreML)
            dummy_x = torch.randn(1, 1, 64, 64)
            dummy_t = torch.tensor([0.5])
            dummy_char = torch.tensor([0], dtype=torch.int32)
            dummy_style = torch.randn(1, 128)
            dummy_mask = torch.zeros(1, 1, 64, 64)

            # Use torch.export (torch.jit.trace has issues with coremltools 9+)
            from torch.export import export as torch_export

            exported = torch_export(
                wrapper,
                (dummy_x, dummy_t, dummy_char, dummy_style, dummy_mask),
            )
            exported = exported.run_decompositions({})

            mlmodel = ct.convert(
                exported,
                inputs=[
                    ct.TensorType(name="x", shape=(1, 1, 64, 64)),
                    ct.TensorType(name="timesteps", shape=(1,)),
                    ct.TensorType(name="char_indices", shape=(1,), dtype=np.int32),
                    ct.TensorType(name="style_embed", shape=(1, 128)),
                    ct.TensorType(name="mask", shape=(1, 1, 64, 64)),
                ],
                outputs=[ct.TensorType(name="velocity")],
                minimum_deployment_target=ct.target.macOS14,
            )
            mlmodel.save(str(coreml_dir / "GlyphDiffusion.mlpackage"))
            logger.info("  Saved GlyphDiffusion.mlpackage")
        except (RuntimeError, Exception) as e:
            logger.warning(f"  GlyphDiffusion conversion failed: {e}")
            logger.warning("  (coremltools may require Python 3.11-3.12)")

    logger.info(f"CoreML models saved to {coreml_dir}")


# ─── Training Curve Visualization ─────────────────────────────────────


def plot_all_curves(output_dir: Path) -> None:
    """Plot training curves for all models."""
    from evaluation.visualize import plot_training_curve

    for model_name in ["style_encoder", "kerning_net", "glyph_diffusion"]:
        history_path = output_dir / f"{model_name}_history.json"
        if not history_path.exists():
            continue

        history = json.loads(history_path.read_text())

        if "train_loss" in history and len(history["train_loss"]) > 1:
            plot_training_curve(
                history["train_loss"],
                output_dir / f"{model_name}_train_loss.png",
                title=f"{model_name} Training Loss",
            )
            logger.info(f"Saved {model_name} training curve")

        if "val_loss" in history and len(history["val_loss"]) > 1:
            plot_training_curve(
                history["val_loss"],
                output_dir / f"{model_name}_val_loss.png",
                title=f"{model_name} Validation Loss",
            )


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train all Typogenesis ML models end-to-end")
    parser.add_argument("--data-dir", type=Path, default=Path("data/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trained"))
    parser.add_argument("--style-epochs", type=int, default=30)
    parser.add_argument("--kerning-epochs", type=int, default=30)
    parser.add_argument("--diffusion-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-diffusion", action="store_true",
                        help="Skip GlyphDiffusion training (takes longest)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    t_total = time.time()

    # 1. Train StyleEncoder
    t0 = time.time()
    se_ckpt = train_style_encoder(
        args.data_dir, args.output_dir, device,
        epochs=args.style_epochs, batch_size=args.batch_size,
    )
    logger.info(f"StyleEncoder training took {time.time() - t0:.0f}s")

    # 2. Pre-compute style embeddings
    t0 = time.time()
    precompute_style_embeddings(se_ckpt, args.data_dir, device)
    logger.info(f"Style embedding computation took {time.time() - t0:.0f}s")

    # 3. Train KerningNet
    t0 = time.time()
    train_kerning_net(
        args.data_dir, args.output_dir, device,
        epochs=args.kerning_epochs, batch_size=64,
    )
    logger.info(f"KerningNet training took {time.time() - t0:.0f}s")

    # 4. Train GlyphDiffusion (longest)
    if not args.skip_diffusion:
        t0 = time.time()
        train_glyph_diffusion(
            args.data_dir, args.output_dir, device,
            epochs=args.diffusion_epochs, batch_size=args.batch_size,
        )
        logger.info(f"GlyphDiffusion training took {time.time() - t0:.0f}s")

    # 5. Evaluate
    t0 = time.time()
    results = evaluate_all(args.data_dir, args.output_dir, device)
    logger.info(f"Evaluation took {time.time() - t0:.0f}s")

    # Save evaluation results
    results_path = args.output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Evaluation results saved to {results_path}")

    # 6. Generate samples
    if not args.skip_diffusion:
        generate_samples(args.data_dir, args.output_dir, device)

    # 7. Plot training curves
    plot_all_curves(args.output_dir)

    # 8. Convert to CoreML
    convert_to_coreml(args.output_dir)

    total_time = time.time() - t_total
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\nOutputs in: {args.output_dir}")
    print("  - Checkpoints: style_encoder.pt, kerning_net.pt, glyph_diffusion.pt")
    print("  - Metrics: evaluation_results.json")
    print("  - Samples: generated_samples/, generated_grid.png")
    print("  - Curves: *_train_loss.png, *_val_loss.png")
    if (args.output_dir / "coreml").exists():
        print("  - CoreML: coreml/StyleEncoder.mlpackage, coreml/KerningNet.mlpackage")


if __name__ == "__main__":
    main()
