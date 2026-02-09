"""Evaluation metrics for all three Typogenesis ML models.

Provides quantitative metrics for:
- StyleEncoder: same-font similarity, cross-font similarity, retrieval accuracy
- KerningNet: MAE, direction accuracy, critical pair MAE
- GlyphDiffusion: style consistency, character accuracy (FID requires scipy)

Usage:
    python -m evaluation.metrics --checkpoint-dir checkpoints --data-dir data/extracted
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Critical kerning pairs that users care most about
CRITICAL_PAIRS = [
    ("A", "V"), ("A", "W"), ("A", "Y"), ("A", "T"),
    ("T", "a"), ("T", "e"), ("T", "o"), ("T", "r"),
    ("V", "a"), ("V", "e"), ("V", "o"),
    ("W", "a"), ("W", "e"), ("W", "o"),
    ("Y", "a"), ("Y", "e"), ("Y", "o"),
    ("L", "T"), ("L", "V"), ("L", "W"), ("L", "Y"),
    ("P", "a"), ("P", "e"), ("P", "o"),
    ("F", "a"), ("F", "e"), ("F", "o"),
    ("f", "i"), ("f", "f"), ("f", "l"),  # Ligature-related pairs
    ("r", "."), ("r", ","),              # Punctuation spacing
]


# ─── StyleEncoder Metrics ─────────────────────────────────────────────


def compute_style_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute style embeddings for all samples in a dataloader.

    Args:
        model: StyleEncoder model.
        dataloader: DataLoader yielding (glyphs, font_labels) batches.
        device: Compute device.

    Returns:
        Tuple of (embeddings, labels) tensors.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for glyphs, labels in dataloader:
            glyphs = glyphs.to(device)
            embeddings = model(glyphs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def same_font_similarity(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """Average cosine similarity between embeddings from the same font.

    Args:
        embeddings: (N, D) embedding tensor.
        labels: (N,) font label tensor.

    Returns:
        Mean cosine similarity for same-font pairs.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = embeddings @ embeddings.T  # (N, N)

    # Same-font mask (excluding self-pairs)
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    eye_mask = ~torch.eye(len(labels), dtype=torch.bool)
    same_mask = label_matrix & eye_mask

    if same_mask.sum() == 0:
        return 0.0

    return sim_matrix[same_mask].mean().item()


def cross_font_similarity(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """Average cosine similarity between embeddings from different fonts.

    Args:
        embeddings: (N, D) embedding tensor.
        labels: (N,) font label tensor.

    Returns:
        Mean cosine similarity for cross-font pairs.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = embeddings @ embeddings.T

    cross_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    if cross_mask.sum() == 0:
        return 0.0

    return sim_matrix[cross_mask].mean().item()


def retrieval_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute retrieval accuracy: given a glyph, can we find other glyphs from the same font?

    For each embedding, ranks all other embeddings by cosine similarity and
    checks if the top-k results include a same-font glyph.

    Args:
        embeddings: (N, D) embedding tensor.
        labels: (N,) font label tensor.
        top_k: Tuple of k values to compute accuracy for.

    Returns:
        Dict mapping "top_k" to accuracy float.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = embeddings @ embeddings.T
    n = len(labels)

    # Mask out self-similarity
    sim_matrix.fill_diagonal_(float("-inf"))

    results = {}
    for k in top_k:
        correct = 0
        for i in range(n):
            # Top-k most similar
            topk_indices = sim_matrix[i].topk(min(k, n - 1)).indices
            topk_labels = labels[topk_indices]
            if labels[i] in topk_labels:
                correct += 1
        results[f"top_{k}"] = correct / n

    return results


# ─── KerningNet Metrics ───────────────────────────────────────────────


def kerning_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    units_per_em: float = 1000.0,
    max_kerning: float = 200.0,
) -> float:
    """Mean absolute error of kerning predictions in font units.

    Args:
        predictions: Model output (normalized).
        targets: Ground truth (normalized).
        units_per_em: Font UPM for denormalization context.
        max_kerning: Max kerning used for normalization.

    Returns:
        MAE in the same units as the input (normalized).
    """
    return (predictions - targets).abs().mean().item() * max_kerning


def kerning_direction_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy of predicting the correct kerning direction (sign).

    Args:
        predictions: Model output.
        targets: Ground truth.

    Returns:
        Fraction of predictions with correct sign (tighter/looser/none).
    """
    # For zero targets, any small prediction is "correct"
    threshold = 0.01
    pred_sign = torch.sign(predictions)
    target_sign = torch.sign(targets)

    # Treat near-zero predictions as zero
    pred_sign[predictions.abs() < threshold] = 0
    target_sign[targets.abs() < threshold] = 0

    correct = (pred_sign == target_sign).float().mean()
    return correct.item()


def critical_pair_mae(
    predictions: dict[tuple[str, str], float],
    targets: dict[tuple[str, str], float],
    max_kerning: float = 200.0,
) -> float:
    """MAE specifically on critical kerning pairs.

    Args:
        predictions: Dict mapping (left, right) char pairs to predicted kerning.
        targets: Dict mapping (left, right) char pairs to ground truth kerning.
        max_kerning: Max kerning for denormalization.

    Returns:
        MAE on critical pairs in font units, or -1 if no critical pairs found.
    """
    errors = []
    for pair in CRITICAL_PAIRS:
        if pair in predictions and pair in targets:
            error = abs(predictions[pair] - targets[pair]) * max_kerning
            errors.append(error)

    if not errors:
        return -1.0
    return sum(errors) / len(errors)


# ─── GlyphDiffusion Metrics ──────────────────────────────────────────


def style_consistency(
    model: torch.nn.Module,
    style_encoder: torch.nn.Module,
    char_indices: list[int],
    style_embed: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = 50,
) -> float:
    """Measure style consistency of generated glyphs.

    Generates multiple characters with the same style embedding, then
    measures the variance of their style embeddings as encoded by
    the StyleEncoder. Lower variance = more consistent style.

    Args:
        model: GlyphDiffusion model.
        style_encoder: Trained StyleEncoder.
        char_indices: List of character indices to generate.
        style_embed: Style embedding to condition on (1, 128).
        device: Compute device.
        num_inference_steps: Number of diffusion steps.

    Returns:
        Standard deviation of style embeddings across generated chars.
    """
    from models.glyph_diffusion import FlowMatchingSchedule, sample_euler

    model.eval()
    style_encoder.eval()
    schedule = FlowMatchingSchedule()

    generated_embeddings = []
    batch_size = len(char_indices)

    with torch.no_grad():
        # Generate all characters
        x = torch.randn(batch_size, 1, 64, 64, device=device)
        chars = torch.tensor(char_indices, device=device)
        style = style_embed.expand(batch_size, -1).to(device)

        generated = sample_euler(model, x, chars, style, schedule)

        # Encode generated glyphs with style encoder
        embeddings = style_encoder(generated)
        generated_embeddings.append(embeddings.cpu())

    all_embeds = torch.cat(generated_embeddings)
    # Compute mean embedding and standard deviation
    mean_embed = all_embeds.mean(dim=0)
    deviations = (all_embeds - mean_embed).norm(dim=1)

    return deviations.mean().item()


def compute_fid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    """Compute Frechet Inception Distance between real and generated feature distributions.

    Args:
        real_features: (N, D) features from real images.
        generated_features: (M, D) features from generated images.

    Returns:
        FID score (lower is better).
    """
    from scipy.linalg import sqrtm

    mu_real = real_features.mean(dim=0).numpy()
    mu_gen = generated_features.mean(dim=0).numpy()

    # Covariance matrices
    real_np = real_features.numpy()
    gen_np = generated_features.numpy()
    sigma_real = np.cov(real_np, rowvar=False)
    sigma_gen = np.cov(gen_np, rowvar=False)

    # FID = ||mu_r - mu_g||^2 + Tr(sigma_r + sigma_g - 2*sqrt(sigma_r @ sigma_g))
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


# ─── Aggregated Evaluation ────────────────────────────────────────────


def evaluate_style_encoder(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Run all StyleEncoder evaluation metrics.

    Args:
        model: Trained StyleEncoder.
        dataloader: Evaluation DataLoader.
        device: Compute device.

    Returns:
        Dict with all metric values.
    """
    embeddings, labels = compute_style_embeddings(model, dataloader, device)

    same_sim = same_font_similarity(embeddings, labels)
    cross_sim = cross_font_similarity(embeddings, labels)
    retrieval = retrieval_accuracy(embeddings, labels, top_k=(1, 5, 10))

    results = {
        "same_font_similarity": same_sim,
        "cross_font_similarity": cross_sim,
        **{f"retrieval_{k}": v for k, v in retrieval.items()},
        "num_embeddings": len(embeddings),
        "num_fonts": len(labels.unique()),
    }

    logger.info(f"StyleEncoder evaluation: {json.dumps(results, indent=2)}")
    return results


def evaluate_kerning_net(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_kerning: float = 200.0,
) -> dict:
    """Run all KerningNet evaluation metrics.

    Args:
        model: Trained KerningNet.
        dataloader: Evaluation DataLoader.
        device: Compute device.
        max_kerning: Normalization factor.

    Returns:
        Dict with all metric values.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for left, right, target in dataloader:
            left, right = left.to(device), right.to(device)
            pred = model(left, right).cpu().squeeze(-1)
            all_preds.append(pred)
            all_targets.append(target)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mae = kerning_mae(preds, targets, max_kerning=max_kerning)
    direction_acc = kerning_direction_accuracy(preds, targets)

    results = {
        "kerning_mae_font_units": mae,
        "direction_accuracy": direction_acc,
        "num_pairs": len(preds),
    }

    logger.info(f"KerningNet evaluation: {json.dumps(results, indent=2)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation output will be saved to {args.output_dir}")

    # This is a CLI entry point - actual evaluation requires trained models
    # which will be available after training phases are complete
    logger.info("Evaluation framework ready. Train models first, then run evaluation.")


if __name__ == "__main__":
    main()
