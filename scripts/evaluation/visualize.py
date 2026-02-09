"""Visualization tools for Typogenesis ML model evaluation.

Generates:
- Sample grids: full alphabet in multiple styles
- Embedding space plots: t-SNE/UMAP of style embeddings
- Kerning comparison: predicted vs ground-truth
- Training curves: loss and metrics over time

Usage:
    from evaluation.visualize import plot_sample_grid, plot_embedding_space
"""

import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_sample_grid(
    images: torch.Tensor,
    labels: list[str],
    output_path: Path,
    nrow: int = 10,
    title: str = "Generated Glyphs",
) -> None:
    """Create a grid of glyph images and save as PNG.

    Args:
        images: (N, 1, H, W) tensor of glyph images in [-1, 1].
        labels: List of N label strings (e.g., character names).
        output_path: Path to save the output image.
        nrow: Number of images per row.
        title: Title for the grid.
    """
    from PIL import Image, ImageDraw

    n = images.shape[0]
    h, w = images.shape[2], images.shape[3]

    # Denormalize to [0, 255]
    imgs = ((images + 1) * 127.5).clamp(0, 255).byte()

    cols = min(nrow, n)
    rows = (n + cols - 1) // cols

    cell_w = w + 4  # padding
    cell_h = h + 20  # padding + label space
    grid_w = cols * cell_w + 4
    grid_h = rows * cell_h + 30  # + title space

    grid = Image.new("L", (grid_w, grid_h), 32)
    draw = ImageDraw.Draw(grid)

    # Title
    draw.text((grid_w // 2 - len(title) * 3, 5), title, fill=200)

    for i in range(n):
        row, col = divmod(i, cols)
        x = col * cell_w + 2
        y = row * cell_h + 28

        # Paste glyph image
        glyph = Image.fromarray(imgs[i, 0].numpy(), mode="L")
        grid.paste(glyph, (x, y))

        # Draw label
        label = labels[i] if i < len(labels) else ""
        draw.text((x + w // 2 - len(label) * 3, y + h + 2), label, fill=180)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    logger.info(f"Saved sample grid to {output_path}")


def plot_embedding_space(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    method: str = "tsne",
    max_points: int = 2000,
    font_names: list[str] | None = None,
) -> None:
    """Plot 2D projection of the style embedding space.

    Args:
        embeddings: (N, D) embedding tensor.
        labels: (N,) integer font labels.
        output_path: Path to save the output image.
        method: Dimensionality reduction method ("tsne" or "pca").
        max_points: Maximum number of points to plot.
        font_names: Optional list of font names for the legend.
    """
    from PIL import Image, ImageDraw
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = min(max_points, len(embeddings))
    if n < len(embeddings):
        indices = torch.randperm(len(embeddings))[:n]
        embeddings = embeddings[indices]
        labels = labels[indices]

    embeds_np = embeddings.numpy()

    if method == "tsne":
        perplexity = min(30, n - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        reducer = PCA(n_components=2)

    coords = reducer.fit_transform(embeds_np)

    # Normalize to image coordinates
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    img_size = 800
    margin = 40

    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Generate colors for each unique label
    unique_labels = labels.unique().tolist()
    colors = _generate_colors(len(unique_labels))
    label_to_color = {lbl: colors[i] for i, lbl in enumerate(unique_labels)}

    for i in range(n):
        x = int(margin + (coords[i, 0] - x_min) / (x_max - x_min + 1e-8) * (img_size - 2 * margin))
        y = int(margin + (coords[i, 1] - y_min) / (y_max - y_min + 1e-8) * (img_size - 2 * margin))
        color = label_to_color[labels[i].item()]
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)

    # Title
    method_name = "t-SNE" if method == "tsne" else "PCA"
    draw.text((10, 5), f"Style Embedding Space ({method_name}), {len(unique_labels)} fonts", fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Saved embedding plot to {output_path}")


def plot_kerning_comparison(
    predictions: list[float],
    targets: list[float],
    pair_labels: list[str],
    output_path: Path,
    max_pairs: int = 50,
) -> None:
    """Plot predicted vs ground-truth kerning values as a bar chart.

    Args:
        predictions: List of predicted kerning values.
        targets: List of ground-truth kerning values.
        pair_labels: List of pair labels (e.g., "AV", "To").
        output_path: Path to save the output image.
        max_pairs: Maximum number of pairs to show.
    """
    from PIL import Image, ImageDraw

    n = min(max_pairs, len(predictions))

    img_w = max(600, n * 20 + 80)
    img_h = 400
    margin_left = 60
    margin_bottom = 60
    margin_top = 30

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Find range
    all_vals = predictions[:n] + targets[:n]
    v_min = min(all_vals) if all_vals else -1
    v_max = max(all_vals) if all_vals else 1
    v_range = max(v_max - v_min, 0.01)

    plot_h = img_h - margin_top - margin_bottom
    plot_w = img_w - margin_left - 20
    bar_w = max(3, plot_w // (n * 2 + n))

    # Draw zero line
    zero_y = int(margin_top + plot_h * (v_max / v_range))
    draw.line([(margin_left, zero_y), (img_w - 20, zero_y)], fill=(200, 200, 200))

    for i in range(n):
        x = margin_left + i * (bar_w * 2 + bar_w)

        # Target bar (blue)
        t_val = targets[i]
        t_h = int(plot_h * abs(t_val) / v_range)
        t_y = zero_y - t_h if t_val >= 0 else zero_y
        draw.rectangle([x, t_y, x + bar_w, t_y + t_h], fill=(70, 130, 180))

        # Prediction bar (orange)
        p_val = predictions[i]
        p_h = int(plot_h * abs(p_val) / v_range)
        p_y = zero_y - p_h if p_val >= 0 else zero_y
        draw.rectangle([x + bar_w, p_y, x + 2 * bar_w, p_y + p_h], fill=(255, 140, 0))

    # Title and legend
    draw.text((10, 5), "Kerning: Ground Truth (blue) vs Predicted (orange)", fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Saved kerning comparison to {output_path}")


def plot_training_curve(
    losses: list[float],
    output_path: Path,
    title: str = "Training Loss",
    metric_name: str = "Loss",
) -> None:
    """Plot a training loss curve and save as PNG.

    Args:
        losses: List of loss values per step/epoch.
        output_path: Path to save the output image.
        title: Plot title.
        metric_name: Name of the metric being plotted.
    """
    from PIL import Image, ImageDraw

    img_w, img_h = 600, 300
    margin = 50

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    n = len(losses)
    if n < 2:
        return

    v_min = min(losses)
    v_max = max(losses)
    v_range = max(v_max - v_min, 1e-8)

    plot_w = img_w - 2 * margin
    plot_h = img_h - 2 * margin

    points = []
    for i, loss in enumerate(losses):
        x = margin + int(i / (n - 1) * plot_w)
        y = margin + int((1 - (loss - v_min) / v_range) * plot_h)
        points.append((x, y))

    # Draw line
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(70, 130, 180), width=2)

    # Axes
    draw.line([(margin, margin), (margin, img_h - margin)], fill=(0, 0, 0))
    draw.line([(margin, img_h - margin), (img_w - margin, img_h - margin)], fill=(0, 0, 0))

    # Title
    draw.text((img_w // 2 - len(title) * 3, 5), title, fill=(0, 0, 0))

    # Axis labels
    draw.text((margin - 5, margin - 15), f"{v_max:.4f}", fill=(100, 100, 100))
    draw.text((margin - 5, img_h - margin + 5), f"{v_min:.4f}", fill=(100, 100, 100))
    draw.text((img_w // 2, img_h - 15), "Step", fill=(100, 100, 100))
    draw.text((5, img_h // 2), metric_name, fill=(100, 100, 100))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Saved training curve to {output_path}")


def _generate_colors(n: int) -> list[tuple[int, int, int]]:
    """Generate n visually distinct colors using HSV color space.

    Args:
        n: Number of colors to generate.

    Returns:
        List of (R, G, B) tuples.
    """
    import colorsys
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        s = 0.7 + (i % 3) * 0.1
        v = 0.8 + (i % 2) * 0.1
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors
