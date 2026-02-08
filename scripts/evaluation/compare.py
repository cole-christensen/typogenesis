"""A/B comparison tooling for Typogenesis model evaluation.

Provides side-by-side comparison between:
- Real vs generated glyphs
- Different model checkpoints
- Different training configurations

Usage:
    from evaluation.compare import compare_fonts, compare_checkpoints
"""

import logging
from pathlib import Path

import torch
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def create_ab_panel(
    images_a: list[Image.Image],
    images_b: list[Image.Image],
    labels: list[str],
    output_path: Path,
    title_a: str = "A (Real)",
    title_b: str = "B (Generated)",
    cell_size: int = 64,
) -> None:
    """Create a side-by-side A/B comparison panel.

    Displays two rows of images (A on top, B on bottom) with labels,
    allowing visual comparison of real vs generated or old vs new.

    Args:
        images_a: List of PIL images for panel A.
        images_b: List of PIL images for panel B (same length as A).
        labels: List of label strings for each column.
        output_path: Path to save the output image.
        title_a: Title for the top row.
        title_b: Title for the bottom row.
        cell_size: Size of each glyph cell.
    """
    n = min(len(images_a), len(images_b))
    if n == 0:
        return

    padding = 4
    label_height = 16
    title_width = 80
    header_height = 24

    img_w = title_width + n * (cell_size + padding) + padding
    img_h = header_height + 2 * (cell_size + label_height + padding) + padding

    panel = Image.new("L", (img_w, img_h), 32)
    draw = ImageDraw.Draw(panel)

    # Header labels
    for i, label in enumerate(labels[:n]):
        x = title_width + i * (cell_size + padding)
        draw.text((x + cell_size // 2 - len(label) * 3, 4), label, fill=200)

    # Row A
    y_a = header_height
    draw.text((4, y_a + cell_size // 2 - 6), title_a[:10], fill=180)
    for i in range(n):
        x = title_width + i * (cell_size + padding)
        img = images_a[i].convert("L").resize((cell_size, cell_size))
        panel.paste(img, (x, y_a))

    # Row B
    y_b = header_height + cell_size + label_height + padding
    draw.text((4, y_b + cell_size // 2 - 6), title_b[:10], fill=180)
    for i in range(n):
        x = title_width + i * (cell_size + padding)
        img = images_b[i].convert("L").resize((cell_size, cell_size))
        panel.paste(img, (x, y_b))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    logger.info(f"Saved A/B panel ({n} pairs) to {output_path}")


def compare_generated_vs_real(
    model: torch.nn.Module,
    real_images: dict[str, Image.Image],
    style_embed: torch.Tensor,
    output_path: Path,
    characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    device: torch.device | None = None,
) -> None:
    """Generate characters and compare against real font images.

    Args:
        model: GlyphDiffusion model.
        real_images: Dict mapping character to real PIL image.
        style_embed: Style embedding for generation (1, 128).
        output_path: Path to save the comparison panel.
        characters: Characters to compare.
        device: Compute device.
    """
    from models.glyph_diffusion import FlowMatchingSchedule, sample_euler
    from models.glyph_diffusion.config import CHAR_TO_IDX

    if device is None:
        device = torch.device("cpu")
    model.eval()
    schedule = FlowMatchingSchedule()

    chars_to_compare = [c for c in characters if c in real_images and c in CHAR_TO_IDX]
    if not chars_to_compare:
        logger.warning("No matching characters for comparison")
        return

    # Generate glyphs
    n = len(chars_to_compare)
    char_indices = torch.tensor([CHAR_TO_IDX[c] for c in chars_to_compare], device=device)
    style = style_embed.expand(n, -1).to(device)

    with torch.no_grad():
        x = torch.randn(n, 1, 64, 64, device=device)
        generated = sample_euler(model, x, char_indices, style, schedule)

    # Convert generated tensors to PIL images
    gen_images = []
    for i in range(n):
        arr = ((generated[i, 0].cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
        gen_images.append(Image.fromarray(arr, mode="L"))

    real_list = [real_images[c] for c in chars_to_compare]

    create_ab_panel(
        real_list,
        gen_images,
        chars_to_compare,
        output_path,
        title_a="Real",
        title_b="Generated",
    )


def compare_checkpoints(
    checkpoint_a: Path,
    checkpoint_b: Path,
    characters: str = "ABCDEFGHIJabcdefghij0123456789",
    style_embed: torch.Tensor | None = None,
    output_path: Path = Path("outputs/checkpoint_comparison.png"),
    device: torch.device | None = None,
) -> None:
    """Compare generations from two different checkpoints.

    Args:
        checkpoint_a: Path to first checkpoint.
        checkpoint_b: Path to second checkpoint.
        characters: Characters to generate.
        style_embed: Style embedding (if None, uses random).
        output_path: Path to save comparison.
        device: Compute device.
    """
    from models.glyph_diffusion import FlowMatchingSchedule, GlyphDiffusionModel, sample_euler
    from models.glyph_diffusion.config import CHAR_TO_IDX

    if device is None:
        device = torch.device("cpu")
    if style_embed is None:
        torch.manual_seed(42)
        style_embed = torch.randn(1, 128)

    schedule = FlowMatchingSchedule()
    chars = [c for c in characters if c in CHAR_TO_IDX]
    n = len(chars)
    char_indices = torch.tensor([CHAR_TO_IDX[c] for c in chars], device=device)
    style = style_embed.expand(n, -1).to(device)

    # Fixed noise for fair comparison
    torch.manual_seed(42)
    x = torch.randn(n, 1, 64, 64, device=device)

    images_a = []
    images_b = []

    for ckpt_path, image_list in [(checkpoint_a, images_a), (checkpoint_b, images_b)]:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = GlyphDiffusionModel()
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            generated = sample_euler(model, x.clone(), char_indices, style, schedule)

        for i in range(n):
            arr = ((generated[i, 0].cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
            image_list.append(Image.fromarray(arr, mode="L"))

    create_ab_panel(
        images_a,
        images_b,
        chars,
        output_path,
        title_a=checkpoint_a.stem,
        title_b=checkpoint_b.stem,
    )
