"""Glyph extraction from font files for ML training.

Parses font files using fonttools, renders glyphs to images using Pillow,
extracts bezier outlines, metrics, and kerning tables.

Usage:
    python -m data.extract_glyphs --font-dir data/fonts --output-dir data/extracted
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def font_file_hash(font_path: Path) -> str:
    """Compute a short hash for a font file to use as directory name.

    Args:
        font_path: Path to the font file.

    Returns:
        8-character hex hash string.
    """
    h = hashlib.sha256()
    h.update(font_path.read_bytes())
    return h.hexdigest()[:12]


def get_font_metrics(tt_font: TTFont) -> dict:
    """Extract font-level metrics from a TTFont object.

    Args:
        tt_font: Parsed font object.

    Returns:
        Dict with UPM, ascender, descender, x-height, cap-height.
    """
    head = tt_font.get("head")
    os2 = tt_font.get("OS/2")
    hhea = tt_font.get("hhea")

    metrics = {
        "units_per_em": head.unitsPerEm if head else 1000,
    }

    if os2:
        metrics["ascender"] = os2.sTypoAscender
        metrics["descender"] = os2.sTypoDescender
        metrics["x_height"] = getattr(os2, "sxHeight", None)
        metrics["cap_height"] = getattr(os2, "sCapHeight", None)
    elif hhea:
        metrics["ascender"] = hhea.ascent
        metrics["descender"] = hhea.descent
        metrics["x_height"] = None
        metrics["cap_height"] = None

    return metrics


def extract_kerning_pairs(tt_font: TTFont) -> list[dict]:
    """Extract kerning pairs from GPOS and legacy kern tables.

    Args:
        tt_font: Parsed font object.

    Returns:
        List of dicts with keys: left, right, value (in font units).
    """
    pairs = {}

    # Try legacy kern table first
    kern_table = tt_font.get("kern")
    if kern_table:
        for subtable in kern_table.kernTables:
            if hasattr(subtable, "kernTable"):
                for (left_name, right_name), value in subtable.kernTable.items():
                    if value != 0:
                        pairs[(left_name, right_name)] = value

    # Try GPOS table (more common in modern fonts)
    gpos = tt_font.get("GPOS")
    if gpos and hasattr(gpos, "table") and gpos.table:
        cmap = tt_font.getBestCmap() or {}
        reverse_cmap = {v: chr(k) for k, v in cmap.items() if 32 <= k < 0x10000}

        try:
            for lookup in gpos.table.LookupList.Lookup:
                if lookup.LookupType == 2:  # PairPos
                    for subtable in lookup.SubTable:
                        if hasattr(subtable, "Format") and subtable.Format == 1:
                            # Format 1: individual pairs
                            for ps_idx, pair_set in enumerate(subtable.PairSet):
                                if ps_idx >= len(subtable.Coverage.glyphs):
                                    continue
                                left_glyph = subtable.Coverage.glyphs[ps_idx]
                                for pvr in pair_set.PairValueRecord:
                                    right_glyph = pvr.SecondGlyph
                                    val = 0
                                    if pvr.Value1 and hasattr(pvr.Value1, "XAdvance"):
                                        val = pvr.Value1.XAdvance
                                    if val != 0:
                                        left_char = reverse_cmap.get(left_glyph, "")
                                        right_char = reverse_cmap.get(right_glyph, "")
                                        if left_char and right_char:
                                            pairs[(left_char, right_char)] = val
                        elif hasattr(subtable, "Format") and subtable.Format == 2:
                            # Format 2: class-based pairs - more complex
                            # Skip for now; individual pairs cover most critical cases
                            pass
        except (AttributeError, IndexError):
            # Malformed GPOS table, skip
            pass

    result = []
    for (left, right), value in pairs.items():
        # Filter to our character set
        if left in ALL_CHARACTERS and right in ALL_CHARACTERS:
            result.append({"left": left, "right": right, "value": value})

    return result


def extract_glyph_outline(tt_font: TTFont, char: str) -> list | None:
    """Extract bezier outline for a character.

    Args:
        tt_font: Parsed font object.
        char: Single character to extract.

    Returns:
        List of contours, each a list of path operations, or None if extraction fails.
    """
    cmap = tt_font.getBestCmap()
    if not cmap:
        return None

    codepoint = ord(char)
    if codepoint not in cmap:
        return None

    glyph_name = cmap[codepoint]
    glyph_set = tt_font.getGlyphSet()

    if glyph_name not in glyph_set:
        return None

    pen = RecordingPen()
    try:
        glyph_set[glyph_name].draw(pen)
    except Exception:
        return None

    # Convert recording to serializable format
    contours = []
    current_contour = []

    for op, args in pen.value:
        if op == "moveTo":
            if current_contour:
                contours.append(current_contour)
            current_contour = [{"type": "moveTo", "points": [_point_to_list(args[0])]}]
        elif op == "lineTo":
            current_contour.append({"type": "lineTo", "points": [_point_to_list(args[0])]})
        elif op == "curveTo":
            current_contour.append({
                "type": "curveTo",
                "points": [_point_to_list(p) for p in args],
            })
        elif op == "qCurveTo":
            current_contour.append({
                "type": "qCurveTo",
                "points": [_point_to_list(p) for p in args],
            })
        elif op == "closePath" or op == "endPath":
            if current_contour:
                contours.append(current_contour)
                current_contour = []

    if current_contour:
        contours.append(current_contour)

    return contours if contours else None


def _point_to_list(pt) -> list[float]:
    """Convert a point tuple to a JSON-serializable list."""
    return [float(pt[0]), float(pt[1])]


def render_glyph_to_image(
    font_path: Path,
    char: str,
    image_size: int,
    padding_fraction: float = 0.1,
) -> Image.Image | None:
    """Render a single glyph character to a grayscale image.

    Uses Pillow's ImageFont to render the character centered in the image
    with anti-aliasing, normalized to fill the frame with padding.

    Args:
        font_path: Path to the TTF/OTF font file.
        char: Single character to render.
        image_size: Output image size (square).
        padding_fraction: Fraction of image size to use as padding on each side.

    Returns:
        Grayscale PIL Image, or None if rendering fails.
    """
    # Render at higher resolution and downsample for better anti-aliasing
    render_size = image_size * 4
    padding = int(render_size * padding_fraction)
    usable_size = render_size - 2 * padding

    try:
        # Start with a large font size and find the right one
        # Binary search for optimal font size
        lo, hi = 10, render_size * 2
        best_font_size = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(str(font_path), mid)
            bbox = font.getbbox(char)
            if bbox is None:
                return None
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w <= usable_size and h <= usable_size:
                best_font_size = mid
                lo = mid + 1
            else:
                hi = mid - 1

        font = ImageFont.truetype(str(font_path), best_font_size)
        bbox = font.getbbox(char)
        if bbox is None:
            return None

        glyph_w = bbox[2] - bbox[0]
        glyph_h = bbox[3] - bbox[1]

        if glyph_w == 0 or glyph_h == 0:
            return None

        # Create high-res image
        img = Image.new("L", (render_size, render_size), 0)
        draw = ImageDraw.Draw(img)

        # Center the glyph
        x = (render_size - glyph_w) // 2 - bbox[0]
        y = (render_size - glyph_h) // 2 - bbox[1]

        draw.text((x, y), char, fill=255, font=font)

        # Downsample with high-quality resampling
        img = img.resize((image_size, image_size), Image.LANCZOS)

        return img

    except Exception as e:
        logger.debug(f"Failed to render '{char}' from {font_path}: {e}")
        return None


def extract_font(
    font_path: Path,
    output_dir: Path,
    image_sizes: tuple[int, ...] = (64, 128),
) -> dict | None:
    """Extract all training data from a single font file.

    Args:
        font_path: Path to TTF/OTF file.
        output_dir: Root output directory.
        image_sizes: Image sizes to render at.

    Returns:
        Extraction summary dict, or None if font is invalid.
    """
    file_hash = font_file_hash(font_path)
    font_dir = output_dir / file_hash
    metadata_path = font_dir / "metadata.json"

    # Skip if already extracted
    if metadata_path.exists():
        return None

    try:
        tt_font = TTFont(font_path)
    except Exception as e:
        logger.debug(f"Failed to parse {font_path}: {e}")
        return None

    cmap = tt_font.getBestCmap()
    if not cmap:
        return None

    # Check character coverage
    font_chars = {chr(cp) for cp in cmap}
    available_chars = [c for c in ALL_CHARACTERS if c in font_chars]
    if len(available_chars) < 52:  # At least all letters
        return None

    # Extract metrics
    metrics = get_font_metrics(tt_font)

    # Get font name
    name_table = tt_font.get("name")
    font_name = str(font_path.stem)
    if name_table:
        for record in name_table.names:
            if record.nameID == 4:  # Full font name
                try:
                    font_name = record.toUnicode()
                    break
                except Exception:
                    pass

    # Extract glyph advance widths and bearings
    hmtx = tt_font.get("hmtx")
    glyph_metrics = {}
    if hmtx:
        for char in available_chars:
            cp = ord(char)
            if cp in cmap:
                glyph_name = cmap[cp]
                if glyph_name in hmtx.metrics:
                    width, lsb = hmtx.metrics[glyph_name]
                    glyph_metrics[char] = {"advance_width": width, "lsb": lsb}

    # Create output directory
    glyphs_dir = font_dir / "glyphs"
    glyphs_dir.mkdir(parents=True, exist_ok=True)

    # Render glyphs at each size
    rendered_count = 0
    for char in available_chars:
        for size in image_sizes:
            img = render_glyph_to_image(font_path, char, size)
            if img is not None:
                # Use character name that's filesystem-safe
                char_name = f"U+{ord(char):04X}"  # e.g. U+0041 for 'A'
                img_path = glyphs_dir / f"{char_name}_{size}.png"
                img.save(img_path)
                rendered_count += 1

        # Extract outline
        outline = extract_glyph_outline(tt_font, char)
        if outline:
            char_name = f"U+{ord(char):04X}"
            outline_path = glyphs_dir / f"{char_name}_outline.json"
            outline_path.write_text(json.dumps(outline))

    if rendered_count == 0:
        # Clean up
        import shutil
        shutil.rmtree(font_dir, ignore_errors=True)
        return None

    # Extract kerning
    kerning_pairs = extract_kerning_pairs(tt_font)
    kerning_path = font_dir / "kerning.json"
    kerning_path.write_text(json.dumps(kerning_pairs, indent=2))

    # Save metadata
    metadata = {
        "font_name": font_name,
        "font_path": str(font_path),
        "file_hash": file_hash,
        "metrics": metrics,
        "glyph_metrics": glyph_metrics,
        "available_characters": available_chars,
        "num_glyphs_rendered": rendered_count,
        "num_kerning_pairs": len(kerning_pairs),
        "image_sizes": list(image_sizes),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    tt_font.close()
    return metadata


def extract_all_fonts(
    font_dir: Path,
    output_dir: Path,
    image_sizes: tuple[int, ...] = (64, 128),
) -> dict:
    """Extract glyphs from all fonts in a directory tree.

    Args:
        font_dir: Directory containing font files (searched recursively).
        output_dir: Directory for extracted data.
        image_sizes: Image sizes to render at.

    Returns:
        Summary dict with extraction statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    font_files = list(font_dir.rglob("*.ttf")) + list(font_dir.rglob("*.otf"))
    logger.info(f"Found {len(font_files)} font files in {font_dir}")

    extracted = 0
    skipped = 0
    failed = 0

    for i, font_path in enumerate(font_files):
        result = extract_font(font_path, output_dir, image_sizes)
        if result is None:
            # Check if skipped vs failed
            file_hash = font_file_hash(font_path)
            if (output_dir / file_hash / "metadata.json").exists():
                skipped += 1
            else:
                failed += 1
        else:
            extracted += 1
            logger.info(
                f"[{i + 1}/{len(font_files)}] Extracted {result['font_name']}: "
                f"{result['num_glyphs_rendered']} images, "
                f"{result['num_kerning_pairs']} kerning pairs"
            )

        if (i + 1) % 100 == 0:
            logger.info(
                f"Progress: {i + 1}/{len(font_files)} "
                f"(extracted={extracted}, skipped={skipped}, failed={failed})"
            )

    summary = {
        "total_fonts": len(font_files),
        "extracted": extracted,
        "skipped": skipped,
        "failed": failed,
        "output_dir": str(output_dir),
    }
    logger.info(f"Extraction complete: {summary}")

    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract glyphs from font files")
    parser.add_argument(
        "--font-dir", type=Path, required=True,
        help="Directory containing font files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/extracted"),
        help="Directory for extracted data",
    )
    parser.add_argument(
        "--image-sizes", type=int, nargs="+", default=[64, 128],
        help="Image sizes to render at",
    )
    args = parser.parse_args()

    summary = extract_all_fonts(args.font_dir, args.output_dir, tuple(args.image_sizes))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
