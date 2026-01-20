#!/usr/bin/env python3
"""
Extract glyph images and outlines from TTF/OTF fonts.

This script parses font files using fonttools, renders glyphs to images,
extracts outline data (bezier curves), and saves as PNG + JSON pairs.

Usage:
    python extract_glyphs.py --input-dir ./fonts --output-dir ./glyphs
    python extract_glyphs.py --input-dir ./fonts --output-dir ./glyphs --size 128
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.boundsPen import BoundsPen
from PIL import Image, ImageDraw
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default character set to extract
DEFAULT_CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
    " "
)

# Extended character set
EXTENDED_CHARSET = DEFAULT_CHARSET + (
    "\u00c0\u00c1\u00c2\u00c3\u00c4\u00c5\u00c6\u00c7\u00c8\u00c9\u00ca\u00cb"  # Latin Extended A-Z
    "\u00cc\u00cd\u00ce\u00cf\u00d0\u00d1\u00d2\u00d3\u00d4\u00d5\u00d6\u00d8"
    "\u00d9\u00da\u00db\u00dc\u00dd\u00de\u00df"
    "\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb"  # Latin Extended a-z
    "\u00ec\u00ed\u00ee\u00ef\u00f0\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f8"
    "\u00f9\u00fa\u00fb\u00fc\u00fd\u00fe\u00ff"
)


@dataclass
class PathPoint:
    """A point in a glyph path."""
    x: float
    y: float
    type: str  # "move", "line", "curve", "qcurve"
    on_curve: bool = True


@dataclass
class PathContour:
    """A contour (closed path) in a glyph."""
    points: list[dict] = field(default_factory=list)
    closed: bool = True


@dataclass
class GlyphOutline:
    """Outline data for a glyph."""
    contours: list[dict] = field(default_factory=list)
    bounds: Optional[tuple[float, float, float, float]] = None
    advance_width: float = 0.0
    left_side_bearing: float = 0.0


@dataclass
class GlyphMetadata:
    """Metadata for an extracted glyph."""
    character: str
    unicode: int
    glyph_name: str
    font_family: str
    font_style: str
    units_per_em: int
    ascender: int
    descender: int
    outline: dict = field(default_factory=dict)
    image_sizes: list[int] = field(default_factory=list)


class GlyphExtractor:
    """Extract glyphs from font files."""

    def __init__(
        self,
        output_dir: Path,
        image_sizes: list[int] = None,
        charset: str = DEFAULT_CHARSET,
        padding: float = 0.1
    ):
        """
        Initialize the glyph extractor.

        Args:
            output_dir: Directory to save extracted glyphs
            image_sizes: List of image sizes to render (e.g., [64, 128])
            charset: Characters to extract
            padding: Padding ratio around glyphs (0.1 = 10%)
        """
        self.output_dir = Path(output_dir)
        self.image_sizes = image_sizes or [64, 128]
        self.charset = charset
        self.padding = padding

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for size in self.image_sizes:
            (self.output_dir / f"images_{size}").mkdir(exist_ok=True)
        (self.output_dir / "outlines").mkdir(exist_ok=True)

    def extract_outline(self, font: TTFont, glyph_name: str) -> GlyphOutline:
        """
        Extract outline data from a glyph.

        Args:
            font: TTFont object
            glyph_name: Name of the glyph to extract

        Returns:
            GlyphOutline with bezier path data
        """
        glyph_set = font.getGlyphSet()

        if glyph_name not in glyph_set:
            return GlyphOutline()

        # Get advance width
        hmtx = font.get("hmtx")
        advance_width = 0.0
        lsb = 0.0
        if hmtx and glyph_name in hmtx.metrics:
            advance_width, lsb = hmtx.metrics[glyph_name]

        # Record the drawing operations
        pen = RecordingPen()
        try:
            glyph_set[glyph_name].draw(pen)
        except Exception as e:
            logger.debug(f"Could not draw glyph {glyph_name}: {e}")
            return GlyphOutline(advance_width=advance_width, left_side_bearing=lsb)

        # Get bounds
        bounds_pen = BoundsPen(glyph_set)
        try:
            glyph_set[glyph_name].draw(bounds_pen)
            bounds = bounds_pen.bounds
        except Exception:
            bounds = None

        # Convert recording to contours
        contours = []
        current_contour = []

        for operation, args in pen.value:
            if operation == "moveTo":
                if current_contour:
                    contours.append(PathContour(
                        points=current_contour,
                        closed=True
                    ))
                current_contour = [{
                    "x": args[0][0],
                    "y": args[0][1],
                    "type": "move",
                    "on_curve": True
                }]

            elif operation == "lineTo":
                current_contour.append({
                    "x": args[0][0],
                    "y": args[0][1],
                    "type": "line",
                    "on_curve": True
                })

            elif operation == "curveTo":
                # Cubic bezier: args are (pt1, pt2, pt3) control points
                for i, pt in enumerate(args):
                    current_contour.append({
                        "x": pt[0],
                        "y": pt[1],
                        "type": "curve",
                        "on_curve": (i == len(args) - 1)  # Last point is on-curve
                    })

            elif operation == "qCurveTo":
                # Quadratic bezier: args are control points
                for i, pt in enumerate(args):
                    current_contour.append({
                        "x": pt[0],
                        "y": pt[1],
                        "type": "qcurve",
                        "on_curve": (i == len(args) - 1)
                    })

            elif operation == "closePath" or operation == "endPath":
                if current_contour:
                    contours.append(PathContour(
                        points=current_contour,
                        closed=(operation == "closePath")
                    ))
                current_contour = []

        # Handle unclosed contour
        if current_contour:
            contours.append(PathContour(points=current_contour, closed=False))

        return GlyphOutline(
            contours=[asdict(c) for c in contours],
            bounds=bounds,
            advance_width=advance_width,
            left_side_bearing=lsb
        )

    def render_glyph(
        self,
        font: TTFont,
        glyph_name: str,
        size: int,
        outline: GlyphOutline
    ) -> Optional[Image.Image]:
        """
        Render a glyph to an image.

        Args:
            font: TTFont object
            glyph_name: Name of the glyph
            size: Output image size
            outline: Pre-extracted outline data

        Returns:
            PIL Image or None if rendering failed
        """
        if not outline.bounds:
            return None

        xmin, ymin, xmax, ymax = outline.bounds

        # Calculate scaling
        glyph_width = xmax - xmin
        glyph_height = ymax - ymin

        if glyph_width <= 0 or glyph_height <= 0:
            return None

        # Account for padding
        padding_px = int(size * self.padding)
        available_size = size - (2 * padding_px)

        # Scale to fit
        scale = min(available_size / glyph_width, available_size / glyph_height)

        # Center in image
        offset_x = padding_px + (available_size - glyph_width * scale) / 2 - xmin * scale
        offset_y = padding_px + (available_size - glyph_height * scale) / 2 + ymax * scale

        # Create image
        img = Image.new('L', (size, size), color=255)  # White background
        draw = ImageDraw.Draw(img)

        # Draw each contour
        for contour_data in outline.contours:
            points = contour_data.get("points", [])
            if len(points) < 2:
                continue

            # Convert points to image coordinates
            path = []
            i = 0
            while i < len(points):
                pt = points[i]
                x = pt["x"] * scale + offset_x
                y = -pt["y"] * scale + offset_y  # Flip Y axis

                pt_type = pt.get("type", "line")

                if pt_type in ("move", "line"):
                    path.append((x, y))
                    i += 1

                elif pt_type == "curve":
                    # Cubic bezier - need 3 points (2 control + 1 end)
                    if i + 2 < len(points):
                        # Approximate cubic bezier with line segments
                        p0 = path[-1] if path else (x, y)
                        p1 = (x, y)
                        p2_data = points[i + 1]
                        p2 = (p2_data["x"] * scale + offset_x, -p2_data["y"] * scale + offset_y)
                        p3_data = points[i + 2]
                        p3 = (p3_data["x"] * scale + offset_x, -p3_data["y"] * scale + offset_y)

                        # Sample bezier curve
                        for t in np.linspace(0, 1, 10):
                            bx = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
                            by = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
                            path.append((bx, by))
                        i += 3
                    else:
                        path.append((x, y))
                        i += 1

                elif pt_type == "qcurve":
                    # Quadratic bezier - need 2 points (1 control + 1 end)
                    if i + 1 < len(points):
                        p0 = path[-1] if path else (x, y)
                        p1 = (x, y)
                        p2_data = points[i + 1]
                        p2 = (p2_data["x"] * scale + offset_x, -p2_data["y"] * scale + offset_y)

                        # Sample quadratic bezier
                        for t in np.linspace(0, 1, 10):
                            bx = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
                            by = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
                            path.append((bx, by))
                        i += 2
                    else:
                        path.append((x, y))
                        i += 1
                else:
                    i += 1

            # Draw filled polygon
            if len(path) >= 3:
                try:
                    draw.polygon(path, fill=0)  # Black fill
                except Exception as e:
                    logger.debug(f"Could not draw polygon: {e}")

        return img

    def extract_font(
        self,
        font_path: Path,
        skip_existing: bool = True
    ) -> list[GlyphMetadata]:
        """
        Extract all glyphs from a font file.

        Args:
            font_path: Path to TTF/OTF file
            skip_existing: Skip glyphs that already exist

        Returns:
            List of GlyphMetadata for extracted glyphs
        """
        try:
            font = TTFont(font_path)
        except Exception as e:
            logger.warning(f"Could not open font {font_path}: {e}")
            return []

        # Get font info
        name_table = font.get("name")
        font_family = ""
        font_style = ""

        if name_table:
            for record in name_table.names:
                if record.nameID == 1:  # Family name
                    try:
                        font_family = record.toUnicode()
                    except:
                        pass
                elif record.nameID == 2:  # Style name
                    try:
                        font_style = record.toUnicode()
                    except:
                        pass

        if not font_family:
            font_family = font_path.stem

        # Get metrics
        head = font.get("head")
        os2 = font.get("OS/2")
        hhea = font.get("hhea")

        units_per_em = head.unitsPerEm if head else 1000
        ascender = hhea.ascent if hhea else 800
        descender = hhea.descent if hhea else -200

        # Create font-specific output directory
        font_key = f"{font_family}_{font_style}".lower().replace(" ", "_")
        font_key = "".join(c for c in font_key if c.isalnum() or c == "_")

        # Get character map
        cmap = font.getBestCmap()
        if not cmap:
            logger.warning(f"No character map in {font_path}")
            font.close()
            return []

        extracted = []

        for char in self.charset:
            unicode_val = ord(char)

            if unicode_val not in cmap:
                continue

            glyph_name = cmap[unicode_val]

            # Generate unique identifier
            glyph_id = f"{font_key}_U{unicode_val:04X}"

            # Check if already exists
            outline_path = self.output_dir / "outlines" / f"{glyph_id}.json"
            if skip_existing and outline_path.exists():
                continue

            # Extract outline
            outline = self.extract_outline(font, glyph_name)

            # Create metadata
            metadata = GlyphMetadata(
                character=char,
                unicode=unicode_val,
                glyph_name=glyph_name,
                font_family=font_family,
                font_style=font_style,
                units_per_em=units_per_em,
                ascender=ascender,
                descender=descender,
                outline=asdict(outline),
                image_sizes=self.image_sizes
            )

            # Save outline JSON
            with open(outline_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)

            # Render and save images
            for img_size in self.image_sizes:
                img = self.render_glyph(font, glyph_name, img_size, outline)
                if img:
                    img_path = self.output_dir / f"images_{img_size}" / f"{glyph_id}.png"
                    img.save(img_path)

            extracted.append(metadata)

        font.close()
        return extracted

    def extract_all(
        self,
        input_dir: Path,
        skip_existing: bool = True
    ) -> tuple[int, int]:
        """
        Extract glyphs from all fonts in a directory.

        Args:
            input_dir: Directory containing font files
            skip_existing: Skip already extracted glyphs

        Returns:
            Tuple of (fonts_processed, glyphs_extracted)
        """
        input_dir = Path(input_dir)

        # Find all font files
        font_files = list(input_dir.rglob("*.ttf")) + list(input_dir.rglob("*.otf"))
        font_files = sorted(font_files)

        logger.info(f"Found {len(font_files)} font files")

        fonts_processed = 0
        total_glyphs = 0

        for font_path in tqdm(font_files, desc="Extracting glyphs"):
            try:
                extracted = self.extract_font(font_path, skip_existing=skip_existing)
                if extracted:
                    fonts_processed += 1
                    total_glyphs += len(extracted)
            except Exception as e:
                logger.warning(f"Failed to process {font_path}: {e}")
                continue

        return fonts_processed, total_glyphs

    def get_statistics(self) -> dict:
        """Get statistics about extracted glyphs."""
        stats = {
            "total_outlines": 0,
            "images_by_size": {},
            "fonts": set(),
            "characters": set()
        }

        # Count outline files
        outline_dir = self.output_dir / "outlines"
        if outline_dir.exists():
            outline_files = list(outline_dir.glob("*.json"))
            stats["total_outlines"] = len(outline_files)

            # Sample to get unique fonts and characters
            for f in outline_files[:1000]:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        stats["fonts"].add(data.get("font_family", ""))
                        stats["characters"].add(data.get("character", ""))
                except:
                    pass

        # Count images by size
        for size in self.image_sizes:
            img_dir = self.output_dir / f"images_{size}"
            if img_dir.exists():
                stats["images_by_size"][size] = len(list(img_dir.glob("*.png")))

        stats["fonts"] = len(stats["fonts"])
        stats["characters"] = len(stats["characters"])

        return stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract glyph images and outlines from font files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract from all fonts in a directory
    python extract_glyphs.py --input-dir ./fonts --output-dir ./glyphs

    # Extract with custom image sizes
    python extract_glyphs.py --input-dir ./fonts --output-dir ./glyphs --sizes 64,128,256

    # Use extended character set
    python extract_glyphs.py --input-dir ./fonts --output-dir ./glyphs --extended
        """
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing font files"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save extracted glyphs"
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default="64,128",
        help="Comma-separated image sizes to render (default: 64,128)"
    )

    parser.add_argument(
        "--extended",
        action="store_true",
        help="Use extended character set (includes accented characters)"
    )

    parser.add_argument(
        "--charset",
        type=str,
        default=None,
        help="Custom character set to extract"
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Padding ratio around glyphs (default: 0.1)"
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-extract existing glyphs"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't extract"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse image sizes
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # Determine character set
    if args.charset:
        charset = args.charset
    elif args.extended:
        charset = EXTENDED_CHARSET
    else:
        charset = DEFAULT_CHARSET

    # Create extractor
    extractor = GlyphExtractor(
        output_dir=args.output_dir,
        image_sizes=sizes,
        charset=charset,
        padding=args.padding
    )

    if args.stats_only:
        stats = extractor.get_statistics()
        print("\nExtraction Statistics:")
        print(f"  Total outlines: {stats['total_outlines']}")
        print(f"  Unique fonts: {stats['fonts']}")
        print(f"  Unique characters: {stats['characters']}")
        print("  Images by size:")
        for size, count in stats["images_by_size"].items():
            print(f"    {size}x{size}: {count}")
        return 0

    # Extract glyphs
    try:
        fonts_processed, total_glyphs = extractor.extract_all(
            input_dir=args.input_dir,
            skip_existing=not args.no_skip_existing
        )

        logger.info(f"\nExtraction complete!")
        logger.info(f"Fonts processed: {fonts_processed}")
        logger.info(f"Glyphs extracted: {total_glyphs}")

        # Show final statistics
        stats = extractor.get_statistics()
        logger.info(f"Total glyphs in output: {stats['total_outlines']}")

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
