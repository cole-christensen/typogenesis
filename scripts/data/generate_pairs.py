#!/usr/bin/env python3
"""
Generate kerning pair images from font files.

This script creates images of glyph pairs (left + right), extracts actual
kerning values from fonts, and focuses on critical pairs (AV, AW, AT, Ta, etc.).

Usage:
    python generate_pairs.py --input-dir ./fonts --output-dir ./kerning_pairs
    python generate_pairs.py --input-dir ./fonts --output-dir ./kerning_pairs --size 128
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from fontTools.ttLib import TTFont
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

# Critical kerning pairs that typically need adjustment
# These are pairs where visual spacing often differs from metric spacing
CRITICAL_PAIRS = [
    # Uppercase combinations
    ("A", "V"), ("A", "W"), ("A", "Y"), ("A", "T"), ("A", "C"),
    ("F", "A"), ("F", "a"), ("F", "o"), ("F", "e"),
    ("L", "T"), ("L", "V"), ("L", "W"), ("L", "Y"),
    ("P", "A"), ("P", "a"), ("P", "o"), ("P", "e"),
    ("T", "A"), ("T", "a"), ("T", "e"), ("T", "i"), ("T", "o"), ("T", "r"), ("T", "u"), ("T", "y"),
    ("V", "A"), ("V", "a"), ("V", "e"), ("V", "i"), ("V", "o"), ("V", "u"),
    ("W", "A"), ("W", "a"), ("W", "e"), ("W", "i"), ("W", "o"), ("W", "u"),
    ("Y", "A"), ("Y", "a"), ("Y", "e"), ("Y", "i"), ("Y", "o"), ("Y", "u"),
    # Lowercase combinations
    ("f", "f"), ("f", "i"), ("f", "l"),
    ("r", "a"), ("r", "e"), ("r", "o"),
    # Mixed case
    ("A", "v"), ("A", "w"), ("A", "y"),
    # Punctuation
    ("T", "."), ("T", ","), ("T", "-"),
    ("V", "."), ("V", ","),
    ("W", "."), ("W", ","),
    ("Y", "."), ("Y", ","),
    ("f", "."), ("f", ","),
    ("r", "."), ("r", ","),
    # Quotes
    ("\"", "A"), ("\"", "a"),
    ("'", "A"), ("'", "a"),
    ("A", "\""), ("A", "'"),
]

# Common pairs for basic coverage
BASIC_PAIRS = [
    (c1, c2)
    for c1 in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for c2 in "AEIOUaeiou"  # Vowels are common second characters
]

# All letter pairs (for comprehensive training data)
ALL_LETTER_PAIRS = [
    (c1, c2)
    for c1 in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for c2 in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
]


@dataclass
class KerningPairData:
    """Data for a kerning pair."""
    left_char: str
    right_char: str
    left_unicode: int
    right_unicode: int
    kerning_value: float
    kerning_normalized: float  # Normalized by units_per_em
    font_family: str
    font_style: str
    units_per_em: int
    pair_id: str
    image_path: Optional[str] = None


class KerningExtractor:
    """Extract kerning pairs and generate pair images."""

    def __init__(
        self,
        output_dir: Path,
        image_size: int = 128,
        pair_set: str = "critical",
        padding: float = 0.1
    ):
        """
        Initialize the kerning extractor.

        Args:
            output_dir: Directory to save kerning pair data
            image_size: Size of pair images (width will be 2x for the pair)
            pair_set: Which pairs to extract ("critical", "basic", "all")
            padding: Padding ratio around glyphs
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.padding = padding

        # Select pair set
        if pair_set == "critical":
            self.pairs_to_extract = CRITICAL_PAIRS
        elif pair_set == "basic":
            self.pairs_to_extract = CRITICAL_PAIRS + BASIC_PAIRS
        else:
            self.pairs_to_extract = ALL_LETTER_PAIRS

        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in self.pairs_to_extract:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)
        self.pairs_to_extract = unique_pairs

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

    def get_kerning_value(
        self,
        font: TTFont,
        left_glyph: str,
        right_glyph: str
    ) -> float:
        """
        Get kerning value for a glyph pair.

        Args:
            font: TTFont object
            left_glyph: Left glyph name
            right_glyph: Right glyph name

        Returns:
            Kerning value in font units (negative = tighter)
        """
        kerning = 0.0

        # Check kern table (older format)
        kern = font.get("kern")
        if kern:
            for table in kern.kernTables:
                if hasattr(table, "kernTable"):
                    pair_key = (left_glyph, right_glyph)
                    if pair_key in table.kernTable:
                        kerning = table.kernTable[pair_key]
                        return kerning

        # Check GPOS table (newer OpenType format)
        gpos = font.get("GPOS")
        if gpos:
            try:
                for feature in gpos.table.FeatureList.FeatureRecord:
                    if feature.FeatureTag == "kern":
                        for lookup_idx in feature.Feature.LookupListIndex:
                            lookup = gpos.table.LookupList.Lookup[lookup_idx]
                            for subtable in lookup.SubTable:
                                # Pair positioning (PairPos)
                                if hasattr(subtable, "PairSet"):
                                    # Format 1: specific pairs
                                    coverage = subtable.Coverage.glyphs
                                    if left_glyph in coverage:
                                        idx = coverage.index(left_glyph)
                                        if idx < len(subtable.PairSet):
                                            pair_set = subtable.PairSet[idx]
                                            for pair in pair_set.PairValueRecord:
                                                if pair.SecondGlyph == right_glyph:
                                                    if hasattr(pair.Value1, "XAdvance"):
                                                        kerning = pair.Value1.XAdvance or 0
                                                    return kerning

                                elif hasattr(subtable, "ClassDef1"):
                                    # Format 2: class-based
                                    class1 = subtable.ClassDef1.classDefs.get(left_glyph, 0)
                                    class2 = subtable.ClassDef2.classDefs.get(right_glyph, 0)

                                    if class1 < len(subtable.Class1Record):
                                        record1 = subtable.Class1Record[class1]
                                        if class2 < len(record1.Class2Record):
                                            record2 = record1.Class2Record[class2]
                                            if hasattr(record2.Value1, "XAdvance"):
                                                kerning = record2.Value1.XAdvance or 0
                                            return kerning
            except Exception as e:
                logger.debug(f"Error reading GPOS: {e}")

        return kerning

    def render_glyph(
        self,
        font: TTFont,
        glyph_name: str,
        size: int
    ) -> tuple[Optional[Image.Image], float, float]:
        """
        Render a single glyph to an image.

        Args:
            font: TTFont object
            glyph_name: Name of glyph
            size: Output image size

        Returns:
            Tuple of (image, advance_width_scaled, left_bearing_scaled)
        """
        glyph_set = font.getGlyphSet()

        if glyph_name not in glyph_set:
            return None, 0, 0

        # Get metrics
        hmtx = font.get("hmtx")
        advance_width = 0
        lsb = 0
        if hmtx and glyph_name in hmtx.metrics:
            advance_width, lsb = hmtx.metrics[glyph_name]

        # Get bounds
        bounds_pen = BoundsPen(glyph_set)
        try:
            glyph_set[glyph_name].draw(bounds_pen)
            bounds = bounds_pen.bounds
        except Exception:
            bounds = None

        if not bounds:
            return None, 0, 0

        xmin, ymin, xmax, ymax = bounds

        # Get font metrics
        head = font.get("head")
        hhea = font.get("hhea")
        units_per_em = head.unitsPerEm if head else 1000
        ascender = hhea.ascent if hhea else 800
        descender = hhea.descent if hhea else -200

        # Calculate scaling based on em height
        em_height = ascender - descender
        padding_px = int(size * self.padding)
        available_size = size - (2 * padding_px)
        scale = available_size / em_height

        # Scaled metrics
        advance_scaled = advance_width * scale
        lsb_scaled = lsb * scale

        # Create image
        img = Image.new('L', (size, size), color=255)
        draw = ImageDraw.Draw(img)

        # Position glyph (baseline at 70% of image height)
        baseline_y = size * 0.7
        offset_x = padding_px + lsb_scaled
        offset_y = baseline_y

        # Draw the glyph using recording pen approach
        from fontTools.pens.recordingPen import RecordingPen
        pen = RecordingPen()
        try:
            glyph_set[glyph_name].draw(pen)
        except Exception:
            return None, advance_scaled, lsb_scaled

        # Convert and draw
        for contour_points in self._get_contours(pen.value, scale, offset_x, offset_y):
            if len(contour_points) >= 3:
                try:
                    draw.polygon(contour_points, fill=0)
                except Exception:
                    pass

        return img, advance_scaled, lsb_scaled

    def _get_contours(
        self,
        pen_value: list,
        scale: float,
        offset_x: float,
        offset_y: float
    ) -> list[list[tuple[float, float]]]:
        """Convert pen recording to list of contour point lists."""
        contours = []
        current = []

        for operation, args in pen_value:
            if operation == "moveTo":
                if current and len(current) >= 3:
                    contours.append(current)
                x, y = args[0]
                current = [(x * scale + offset_x, -y * scale + offset_y)]

            elif operation == "lineTo":
                x, y = args[0]
                current.append((x * scale + offset_x, -y * scale + offset_y))

            elif operation == "curveTo":
                if current:
                    p0 = current[-1]
                    pts = [(a[0] * scale + offset_x, -a[1] * scale + offset_y) for a in args]
                    # Sample cubic bezier
                    for t in np.linspace(0, 1, 8)[1:]:
                        p1, p2, p3 = pts
                        bx = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
                        by = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
                        current.append((bx, by))

            elif operation == "qCurveTo":
                if current:
                    p0 = current[-1]
                    for i in range(len(args)):
                        if i == len(args) - 1:
                            # Last point is on-curve
                            p1 = (args[i][0] * scale + offset_x, -args[i][1] * scale + offset_y)
                            current.append(p1)
                        else:
                            # Implied on-curve point between off-curve points
                            p1 = (args[i][0] * scale + offset_x, -args[i][1] * scale + offset_y)
                            if i + 1 < len(args):
                                p2 = (args[i+1][0] * scale + offset_x, -args[i+1][1] * scale + offset_y)
                                # Sample quadratic bezier
                                for t in np.linspace(0, 1, 6)[1:]:
                                    bx = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
                                    by = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
                                    current.append((bx, by))
                                p0 = p2

            elif operation in ("closePath", "endPath"):
                if current and len(current) >= 3:
                    contours.append(current)
                current = []

        if current and len(current) >= 3:
            contours.append(current)

        return contours

    def render_pair(
        self,
        font: TTFont,
        left_char: str,
        right_char: str,
        kerning: float
    ) -> Optional[Image.Image]:
        """
        Render a kerning pair to an image.

        Args:
            font: TTFont object
            left_char: Left character
            right_char: Right character
            kerning: Kerning value to apply

        Returns:
            PIL Image of the pair
        """
        cmap = font.getBestCmap()
        if not cmap:
            return None

        left_unicode = ord(left_char)
        right_unicode = ord(right_char)

        if left_unicode not in cmap or right_unicode not in cmap:
            return None

        left_glyph = cmap[left_unicode]
        right_glyph = cmap[right_unicode]

        # Render individual glyphs
        left_img, left_advance, _ = self.render_glyph(font, left_glyph, self.image_size)
        right_img, right_advance, right_lsb = self.render_glyph(font, right_glyph, self.image_size)

        if left_img is None or right_img is None:
            return None

        # Get font metrics for scaling kerning
        head = font.get("head")
        hhea = font.get("hhea")
        units_per_em = head.unitsPerEm if head else 1000
        ascender = hhea.ascent if hhea else 800
        descender = hhea.descent if hhea else -200

        em_height = ascender - descender
        padding_px = int(self.image_size * self.padding)
        available_size = self.image_size - (2 * padding_px)
        scale = available_size / em_height

        # Scale kerning
        kerning_scaled = kerning * scale

        # Create pair image (wider to fit both glyphs)
        pair_width = int(self.image_size * 2)
        pair_img = Image.new('L', (pair_width, self.image_size), color=255)

        # Place left glyph
        pair_img.paste(left_img, (0, 0))

        # Place right glyph with kerning adjustment
        right_offset = int(left_advance + kerning_scaled)

        # Ensure right glyph fits
        if right_offset < pair_width - self.image_size:
            # Composite right glyph
            right_array = np.array(right_img)
            pair_array = np.array(pair_img)

            for y in range(self.image_size):
                for x in range(self.image_size):
                    dest_x = right_offset + x
                    if 0 <= dest_x < pair_width:
                        # Multiply (both are grayscale, darker = more ink)
                        pair_array[y, dest_x] = min(
                            pair_array[y, dest_x],
                            right_array[y, x]
                        )

            pair_img = Image.fromarray(pair_array)

        return pair_img

    def extract_font(
        self,
        font_path: Path,
        skip_existing: bool = True
    ) -> list[KerningPairData]:
        """
        Extract kerning pairs from a font.

        Args:
            font_path: Path to font file
            skip_existing: Skip pairs that already exist

        Returns:
            List of KerningPairData
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
                if record.nameID == 1:
                    try:
                        font_family = record.toUnicode()
                    except:
                        pass
                elif record.nameID == 2:
                    try:
                        font_style = record.toUnicode()
                    except:
                        pass

        if not font_family:
            font_family = font_path.stem

        # Get units per em
        head = font.get("head")
        units_per_em = head.unitsPerEm if head else 1000

        # Get cmap
        cmap = font.getBestCmap()
        if not cmap:
            font.close()
            return []

        font_key = f"{font_family}_{font_style}".lower().replace(" ", "_")
        font_key = "".join(c for c in font_key if c.isalnum() or c == "_")

        extracted = []

        for left_char, right_char in self.pairs_to_extract:
            left_unicode = ord(left_char)
            right_unicode = ord(right_char)

            if left_unicode not in cmap or right_unicode not in cmap:
                continue

            left_glyph = cmap[left_unicode]
            right_glyph = cmap[right_unicode]

            # Generate pair ID
            pair_id = f"{font_key}_{left_unicode:04X}_{right_unicode:04X}"

            # Check if exists
            metadata_path = self.output_dir / "metadata" / f"{pair_id}.json"
            if skip_existing and metadata_path.exists():
                continue

            # Get kerning value
            kerning_value = self.get_kerning_value(font, left_glyph, right_glyph)
            kerning_normalized = kerning_value / units_per_em if units_per_em > 0 else 0

            # Render pair image
            pair_img = self.render_pair(font, left_char, right_char, kerning_value)

            image_path = None
            if pair_img:
                image_path = f"images/{pair_id}.png"
                pair_img.save(self.output_dir / image_path)

            # Create pair data
            pair_data = KerningPairData(
                left_char=left_char,
                right_char=right_char,
                left_unicode=left_unicode,
                right_unicode=right_unicode,
                kerning_value=kerning_value,
                kerning_normalized=kerning_normalized,
                font_family=font_family,
                font_style=font_style,
                units_per_em=units_per_em,
                pair_id=pair_id,
                image_path=image_path
            )

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(asdict(pair_data), f, indent=2)

            extracted.append(pair_data)

        font.close()
        return extracted

    def extract_all(
        self,
        input_dir: Path,
        skip_existing: bool = True
    ) -> tuple[int, int]:
        """
        Extract kerning pairs from all fonts.

        Args:
            input_dir: Directory containing fonts
            skip_existing: Skip existing pairs

        Returns:
            Tuple of (fonts_processed, pairs_extracted)
        """
        input_dir = Path(input_dir)
        font_files = list(input_dir.rglob("*.ttf")) + list(input_dir.rglob("*.otf"))
        font_files = sorted(font_files)

        logger.info(f"Found {len(font_files)} font files")
        logger.info(f"Extracting {len(self.pairs_to_extract)} pair types per font")

        fonts_processed = 0
        total_pairs = 0

        for font_path in tqdm(font_files, desc="Extracting kerning pairs"):
            try:
                pairs = self.extract_font(font_path, skip_existing=skip_existing)
                if pairs:
                    fonts_processed += 1
                    total_pairs += len(pairs)
            except Exception as e:
                logger.warning(f"Failed to process {font_path}: {e}")

        return fonts_processed, total_pairs

    def get_statistics(self) -> dict:
        """Get statistics about extracted pairs."""
        metadata_dir = self.output_dir / "metadata"
        images_dir = self.output_dir / "images"

        stats = {
            "total_pairs": 0,
            "total_images": 0,
            "pairs_with_kerning": 0,
            "avg_kerning": 0,
            "fonts": set()
        }

        if not metadata_dir.exists():
            return stats

        metadata_files = list(metadata_dir.glob("*.json"))
        stats["total_pairs"] = len(metadata_files)

        kerning_values = []
        for f in metadata_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    stats["fonts"].add(data.get("font_family", ""))
                    kv = data.get("kerning_value", 0)
                    if kv != 0:
                        stats["pairs_with_kerning"] += 1
                        kerning_values.append(kv)
            except:
                pass

        if kerning_values:
            stats["avg_kerning"] = sum(kerning_values) / len(kerning_values)

        if images_dir.exists():
            stats["total_images"] = len(list(images_dir.glob("*.png")))

        stats["fonts"] = len(stats["fonts"])

        return stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate kerning pair images from font files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract critical kerning pairs
    python generate_pairs.py --input-dir ./fonts --output-dir ./kerning

    # Extract all letter pairs (comprehensive)
    python generate_pairs.py --input-dir ./fonts --output-dir ./kerning --pairs all

    # Custom image size
    python generate_pairs.py --input-dir ./fonts --output-dir ./kerning --size 256
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
        help="Directory to save kerning pair data"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Image size for single glyphs (pair width is 2x)"
    )

    parser.add_argument(
        "--pairs",
        type=str,
        choices=["critical", "basic", "all"],
        default="critical",
        help="Which pairs to extract (default: critical)"
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
        help="Re-extract existing pairs"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics"
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

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    extractor = KerningExtractor(
        output_dir=args.output_dir,
        image_size=args.size,
        pair_set=args.pairs,
        padding=args.padding
    )

    if args.stats_only:
        stats = extractor.get_statistics()
        print("\nKerning Pair Statistics:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Pairs with kerning: {stats['pairs_with_kerning']}")
        print(f"  Average kerning: {stats['avg_kerning']:.2f}")
        print(f"  Unique fonts: {stats['fonts']}")
        return 0

    try:
        fonts_processed, total_pairs = extractor.extract_all(
            input_dir=args.input_dir,
            skip_existing=not args.no_skip_existing
        )

        logger.info(f"\nExtraction complete!")
        logger.info(f"Fonts processed: {fonts_processed}")
        logger.info(f"Pairs extracted: {total_pairs}")

        stats = extractor.get_statistics()
        logger.info(f"Total pairs in output: {stats['total_pairs']}")
        logger.info(f"Pairs with non-zero kerning: {stats['pairs_with_kerning']}")

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
