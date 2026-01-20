"""
Data preparation scripts for Typogenesis training pipeline.

This package contains scripts for:
- download_fonts: Download fonts from Google Fonts
- extract_glyphs: Extract glyph images and outlines from fonts
- generate_pairs: Generate kerning pair images
- augment_data: Apply data augmentation to glyph images
"""

from pathlib import Path

DATA_SCRIPTS_DIR = Path(__file__).parent

__all__ = [
    "DATA_SCRIPTS_DIR",
]
