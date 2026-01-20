"""
Typogenesis Training Scripts.

This package contains the training data pipeline and model training scripts
for the Typogenesis AI font generation system.

Subpackages:
- data: Data preparation scripts (download, extract, augment)
- datasets: PyTorch dataset classes

Target dataset sizes:
- GlyphDiffusion: 500K+ glyph images
- StyleEncoder: 100K+ font-glyph pairs
- KerningNet: 1M+ kerning pair images

Usage:
    # Download fonts
    python -m scripts.data.download_fonts --output-dir ./fonts --max-fonts 100

    # Extract glyphs
    python -m scripts.data.extract_glyphs --input-dir ./fonts --output-dir ./glyphs

    # Generate kerning pairs
    python -m scripts.data.generate_pairs --input-dir ./fonts --output-dir ./kerning

    # Augment data
    python -m scripts.data.augment_data --input-dir ./glyphs/images_64 --output-dir ./augmented
"""

__version__ = "0.1.0"
