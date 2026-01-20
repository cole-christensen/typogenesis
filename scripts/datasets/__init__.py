"""
PyTorch Datasets for Typogenesis model training.

This package contains dataset classes for:
- GlyphDataset: Individual glyph images with character conditioning
- StylePairDataset: Pairs of glyphs for contrastive style learning
- StyleTripletDataset: Triplets for triplet loss training
- KerningDataset: Glyph pairs with kerning values
"""

from .glyph_dataset import (
    GlyphDataset,
    GlyphConditionedDataset,
    create_dataloaders,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    NUM_CHARS,
    char_to_onehot,
    char_to_embedding,
)

from .style_dataset import (
    StylePairDataset,
    StyleTripletDataset,
    create_style_dataloaders,
)

from .kerning_dataset import (
    KerningDataset,
    KerningRegressionDataset,
    KerningClassificationDataset,
    create_kerning_dataloaders,
)

__all__ = [
    # Glyph datasets
    "GlyphDataset",
    "GlyphConditionedDataset",
    "create_dataloaders",
    "CHAR_TO_IDX",
    "IDX_TO_CHAR",
    "NUM_CHARS",
    "char_to_onehot",
    "char_to_embedding",
    # Style datasets
    "StylePairDataset",
    "StyleTripletDataset",
    "create_style_dataloaders",
    # Kerning datasets
    "KerningDataset",
    "KerningRegressionDataset",
    "KerningClassificationDataset",
    "create_kerning_dataloaders",
]
