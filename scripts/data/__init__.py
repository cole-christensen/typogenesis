"""Data pipeline for Typogenesis ML training.

Modules:
    download_fonts: Google Fonts downloader
    extract_glyphs: Font → glyph images + outlines
    prepare_datasets: Build train/val/test splits
    augmentations: Data augmentation transforms
"""

from .augmentations import get_eval_transforms, get_train_transforms
from .extract_glyphs import extract_all_fonts, extract_font
from .prepare_datasets import prepare_datasets

__all__ = [
    "get_train_transforms",
    "get_eval_transforms",
    "extract_font",
    "extract_all_fonts",
    "prepare_datasets",
]
