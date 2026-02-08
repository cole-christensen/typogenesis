"""
Typogenesis CoreML Model Conversion Package.

This package provides tools for converting PyTorch models to CoreML format
for use in the Typogenesis iOS/macOS font creation application.

Conversion Pipeline:
    PyTorch (.pt) -> ONNX (.onnx) -> CoreML (.mlpackage)

Modules:
    config: Configuration and model specifications
    utils: Shared utilities for conversion
    convert_glyph_diffusion: GlyphDiffusion model conversion
    convert_style_encoder: StyleEncoder model conversion
    convert_kerning_net: KerningNet model conversion
    verify_conversion: Numerical accuracy verification

Example Usage:
    # Convert all models
    python -m convert.convert_glyph_diffusion --dummy
    python -m convert.convert_style_encoder --dummy
    python -m convert.convert_kerning_net --dummy

    # Verify conversions
    python -m convert.verify_conversion --model all --dummy

    # Programmatic usage
    from convert.convert_glyph_diffusion import convert_glyph_diffusion
    coreml_path = convert_glyph_diffusion(input_path="model.pt")
"""

from .config import (
    PROJECT_ROOT,
    PYTORCH_MODELS_DIR,
    ONNX_MODELS_DIR,
    COREML_OUTPUT_DIR,
    GLYPH_DIFFUSION_SPEC,
    STYLE_ENCODER_SPEC,
    KERNING_NET_SPEC,
    ALL_MODEL_SPECS,
    ConversionSettings,
    DEFAULT_SETTINGS,
    get_pytorch_path,
    get_onnx_path,
    get_coreml_path,
    ensure_directories,
    get_model_spec,
    list_model_names,
)

__version__ = "1.0.0"
__author__ = "Typogenesis"

__all__ = [
    # Configuration
    "PROJECT_ROOT",
    "PYTORCH_MODELS_DIR",
    "ONNX_MODELS_DIR",
    "COREML_OUTPUT_DIR",
    "GLYPH_DIFFUSION_SPEC",
    "STYLE_ENCODER_SPEC",
    "KERNING_NET_SPEC",
    "ALL_MODEL_SPECS",
    "ConversionSettings",
    "DEFAULT_SETTINGS",
    # Path utilities
    "get_pytorch_path",
    "get_onnx_path",
    "get_coreml_path",
    "ensure_directories",
    "get_model_spec",
    "list_model_names",
]
