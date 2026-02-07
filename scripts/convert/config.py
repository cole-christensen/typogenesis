#!/usr/bin/env python3
"""
Configuration for CoreML model conversion pipeline.

This module defines paths, settings, and model specifications for converting
PyTorch models to CoreML format for use in the Typogenesis iOS/macOS app.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os


# =============================================================================
# Directory Paths
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory (Typogenesis)."""
    # This file is at: scripts/convert/config.py
    # Project root is: ../../
    return Path(__file__).parent.parent.parent.resolve()


# Default paths (can be overridden via environment variables or CLI args)
PROJECT_ROOT = get_project_root()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONVERT_DIR = SCRIPTS_DIR / "convert"

# Input model paths (PyTorch .pt files)
PYTORCH_MODELS_DIR = Path(os.environ.get(
    "TYPOGENESIS_PYTORCH_MODELS",
    SCRIPTS_DIR / "models" / "pytorch"
))

# Intermediate ONNX models
ONNX_MODELS_DIR = Path(os.environ.get(
    "TYPOGENESIS_ONNX_MODELS",
    SCRIPTS_DIR / "models" / "onnx"
))

# Output CoreML models (.mlpackage)
COREML_OUTPUT_DIR = Path(os.environ.get(
    "TYPOGENESIS_COREML_OUTPUT",
    PROJECT_ROOT / "Typogenesis" / "Resources" / "Models"
))


# =============================================================================
# Model Specifications
# =============================================================================

@dataclass
class ModelInputSpec:
    """Specification for a model input tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    description: str = ""
    # For flexible shapes, specify ranges as (min, max) tuples
    # Use -1 to indicate dynamic dimension
    flexible_shape: Optional[Tuple[Tuple[int, int], ...]] = None


@dataclass
class ModelOutputSpec:
    """Specification for a model output tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    description: str = ""


@dataclass
class ModelSpec:
    """Complete specification for a model to be converted."""
    name: str
    display_name: str
    description: str
    pytorch_filename: str
    onnx_filename: str
    coreml_filename: str
    inputs: List[ModelInputSpec]
    outputs: List[ModelOutputSpec]
    opset_version: int = 17  # ONNX opset version
    # CoreML-specific settings
    compute_units: str = "ALL"  # ALL, CPU_AND_NE, CPU_ONLY
    minimum_deployment_target: str = "macOS14"
    # Conversion settings
    convert_to_float16: bool = True  # Use fp16 for smaller models
    # Model-specific metadata
    metadata: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Model Definitions
# =============================================================================

GLYPH_DIFFUSION_SPEC = ModelSpec(
    name="GlyphDiffusion",
    display_name="Glyph Generator",
    description="Diffusion-based glyph generation model. Generates glyph outlines "
                "conditioned on character class and style embedding.",
    pytorch_filename="glyph_diffusion.pt",
    onnx_filename="glyph_diffusion.onnx",
    coreml_filename="GlyphDiffusion.mlpackage",
    inputs=[
        ModelInputSpec(
            name="noise",
            shape=(1, 4, 64, 64),  # Batch, channels, height, width
            description="Latent noise tensor for diffusion process"
        ),
        ModelInputSpec(
            name="character_embedding",
            shape=(1, 128),
            description="One-hot or learned embedding for target character"
        ),
        ModelInputSpec(
            name="style_embedding",
            shape=(1, 128),
            description="Style vector from StyleEncoder"
        ),
        ModelInputSpec(
            name="timestep",
            shape=(1,),
            dtype="int64",
            description="Current diffusion timestep (0 to num_steps-1)"
        ),
    ],
    outputs=[
        ModelOutputSpec(
            name="denoised",
            shape=(1, 4, 64, 64),
            description="Denoised latent representation"
        ),
    ],
    opset_version=17,
    compute_units="ALL",  # Use GPU/ANE when available
    convert_to_float16=True,
    metadata={
        "author": "Typogenesis",
        "version": "1.0.0",
        "diffusion_steps": "50",
        "guidance_scale": "7.5",
    }
)

STYLE_ENCODER_SPEC = ModelSpec(
    name="StyleEncoder",
    display_name="Style Encoder",
    description="Extracts 128-dimensional style embedding from glyph images. "
                "Input is a 64x64 grayscale image, output is a normalized embedding vector.",
    pytorch_filename="style_encoder.pt",
    onnx_filename="style_encoder.onnx",
    coreml_filename="StyleEncoder.mlpackage",
    inputs=[
        ModelInputSpec(
            name="image",
            shape=(1, 1, 64, 64),  # Batch, channels (grayscale), height, width
            description="64x64 grayscale glyph image, normalized to [0, 1]"
        ),
    ],
    outputs=[
        ModelOutputSpec(
            name="embedding",
            shape=(1, 128),
            description="128-dimensional style embedding vector"
        ),
    ],
    opset_version=17,
    compute_units="CPU_AND_NE",  # Optimized for Neural Engine
    convert_to_float16=True,
    metadata={
        "author": "Typogenesis",
        "version": "1.0.0",
        "embedding_dim": "128",
    }
)

KERNING_NET_SPEC = ModelSpec(
    name="KerningNet",
    display_name="Kerning Predictor",
    description="Predicts optimal kerning value for a glyph pair. "
                "Input is two 64x64 grayscale glyph images, output is kerning in font units.",
    pytorch_filename="kerning_net.pt",
    onnx_filename="kerning_net.onnx",
    coreml_filename="KerningNet.mlpackage",
    inputs=[
        ModelInputSpec(
            name="left_glyph",
            shape=(1, 1, 64, 64),  # Batch, channels (grayscale), height, width
            description="64x64 grayscale image of left glyph"
        ),
        ModelInputSpec(
            name="right_glyph",
            shape=(1, 1, 64, 64),  # Batch, channels (grayscale), height, width
            description="64x64 grayscale image of right glyph"
        ),
    ],
    outputs=[
        ModelOutputSpec(
            name="kerning",
            shape=(1, 1),
            description="Predicted kerning value in font units"
        ),
    ],
    opset_version=17,
    compute_units="CPU_AND_NE",  # Optimized for Neural Engine
    convert_to_float16=True,
    metadata={
        "author": "Typogenesis",
        "version": "1.0.0",
    }
)

# All model specifications
ALL_MODEL_SPECS: Dict[str, ModelSpec] = {
    "glyph_diffusion": GLYPH_DIFFUSION_SPEC,
    "style_encoder": STYLE_ENCODER_SPEC,
    "kerning_net": KERNING_NET_SPEC,
}


# =============================================================================
# Conversion Settings
# =============================================================================

@dataclass
class ConversionSettings:
    """Global settings for the conversion pipeline."""
    # ONNX export settings
    onnx_opset_version: int = 17
    onnx_export_params: bool = True
    onnx_do_constant_folding: bool = True

    # CoreML settings
    coreml_minimum_deployment_target: str = "macOS14"
    coreml_compute_precision: str = "FLOAT16"  # FLOAT32 or FLOAT16
    coreml_convert_to_mlprogram: bool = True  # Use ML Program format for better optimization

    # Verification settings
    verification_rtol: float = 1e-4  # Relative tolerance for numerical comparison
    verification_atol: float = 1e-4  # Absolute tolerance for numerical comparison
    verification_samples: int = 10   # Number of random samples to test

    # Logging
    verbose: bool = True
    log_file: Optional[Path] = None


DEFAULT_SETTINGS = ConversionSettings()


# =============================================================================
# Helper Functions
# =============================================================================

def get_pytorch_path(spec: ModelSpec) -> Path:
    """Get the full path to a PyTorch model file."""
    return PYTORCH_MODELS_DIR / spec.pytorch_filename


def get_onnx_path(spec: ModelSpec) -> Path:
    """Get the full path to an ONNX model file."""
    return ONNX_MODELS_DIR / spec.onnx_filename


def get_coreml_path(spec: ModelSpec) -> Path:
    """Get the full path to a CoreML model file."""
    return COREML_OUTPUT_DIR / spec.coreml_filename


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    PYTORCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    COREML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_model_spec(model_name: str) -> Optional[ModelSpec]:
    """Get a model specification by name (case-insensitive)."""
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    return ALL_MODEL_SPECS.get(key)


def list_model_names() -> List[str]:
    """Get list of all available model names."""
    return list(ALL_MODEL_SPECS.keys())


# =============================================================================
# CLI-friendly output
# =============================================================================

if __name__ == "__main__":
    print("Typogenesis CoreML Conversion Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"PyTorch Models: {PYTORCH_MODELS_DIR}")
    print(f"ONNX Models: {ONNX_MODELS_DIR}")
    print(f"CoreML Output: {COREML_OUTPUT_DIR}")
    print(f"\nAvailable Models:")
    for name, spec in ALL_MODEL_SPECS.items():
        print(f"  - {name}: {spec.display_name}")
        print(f"    Inputs: {[i.name for i in spec.inputs]}")
        print(f"    Outputs: {[o.name for o in spec.outputs]}")
