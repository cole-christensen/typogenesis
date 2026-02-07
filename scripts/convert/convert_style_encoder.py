#!/usr/bin/env python3
"""
Convert StyleEncoder PyTorch model to CoreML format.

This script handles the conversion of the style encoder model that extracts
128-dimensional style embeddings from glyph images.

The StyleEncoder model:
- Input: 64x64 grayscale image
- Output: 128-dimensional embedding vector

Usage:
    python convert_style_encoder.py [options]

Options:
    --input PATH      Path to PyTorch model (.pt file)
    --output PATH     Path for CoreML output (.mlpackage)
    --opset VERSION   ONNX opset version (default: 17)
    --float16         Convert to float16 precision (default: True)
    --verify          Run verification after conversion
    --verbose         Enable verbose output
    --help            Show this help message
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STYLE_ENCODER_SPEC,
    get_pytorch_path,
    get_onnx_path,
    get_coreml_path,
    ensure_directories,
    DEFAULT_SETTINGS,
)
from utils import (
    check_dependencies,
    load_pytorch_model,
    get_model_info,
    export_to_onnx,
    validate_onnx_model,
    convert_onnx_to_coreml,
    compare_outputs,
    generate_random_input,
    setup_logging,
    print_separator,
    format_size,
    logger,
    ProgressTracker,
)


# =============================================================================
# Model Architecture (Reference Implementation)
# =============================================================================

def create_dummy_style_encoder_model():
    """
    Create a dummy StyleEncoder model for testing the conversion pipeline.

    In production, this would be replaced with the actual trained model.
    This dummy implementation has the correct input/output signatures.
    """
    import torch
    import torch.nn as nn

    class DummyStyleEncoder(nn.Module):
        """
        Dummy style encoder with correct interface for conversion testing.

        Real implementation would include:
        - CNN-based feature extraction
        - Global pooling
        - Dense layers for embedding
        - L2 normalization on output
        """

        def __init__(
            self,
            input_channels: int = 1,
            embedding_dim: int = 128,
        ):
            super().__init__()
            self.embedding_dim = embedding_dim

            # Feature extraction CNN
            self.features = nn.Sequential(
                # 64x64 -> 32x32
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                # 32x32 -> 16x16
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                # 16x16 -> 8x8
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                # 8x8 -> 4x4
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                # Global average pooling
                nn.AdaptiveAvgPool2d(1),
            )

            # Embedding layers
            self.embedding = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, embedding_dim),
            )

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for style encoding.

            Args:
                image: Grayscale glyph image (B, 1, 64, 64), normalized to [0, 1]

            Returns:
                Style embedding (B, 128), L2 normalized
            """
            # Extract features
            features = self.features(image)

            # Compute embedding
            embedding = self.embedding(features)

            # L2 normalize (unit vector)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding

    return DummyStyleEncoder()


# =============================================================================
# Conversion Functions
# =============================================================================

def export_style_encoder_to_onnx(
    model: Any,
    output_path: Path,
    opset_version: int = 17,
    verbose: bool = False,
) -> Path:
    """
    Export StyleEncoder model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path for ONNX output
        opset_version: ONNX opset version
        verbose: Enable verbose output

    Returns:
        Path to exported ONNX model
    """
    spec = STYLE_ENCODER_SPEC

    # Prepare input specifications
    input_specs = [
        {"name": inp.name, "shape": inp.shape, "dtype": inp.dtype}
        for inp in spec.inputs
    ]

    output_names = [out.name for out in spec.outputs]

    # Define dynamic axes for flexible batch size
    dynamic_axes = {
        "image": {0: "batch_size"},
        "embedding": {0: "batch_size"},
    }

    return export_to_onnx(
        model=model,
        output_path=output_path,
        input_specs=input_specs,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )


def convert_style_encoder_to_coreml(
    onnx_path: Path,
    output_path: Path,
    convert_to_float16: bool = True,
) -> Path:
    """
    Convert StyleEncoder ONNX model to CoreML.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path for CoreML output
        convert_to_float16: Use float16 precision

    Returns:
        Path to CoreML model
    """
    spec = STYLE_ENCODER_SPEC

    # Input/output descriptions
    input_descriptions = {
        inp.name: inp.description for inp in spec.inputs
    }
    output_descriptions = {
        out.name: out.description for out in spec.outputs
    }

    return convert_onnx_to_coreml(
        onnx_path=onnx_path,
        output_path=output_path,
        minimum_deployment_target=spec.minimum_deployment_target,
        compute_units=spec.compute_units,
        convert_to_float16=convert_to_float16,
        input_descriptions=input_descriptions,
        output_descriptions=output_descriptions,
        model_description=spec.description,
        author=spec.metadata.get("author", "Typogenesis"),
        version=spec.metadata.get("version", "1.0.0"),
    )


def verify_style_encoder_conversion(
    pytorch_model: Any,
    coreml_path: Path,
    num_samples: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify CoreML conversion by comparing outputs with PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        coreml_path: Path to converted CoreML model
        num_samples: Number of random samples to test
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (all_passed, results_dict)
    """
    import torch
    import coremltools as ct

    logger.info(f"Verifying conversion with {num_samples} samples...")

    # Load CoreML model
    coreml_model = ct.models.MLModel(str(coreml_path))

    results = {
        "samples_tested": num_samples,
        "samples_passed": 0,
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "embedding_norms": [],
        "sample_results": [],
    }

    pytorch_model.eval()

    for i in range(num_samples):
        # Generate random input image
        image = generate_random_input((1, 1, 64, 64), seed=i*100)

        # Run PyTorch model
        with torch.no_grad():
            pt_image = torch.from_numpy(image)
            pt_output = pytorch_model(pt_image)
            pt_output = pt_output.numpy()

        # Check embedding norm (should be ~1.0 for L2 normalized)
        pt_norm = np.linalg.norm(pt_output)
        results["embedding_norms"].append(float(pt_norm))

        # Run CoreML model
        coreml_input = {"image": image}
        coreml_output = coreml_model.predict(coreml_input)
        coreml_output = coreml_output["embedding"]

        # Compare outputs
        is_close, stats = compare_outputs(pt_output, coreml_output, rtol=rtol, atol=atol)

        results["sample_results"].append({
            "sample_idx": i,
            "passed": is_close,
            "pt_norm": float(pt_norm),
            **stats
        })

        if is_close:
            results["samples_passed"] += 1

        results["max_abs_diff"] = max(results["max_abs_diff"], stats["max_abs_diff"])
        results["max_rel_diff"] = max(results["max_rel_diff"], stats["max_rel_diff"])

    all_passed = results["samples_passed"] == num_samples

    # Log results
    avg_norm = np.mean(results["embedding_norms"])
    logger.info(f"Verification results: {results['samples_passed']}/{num_samples} passed")
    logger.info(f"  Max absolute difference: {results['max_abs_diff']:.2e}")
    logger.info(f"  Max relative difference: {results['max_rel_diff']:.2e}")
    logger.info(f"  Average embedding norm: {avg_norm:.4f} (should be ~1.0 for normalized)")

    return all_passed, results


# =============================================================================
# Main Conversion Pipeline
# =============================================================================

def convert_style_encoder(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    opset_version: int = 17,
    convert_to_float16: bool = True,
    verify: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Full conversion pipeline for StyleEncoder model.

    Args:
        input_path: Path to PyTorch model (uses default if None)
        output_path: Path for CoreML output (uses default if None)
        opset_version: ONNX opset version
        convert_to_float16: Use float16 precision
        verify: Run verification after conversion
        verbose: Enable verbose output

    Returns:
        Path to converted CoreML model
    """
    spec = STYLE_ENCODER_SPEC

    # Use default paths if not specified
    if input_path is None:
        input_path = get_pytorch_path(spec)
    if output_path is None:
        output_path = get_coreml_path(spec)

    onnx_path = get_onnx_path(spec)

    # Ensure directories exist
    ensure_directories()

    progress = ProgressTracker(4 if verify else 3, "StyleEncoder")

    print_separator()
    logger.info(f"Converting {spec.display_name}")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    print_separator()

    # Step 1: Load PyTorch model
    progress.update("Loading PyTorch model")

    if input_path.exists():
        model = load_pytorch_model(input_path)
    else:
        logger.warning(f"PyTorch model not found at {input_path}")
        logger.warning("Using dummy model for conversion testing")
        model = create_dummy_style_encoder_model()

    # Log model info
    info = get_model_info(model)
    logger.info(f"  Model type: {info['type']}")
    logger.info(f"  Parameters: {info['total_params']:,}")
    logger.info(f"  Size: {format_size(int(info['size_mb'] * 1024 * 1024))}")

    # Step 2: Export to ONNX
    progress.update("Exporting to ONNX")
    onnx_path = export_style_encoder_to_onnx(
        model=model,
        output_path=onnx_path,
        opset_version=opset_version,
        verbose=verbose,
    )

    # Validate ONNX
    if not validate_onnx_model(onnx_path):
        raise RuntimeError("ONNX model validation failed")

    # Step 3: Convert to CoreML
    progress.update("Converting to CoreML")
    coreml_path = convert_style_encoder_to_coreml(
        onnx_path=onnx_path,
        output_path=output_path,
        convert_to_float16=convert_to_float16,
    )

    # Step 4: Verify conversion (optional)
    if verify:
        progress.update("Verifying conversion")
        passed, results = verify_style_encoder_conversion(
            pytorch_model=model,
            coreml_path=coreml_path,
            rtol=DEFAULT_SETTINGS.verification_rtol,
            atol=DEFAULT_SETTINGS.verification_atol,
        )

        if not passed:
            logger.warning("Verification found numerical differences!")
            logger.warning("The model may still work but outputs differ from PyTorch")

    progress.complete()

    print_separator()
    logger.info(f"Conversion complete: {coreml_path}")
    print_separator()

    return coreml_path


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert StyleEncoder PyTorch model to CoreML format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to PyTorch model (.pt file). Uses default path if not specified.",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path for CoreML output (.mlpackage). Uses default path if not specified.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )

    parser.add_argument(
        "--no-float16",
        action="store_true",
        help="Disable float16 conversion (use float32)",
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after conversion",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy model (for testing conversion pipeline)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Check dependencies
    if not check_dependencies():
        return 1

    try:
        # If dummy flag is set, force using dummy model by setting input to non-existent path
        input_path = args.input
        if args.dummy:
            input_path = Path("/nonexistent/dummy_model.pt")

        convert_style_encoder(
            input_path=input_path,
            output_path=args.output,
            opset_version=args.opset,
            convert_to_float16=not args.no_float16,
            verify=not args.no_verify,
            verbose=args.verbose,
        )
        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
