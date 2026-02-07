#!/usr/bin/env python3
"""
Convert GlyphDiffusion PyTorch model to CoreML format.

This script handles the conversion of the diffusion-based glyph generation model
from PyTorch to CoreML for use in the Typogenesis iOS/macOS application.

The GlyphDiffusion model has special requirements:
- Multiple inference steps (diffusion denoising loop)
- Multiple conditioning inputs (character, style, timestep)
- Flexible handling of guidance scale

Usage:
    python convert_glyph_diffusion.py [options]

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
    GLYPH_DIFFUSION_SPEC,
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

def create_dummy_glyph_diffusion_model():
    """
    Create a dummy GlyphDiffusion model for testing the conversion pipeline.

    In production, this would be replaced with the actual trained model.
    This dummy implementation has the correct input/output signatures.
    """
    import torch
    import torch.nn as nn

    class DummyGlyphDiffusion(nn.Module):
        """
        Dummy diffusion model with correct interface for conversion testing.

        Real implementation would include:
        - UNet-style architecture for noise prediction
        - Timestep embedding
        - Character and style conditioning
        - Attention mechanisms for high-quality generation
        """

        def __init__(
            self,
            latent_channels: int = 4,
            latent_size: int = 64,
            character_embedding_dim: int = 128,
            style_embedding_dim: int = 128,
            hidden_dim: int = 256,
        ):
            super().__init__()
            self.latent_channels = latent_channels
            self.latent_size = latent_size

            # Timestep embedding
            self.time_embed = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # Character conditioning
            self.char_embed = nn.Sequential(
                nn.Linear(character_embedding_dim, hidden_dim),
                nn.SiLU(),
            )

            # Style conditioning
            self.style_embed = nn.Sequential(
                nn.Linear(style_embedding_dim, hidden_dim),
                nn.SiLU(),
            )

            # Main denoising network (simplified)
            self.conv_in = nn.Conv2d(latent_channels, hidden_dim, 3, padding=1)

            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(4)
            ])

            self.conv_out = nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)

        def forward(
            self,
            noise: torch.Tensor,
            character_embedding: torch.Tensor,
            style_embedding: torch.Tensor,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass for noise prediction.

            Args:
                noise: Noisy latent (B, C, H, W)
                character_embedding: Character conditioning (B, 128)
                style_embedding: Style conditioning (B, 128)
                timestep: Current timestep (B,)

            Returns:
                Predicted noise (B, C, H, W)
            """
            batch_size = noise.shape[0]

            # Embed timestep (convert to float)
            t_emb = self.time_embed(timestep.float().unsqueeze(-1))  # (B, hidden_dim)

            # Embed conditioning
            c_emb = self.char_embed(character_embedding)  # (B, hidden_dim)
            s_emb = self.style_embed(style_embedding)  # (B, hidden_dim)

            # Combined conditioning
            cond = t_emb + c_emb + s_emb  # (B, hidden_dim)

            # Initial convolution
            h = self.conv_in(noise)  # (B, hidden_dim, H, W)

            # Add conditioning (broadcast across spatial dims)
            h = h + cond.unsqueeze(-1).unsqueeze(-1)

            # Process through blocks
            for block in self.blocks:
                h = block(h) + h  # Residual connection

            # Output convolution
            out = self.conv_out(h)

            return out

    return DummyGlyphDiffusion()


# =============================================================================
# Conversion Functions
# =============================================================================

def export_glyph_diffusion_to_onnx(
    model: Any,
    output_path: Path,
    opset_version: int = 17,
    verbose: bool = False,
) -> Path:
    """
    Export GlyphDiffusion model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path for ONNX output
        opset_version: ONNX opset version
        verbose: Enable verbose output

    Returns:
        Path to exported ONNX model
    """
    import torch

    spec = GLYPH_DIFFUSION_SPEC

    # Prepare input specifications
    input_specs = [
        {"name": inp.name, "shape": inp.shape, "dtype": inp.dtype}
        for inp in spec.inputs
    ]

    output_names = [out.name for out in spec.outputs]

    # Define dynamic axes for flexible batch size
    dynamic_axes = {
        "noise": {0: "batch_size"},
        "character_embedding": {0: "batch_size"},
        "style_embedding": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "denoised": {0: "batch_size"},
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


def convert_glyph_diffusion_to_coreml(
    onnx_path: Path,
    output_path: Path,
    convert_to_float16: bool = True,
) -> Path:
    """
    Convert GlyphDiffusion ONNX model to CoreML.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path for CoreML output
        convert_to_float16: Use float16 precision

    Returns:
        Path to CoreML model
    """
    spec = GLYPH_DIFFUSION_SPEC

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


def verify_glyph_diffusion_conversion(
    pytorch_model: Any,
    coreml_path: Path,
    num_samples: int = 5,
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

    spec = GLYPH_DIFFUSION_SPEC
    results = {
        "samples_tested": num_samples,
        "samples_passed": 0,
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "sample_results": [],
    }

    pytorch_model.eval()

    for i in range(num_samples):
        # Generate random inputs
        noise = generate_random_input((1, 4, 64, 64), seed=i*100)
        char_emb = generate_random_input((1, 128), seed=i*100+1)
        style_emb = generate_random_input((1, 128), seed=i*100+2)
        timestep = np.array([i % 50], dtype=np.int64)

        # Run PyTorch model
        with torch.no_grad():
            pt_noise = torch.from_numpy(noise)
            pt_char = torch.from_numpy(char_emb)
            pt_style = torch.from_numpy(style_emb)
            pt_time = torch.from_numpy(timestep)

            pt_output = pytorch_model(pt_noise, pt_char, pt_style, pt_time)
            pt_output = pt_output.numpy()

        # Run CoreML model
        coreml_input = {
            "noise": noise,
            "character_embedding": char_emb,
            "style_embedding": style_emb,
            "timestep": timestep,
        }
        coreml_output = coreml_model.predict(coreml_input)
        coreml_output = coreml_output["denoised"]

        # Compare outputs
        is_close, stats = compare_outputs(pt_output, coreml_output, rtol=rtol, atol=atol)

        results["sample_results"].append({
            "sample_idx": i,
            "passed": is_close,
            **stats
        })

        if is_close:
            results["samples_passed"] += 1

        results["max_abs_diff"] = max(results["max_abs_diff"], stats["max_abs_diff"])
        results["max_rel_diff"] = max(results["max_rel_diff"], stats["max_rel_diff"])

    all_passed = results["samples_passed"] == num_samples

    logger.info(f"Verification results: {results['samples_passed']}/{num_samples} passed")
    logger.info(f"  Max absolute difference: {results['max_abs_diff']:.2e}")
    logger.info(f"  Max relative difference: {results['max_rel_diff']:.2e}")

    return all_passed, results


# =============================================================================
# Main Conversion Pipeline
# =============================================================================

def convert_glyph_diffusion(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    opset_version: int = 17,
    convert_to_float16: bool = True,
    verify: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Full conversion pipeline for GlyphDiffusion model.

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
    spec = GLYPH_DIFFUSION_SPEC

    # Use default paths if not specified
    if input_path is None:
        input_path = get_pytorch_path(spec)
    if output_path is None:
        output_path = get_coreml_path(spec)

    onnx_path = get_onnx_path(spec)

    # Ensure directories exist
    ensure_directories()

    progress = ProgressTracker(4 if verify else 3, "GlyphDiffusion")

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
        model = create_dummy_glyph_diffusion_model()

    # Log model info
    info = get_model_info(model)
    logger.info(f"  Model type: {info['type']}")
    logger.info(f"  Parameters: {info['total_params']:,}")
    logger.info(f"  Size: {format_size(int(info['size_mb'] * 1024 * 1024))}")

    # Step 2: Export to ONNX
    progress.update("Exporting to ONNX")
    onnx_path = export_glyph_diffusion_to_onnx(
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
    coreml_path = convert_glyph_diffusion_to_coreml(
        onnx_path=onnx_path,
        output_path=output_path,
        convert_to_float16=convert_to_float16,
    )

    # Step 4: Verify conversion (optional)
    if verify:
        progress.update("Verifying conversion")
        passed, results = verify_glyph_diffusion_conversion(
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
        description="Convert GlyphDiffusion PyTorch model to CoreML format.",
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

        convert_glyph_diffusion(
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
