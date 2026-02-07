#!/usr/bin/env python3
"""
Verify CoreML model conversions by comparing outputs with PyTorch models.

This script performs comprehensive numerical accuracy tests to ensure that
converted CoreML models produce outputs that match the original PyTorch
models within acceptable tolerances.

Usage:
    python verify_conversion.py [options]

Options:
    --model NAME      Model to verify (glyph_diffusion, style_encoder, kerning_net, or 'all')
    --samples N       Number of test samples (default: 10)
    --rtol FLOAT      Relative tolerance (default: 1e-4)
    --atol FLOAT      Absolute tolerance (default: 1e-4)
    --output PATH     Path to write JSON report
    --verbose         Enable verbose output
    --help            Show this help message
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALL_MODEL_SPECS,
    GLYPH_DIFFUSION_SPEC,
    STYLE_ENCODER_SPEC,
    KERNING_NET_SPEC,
    get_pytorch_path,
    get_coreml_path,
    list_model_names,
)
from utils import (
    check_dependencies,
    load_pytorch_model,
    load_coreml_model,
    compare_outputs,
    generate_random_input,
    setup_logging,
    print_separator,
    logger,
)


# =============================================================================
# Verification Functions
# =============================================================================

def verify_model(
    model_name: str,
    pytorch_model: Any,
    coreml_model: Any,
    num_samples: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Verify a single model conversion by comparing outputs.

    Args:
        model_name: Name of the model being verified
        pytorch_model: Loaded PyTorch model
        coreml_model: Loaded CoreML model
        num_samples: Number of test samples
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with verification results
    """
    import torch

    logger.info(f"Verifying {model_name} with {num_samples} samples...")

    results = {
        "model_name": model_name,
        "samples_tested": num_samples,
        "samples_passed": 0,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "mean_rel_diff": 0.0,
        "rtol": rtol,
        "atol": atol,
        "sample_results": [],
    }

    pytorch_model.eval()
    abs_diffs = []
    rel_diffs = []

    for i in range(num_samples):
        try:
            # Generate inputs and run both models based on model type
            if model_name == "glyph_diffusion":
                pt_output, coreml_output = _run_glyph_diffusion_sample(
                    pytorch_model, coreml_model, i
                )
            elif model_name == "style_encoder":
                pt_output, coreml_output = _run_style_encoder_sample(
                    pytorch_model, coreml_model, i
                )
            elif model_name == "kerning_net":
                pt_output, coreml_output = _run_kerning_net_sample(
                    pytorch_model, coreml_model, i
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Compare outputs
            is_close, stats = compare_outputs(pt_output, coreml_output, rtol=rtol, atol=atol)

            sample_result = {
                "sample_idx": i,
                "passed": is_close,
                **stats
            }
            results["sample_results"].append(sample_result)

            if is_close:
                results["samples_passed"] += 1

            abs_diffs.append(stats["mean_abs_diff"])
            rel_diffs.append(stats["mean_rel_diff"])
            results["max_abs_diff"] = max(results["max_abs_diff"], stats["max_abs_diff"])
            results["max_rel_diff"] = max(results["max_rel_diff"], stats["max_rel_diff"])

        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            results["sample_results"].append({
                "sample_idx": i,
                "passed": False,
                "error": str(e)
            })

    # Compute mean differences
    if abs_diffs:
        results["mean_abs_diff"] = float(np.mean(abs_diffs))
        results["mean_rel_diff"] = float(np.mean(rel_diffs))

    # Overall pass/fail
    results["all_passed"] = results["samples_passed"] == num_samples
    results["pass_rate"] = results["samples_passed"] / max(num_samples, 1)

    return results


def _run_glyph_diffusion_sample(
    pytorch_model: Any,
    coreml_model: Any,
    sample_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single sample through GlyphDiffusion models."""
    import torch

    # Generate inputs
    noise = generate_random_input((1, 4, 64, 64), seed=sample_idx*100)
    char_emb = generate_random_input((1, 128), seed=sample_idx*100+1)
    style_emb = generate_random_input((1, 128), seed=sample_idx*100+2)
    timestep = np.array([sample_idx % 50], dtype=np.int64)

    # PyTorch
    with torch.no_grad():
        pt_output = pytorch_model(
            torch.from_numpy(noise),
            torch.from_numpy(char_emb),
            torch.from_numpy(style_emb),
            torch.from_numpy(timestep),
        )
        pt_output = pt_output.numpy()

    # CoreML
    coreml_input = {
        "noise": noise,
        "character_embedding": char_emb,
        "style_embedding": style_emb,
        "timestep": timestep,
    }
    coreml_output = coreml_model.predict(coreml_input)
    coreml_output = np.array(coreml_output["denoised"])

    return pt_output, coreml_output


def _run_style_encoder_sample(
    pytorch_model: Any,
    coreml_model: Any,
    sample_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single sample through StyleEncoder models."""
    import torch

    # Generate input
    image = generate_random_input((1, 1, 64, 64), seed=sample_idx*100)

    # PyTorch
    with torch.no_grad():
        pt_output = pytorch_model(torch.from_numpy(image))
        pt_output = pt_output.numpy()

    # CoreML
    coreml_input = {"image": image}
    coreml_output = coreml_model.predict(coreml_input)
    coreml_output = np.array(coreml_output["embedding"])

    return pt_output, coreml_output


def _run_kerning_net_sample(
    pytorch_model: Any,
    coreml_model: Any,
    sample_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single sample through KerningNet models."""
    import torch

    # Generate inputs
    left_glyph = generate_random_input((1, 1, 64, 64), seed=sample_idx*100)
    right_glyph = generate_random_input((1, 1, 64, 64), seed=sample_idx*100+1)

    # PyTorch
    with torch.no_grad():
        pt_output = pytorch_model(
            torch.from_numpy(left_glyph),
            torch.from_numpy(right_glyph),
        )
        pt_output = pt_output.numpy()

    # CoreML
    coreml_input = {
        "left_glyph": left_glyph,
        "right_glyph": right_glyph,
    }
    coreml_output = coreml_model.predict(coreml_input)
    coreml_output = np.array(coreml_output["kerning"])

    return pt_output, coreml_output


# =============================================================================
# Model Loading with Dummy Fallback
# =============================================================================

def load_models_for_verification(
    model_name: str,
    use_dummy_pytorch: bool = False,
) -> Tuple[Any, Any]:
    """
    Load PyTorch and CoreML models for verification.

    Args:
        model_name: Name of the model
        use_dummy_pytorch: Use dummy PyTorch model if original not found

    Returns:
        Tuple of (pytorch_model, coreml_model)
    """
    from convert_glyph_diffusion import create_dummy_glyph_diffusion_model
    from convert_style_encoder import create_dummy_style_encoder_model
    from convert_kerning_net import create_dummy_kerning_net_model

    spec = ALL_MODEL_SPECS[model_name]
    pytorch_path = get_pytorch_path(spec)
    coreml_path = get_coreml_path(spec)

    # Load or create PyTorch model
    if pytorch_path.exists() and not use_dummy_pytorch:
        pytorch_model = load_pytorch_model(pytorch_path)
    else:
        logger.warning(f"PyTorch model not found, using dummy model for {model_name}")
        if model_name == "glyph_diffusion":
            pytorch_model = create_dummy_glyph_diffusion_model()
        elif model_name == "style_encoder":
            pytorch_model = create_dummy_style_encoder_model()
        elif model_name == "kerning_net":
            pytorch_model = create_dummy_kerning_net_model()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Load CoreML model
    if not coreml_path.exists():
        raise FileNotFoundError(f"CoreML model not found: {coreml_path}")

    coreml_model = load_coreml_model(coreml_path)

    return pytorch_model, coreml_model


# =============================================================================
# Comprehensive Verification Pipeline
# =============================================================================

def verify_all_models(
    num_samples: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    use_dummy_pytorch: bool = False,
) -> Dict[str, Any]:
    """
    Verify all converted CoreML models.

    Args:
        num_samples: Number of test samples per model
        rtol: Relative tolerance
        atol: Absolute tolerance
        use_dummy_pytorch: Use dummy PyTorch models

    Returns:
        Dictionary with verification results for all models
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "num_samples": num_samples,
            "rtol": rtol,
            "atol": atol,
        },
        "models": {},
        "summary": {
            "total_models": 0,
            "models_passed": 0,
            "models_failed": 0,
            "models_skipped": 0,
        }
    }

    for model_name in list_model_names():
        print_separator("-")
        logger.info(f"Verifying model: {model_name}")

        try:
            pytorch_model, coreml_model = load_models_for_verification(
                model_name, use_dummy_pytorch
            )

            model_results = verify_model(
                model_name=model_name,
                pytorch_model=pytorch_model,
                coreml_model=coreml_model,
                num_samples=num_samples,
                rtol=rtol,
                atol=atol,
            )

            results["models"][model_name] = model_results
            results["summary"]["total_models"] += 1

            if model_results["all_passed"]:
                results["summary"]["models_passed"] += 1
                logger.info(f"  PASSED: {model_results['samples_passed']}/{num_samples} samples")
            else:
                results["summary"]["models_failed"] += 1
                logger.warning(f"  FAILED: {model_results['samples_passed']}/{num_samples} samples")

            logger.info(f"  Max abs diff: {model_results['max_abs_diff']:.2e}")
            logger.info(f"  Max rel diff: {model_results['max_rel_diff']:.2e}")

        except FileNotFoundError as e:
            logger.warning(f"  SKIPPED: {e}")
            results["models"][model_name] = {"skipped": True, "reason": str(e)}
            results["summary"]["models_skipped"] += 1

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results["models"][model_name] = {"error": True, "reason": str(e)}
            results["summary"]["models_failed"] += 1

    return results


def verify_single_model(
    model_name: str,
    num_samples: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    use_dummy_pytorch: bool = False,
) -> Dict[str, Any]:
    """
    Verify a single converted CoreML model.

    Args:
        model_name: Name of the model to verify
        num_samples: Number of test samples
        rtol: Relative tolerance
        atol: Absolute tolerance
        use_dummy_pytorch: Use dummy PyTorch model

    Returns:
        Dictionary with verification results
    """
    if model_name not in ALL_MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list_model_names()}")

    pytorch_model, coreml_model = load_models_for_verification(
        model_name, use_dummy_pytorch
    )

    return verify_model(
        model_name=model_name,
        pytorch_model=pytorch_model,
        coreml_model=coreml_model,
        num_samples=num_samples,
        rtol=rtol,
        atol=atol,
    )


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a human-readable verification report.

    Args:
        results: Verification results dictionary
        output_path: Optional path to save report

    Returns:
        Report as string
    """
    lines = [
        "=" * 60,
        "COREML CONVERSION VERIFICATION REPORT",
        "=" * 60,
        f"Timestamp: {results.get('timestamp', 'N/A')}",
        "",
    ]

    # Settings
    settings = results.get("settings", {})
    lines.extend([
        "Settings:",
        f"  Samples per model: {settings.get('num_samples', 'N/A')}",
        f"  Relative tolerance: {settings.get('rtol', 'N/A')}",
        f"  Absolute tolerance: {settings.get('atol', 'N/A')}",
        "",
    ])

    # Summary
    summary = results.get("summary", {})
    lines.extend([
        "Summary:",
        f"  Total models: {summary.get('total_models', 0)}",
        f"  Passed: {summary.get('models_passed', 0)}",
        f"  Failed: {summary.get('models_failed', 0)}",
        f"  Skipped: {summary.get('models_skipped', 0)}",
        "",
    ])

    # Model details
    lines.append("Model Results:")
    lines.append("-" * 60)

    for model_name, model_results in results.get("models", {}).items():
        lines.append(f"\n{model_name}:")

        if model_results.get("skipped"):
            lines.append(f"  Status: SKIPPED")
            lines.append(f"  Reason: {model_results.get('reason', 'Unknown')}")
        elif model_results.get("error"):
            lines.append(f"  Status: ERROR")
            lines.append(f"  Reason: {model_results.get('reason', 'Unknown')}")
        else:
            status = "PASSED" if model_results.get("all_passed") else "FAILED"
            lines.append(f"  Status: {status}")
            lines.append(f"  Samples: {model_results.get('samples_passed', 0)}/{model_results.get('samples_tested', 0)}")
            lines.append(f"  Max abs diff: {model_results.get('max_abs_diff', 0):.2e}")
            lines.append(f"  Mean abs diff: {model_results.get('mean_abs_diff', 0):.2e}")
            lines.append(f"  Max rel diff: {model_results.get('max_rel_diff', 0):.2e}")
            lines.append(f"  Mean rel diff: {model_results.get('mean_rel_diff', 0):.2e}")

    lines.extend(["", "=" * 60])

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info(f"Report saved to: {output_path}")

    return report


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify CoreML model conversions by comparing with PyTorch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="all",
        choices=["all"] + list_model_names(),
        help="Model to verify (default: all)",
    )

    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of test samples per model (default: 10)",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for comparison (default: 1e-4)",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for comparison (default: 1e-4)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to write JSON results",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to write human-readable report",
    )

    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy PyTorch models (for testing)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
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

    print_separator()
    logger.info("CoreML Conversion Verification")
    print_separator()

    try:
        if args.model == "all":
            results = verify_all_models(
                num_samples=args.samples,
                rtol=args.rtol,
                atol=args.atol,
                use_dummy_pytorch=args.dummy,
            )
        else:
            results = {
                "timestamp": datetime.now().isoformat(),
                "settings": {
                    "num_samples": args.samples,
                    "rtol": args.rtol,
                    "atol": args.atol,
                },
                "models": {
                    args.model: verify_single_model(
                        model_name=args.model,
                        num_samples=args.samples,
                        rtol=args.rtol,
                        atol=args.atol,
                        use_dummy_pytorch=args.dummy,
                    )
                },
                "summary": {
                    "total_models": 1,
                    "models_passed": 0,
                    "models_failed": 0,
                    "models_skipped": 0,
                }
            }

            # Update summary
            model_result = results["models"][args.model]
            if model_result.get("all_passed"):
                results["summary"]["models_passed"] = 1
            elif model_result.get("skipped"):
                results["summary"]["models_skipped"] = 1
            else:
                results["summary"]["models_failed"] = 1

        # Output JSON results
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"JSON results saved to: {args.output}")

        # Generate and print report
        print_separator()
        report = generate_report(results, args.report)
        print(report)

        # Return exit code based on results
        summary = results.get("summary", {})
        if summary.get("models_failed", 0) > 0:
            return 1

        return 0

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
