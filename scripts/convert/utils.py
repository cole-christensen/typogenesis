#!/usr/bin/env python3
"""
Shared utilities for CoreML model conversion pipeline.

This module provides helper functions for:
- Loading PyTorch models
- Exporting to ONNX format
- Converting ONNX to CoreML
- Input preprocessing and output postprocessing
- Numerical validation
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Import Guards
# =============================================================================

def check_dependencies() -> bool:
    """Check that all required dependencies are available."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    try:
        import coremltools
    except ImportError:
        missing.append("coremltools")

    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False

    return True


# =============================================================================
# Model Loading
# =============================================================================

def load_pytorch_model(
    model_path: Union[str, Path],
    model_class: Optional[type] = None,
    device: str = "cpu",
    **kwargs: Any,
) -> Any:
    """
    Load a PyTorch model from a .pt file.

    Args:
        model_path: Path to the .pt model file
        model_class: Optional model class for instantiation (if not a full checkpoint)
        device: Device to load the model on ('cpu', 'cuda', 'mps')
        **kwargs: Additional arguments for model instantiation

    Returns:
        Loaded PyTorch model in eval mode

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {model_path}")

    logger.info(f"Loading PyTorch model from: {model_path}")

    try:
        # Try loading as a full model first
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Check if it's a state dict or a full model
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Training checkpoint format (used by Typogenesis training scripts)
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                # Alternative training checkpoint format
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume it's a raw state dict
                state_dict = checkpoint

            if model_class is None:
                raise ValueError(
                    "model_class must be provided when loading from state dict"
                )

            model = model_class(**kwargs)
            model.load_state_dict(state_dict)
        else:
            # Full model saved with torch.save(model, path)
            model = checkpoint

        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully (device: {device})")
        return model

    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        raise RuntimeError(f"Failed to load PyTorch model: {e}") from e


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get information about a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    import torch

    info = {
        "type": type(model).__name__,
        "trainable_params": 0,
        "total_params": 0,
        "layers": [],
    }

    for name, param in model.named_parameters():
        info["total_params"] += param.numel()
        if param.requires_grad:
            info["trainable_params"] += param.numel()
        info["layers"].append({
            "name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "trainable": param.requires_grad,
        })

    info["size_mb"] = info["total_params"] * 4 / (1024 * 1024)  # Assuming float32

    return info


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(
    model: Any,
    output_path: Union[str, Path],
    input_specs: List[Dict[str, Any]],
    output_names: List[str],
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    do_constant_folding: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode
        output_path: Path for the output .onnx file
        input_specs: List of input specifications, each containing:
            - name: Input name
            - shape: Input shape tuple
            - dtype: Data type ('float32', 'int64', etc.)
        output_names: List of output tensor names
        opset_version: ONNX opset version
        dynamic_axes: Optional dict specifying dynamic axes
        do_constant_folding: Whether to perform constant folding optimization
        verbose: Whether to print verbose output

    Returns:
        Path to the exported ONNX model

    Raises:
        RuntimeError: If export fails
    """
    import torch
    import onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting model to ONNX: {output_path}")
    logger.info(f"  Opset version: {opset_version}")
    logger.info(f"  Inputs: {[s['name'] for s in input_specs]}")
    logger.info(f"  Outputs: {output_names}")

    # Create dummy inputs
    dummy_inputs = []
    input_names = []

    for spec in input_specs:
        name = spec["name"]
        shape = spec["shape"]
        dtype = spec.get("dtype", "float32")

        input_names.append(name)

        # Create dummy tensor with appropriate dtype
        if dtype == "int64":
            dummy = torch.zeros(shape, dtype=torch.int64)
        elif dtype == "int32":
            dummy = torch.zeros(shape, dtype=torch.int32)
        elif dtype == "float16":
            dummy = torch.zeros(shape, dtype=torch.float16)
        else:
            dummy = torch.zeros(shape, dtype=torch.float32)

        dummy_inputs.append(dummy)

    # Handle single vs multiple inputs
    if len(dummy_inputs) == 1:
        dummy_input = dummy_inputs[0]
    else:
        dummy_input = tuple(dummy_inputs)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

        # Validate the exported model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        logger.info(f"ONNX export successful: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

        return output_path

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise RuntimeError(f"ONNX export failed: {e}") from e


def validate_onnx_model(onnx_path: Union[str, Path]) -> bool:
    """
    Validate an ONNX model file.

    Args:
        onnx_path: Path to the ONNX model

    Returns:
        True if valid, False otherwise
    """
    import onnx

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        logger.error(f"ONNX file not found: {onnx_path}")
        return False

    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info(f"ONNX model validation passed: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX model validation failed: {e}")
        return False


# =============================================================================
# CoreML Conversion
# =============================================================================

def convert_onnx_to_coreml(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    minimum_deployment_target: str = "macOS14",
    compute_units: str = "ALL",
    convert_to_float16: bool = True,
    input_descriptions: Optional[Dict[str, str]] = None,
    output_descriptions: Optional[Dict[str, str]] = None,
    model_description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
) -> Path:
    """
    Convert an ONNX model to CoreML format.

    Args:
        onnx_path: Path to the input ONNX model
        output_path: Path for the output .mlpackage
        minimum_deployment_target: Target macOS/iOS version
        compute_units: Compute units to use ('ALL', 'CPU_AND_NE', 'CPU_ONLY')
        convert_to_float16: Whether to convert weights to float16
        input_descriptions: Optional descriptions for each input
        output_descriptions: Optional descriptions for each output
        model_description: Optional model description
        author: Optional author name
        version: Optional version string

    Returns:
        Path to the exported CoreML model

    Raises:
        FileNotFoundError: If ONNX model doesn't exist
        RuntimeError: If conversion fails
    """
    import coremltools as ct

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting ONNX to CoreML: {onnx_path} -> {output_path}")
    logger.info(f"  Deployment target: {minimum_deployment_target}")
    logger.info(f"  Compute units: {compute_units}")
    logger.info(f"  Float16: {convert_to_float16}")

    try:
        # Map deployment target string to coremltools constant
        target_map = {
            "macOS13": ct.target.macOS13,
            "macOS14": ct.target.macOS14,
            "iOS16": ct.target.iOS16,
            "iOS17": ct.target.iOS17,
        }
        target = target_map.get(minimum_deployment_target, ct.target.macOS14)

        # Map compute units string to coremltools constant
        compute_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        }
        compute_unit = compute_map.get(compute_units, ct.ComputeUnit.ALL)

        # Convert ONNX to CoreML
        mlmodel = ct.convert(
            str(onnx_path),
            minimum_deployment_target=target,
            compute_units=compute_unit,
            convert_to="mlprogram",  # Use ML Program format for better optimization
        )

        # Convert to float16 if requested (for smaller model size)
        if convert_to_float16:
            logger.info("  Converting to float16 precision...")
            try:
                # Use the modern coremltools >= 7.0 API
                op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype="float16"
                )
                config = ct.optimize.coreml.OptimizationConfig(
                    global_config=op_config
                )
                mlmodel = ct.optimize.coreml.linear_quantize_weights(
                    mlmodel, config=config
                )
            except AttributeError:
                logger.warning(
                    "  ct.optimize.coreml.linear_quantize_weights not available. "
                    "Skipping float16 quantization. Update coremltools >= 7.0."
                )

        # Set metadata
        mlmodel.author = author or "Typogenesis"
        mlmodel.short_description = model_description or ""
        mlmodel.version = version or "1.0.0"

        # Set input/output descriptions if provided
        if input_descriptions:
            for name, desc in input_descriptions.items():
                if name in mlmodel.input_description:
                    mlmodel.input_description[name] = desc

        if output_descriptions:
            for name, desc in output_descriptions.items():
                if name in mlmodel.output_description:
                    mlmodel.output_description[name] = desc

        # Save the model
        mlmodel.save(str(output_path))

        logger.info(f"CoreML conversion successful: {output_path}")

        # Log model size
        if output_path.is_dir():
            # .mlpackage is a directory
            total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
            logger.info(f"  Package size: {total_size / (1024*1024):.2f} MB")

        return output_path

    except Exception as e:
        logger.error(f"CoreML conversion failed: {e}")
        raise RuntimeError(f"CoreML conversion failed: {e}") from e


def load_coreml_model(model_path: Union[str, Path]) -> Any:
    """
    Load a CoreML model for inference.

    Args:
        model_path: Path to the .mlpackage or .mlmodelc

    Returns:
        Loaded CoreML model

    Raises:
        FileNotFoundError: If model doesn't exist
        RuntimeError: If loading fails
    """
    import coremltools as ct

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"CoreML model not found: {model_path}")

    logger.info(f"Loading CoreML model: {model_path}")

    try:
        model = ct.models.MLModel(str(model_path))
        logger.info(f"CoreML model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load CoreML model: {e}")
        raise RuntimeError(f"Failed to load CoreML model: {e}") from e


# =============================================================================
# Input/Output Processing
# =============================================================================

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
    normalize: bool = True,
    add_batch_dim: bool = True,
    add_channel_dim: bool = True,
) -> np.ndarray:
    """
    Preprocess an image for model input.

    Args:
        image: Input image as numpy array (H, W) or (H, W, C)
        target_size: Target (height, width)
        grayscale: Convert to grayscale if True
        normalize: Normalize to [0, 1] range
        add_batch_dim: Add batch dimension at front
        add_channel_dim: Add channel dimension

    Returns:
        Preprocessed image array
    """
    from PIL import Image

    # Convert to PIL Image for resizing
    if image.ndim == 3 and image.shape[2] == 3:
        pil_image = Image.fromarray(image.astype(np.uint8))
    elif image.ndim == 2:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        pil_image = Image.fromarray(image.astype(np.uint8))

    # Convert to grayscale if needed
    if grayscale and pil_image.mode != "L":
        pil_image = pil_image.convert("L")

    # Resize
    if pil_image.size != target_size[::-1]:  # PIL uses (W, H)
        pil_image = pil_image.resize(target_size[::-1], Image.Resampling.BILINEAR)

    # Convert back to numpy
    processed = np.array(pil_image, dtype=np.float32)

    # Normalize
    if normalize:
        processed = processed / 255.0

    # Add dimensions
    if add_channel_dim and processed.ndim == 2:
        processed = processed[np.newaxis, :, :]  # Add channel dim

    if add_batch_dim:
        processed = processed[np.newaxis, :, :, :] if processed.ndim == 3 else processed[np.newaxis, :]

    return processed


def postprocess_glyph_output(
    output: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Postprocess model output to glyph image.

    Args:
        output: Model output tensor
        threshold: Binarization threshold

    Returns:
        Processed glyph image (H, W), values in [0, 255]
    """
    # Remove batch and channel dimensions if present
    if output.ndim == 4:
        output = output[0, 0]  # (B, C, H, W) -> (H, W)
    elif output.ndim == 3:
        output = output[0]  # (C, H, W) -> (H, W) or (B, H, W) -> (H, W)

    # Normalize to [0, 1]
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # Convert to uint8
    output = (output * 255).astype(np.uint8)

    return output


# =============================================================================
# Numerical Validation
# =============================================================================

def compare_outputs(
    output1: np.ndarray,
    output2: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Tuple[bool, Dict[str, float]]:
    """
    Compare two output arrays for numerical equivalence.

    Args:
        output1: First output array
        output2: Second output array
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (is_close, statistics_dict)
    """
    # Flatten for comparison
    flat1 = output1.flatten().astype(np.float64)
    flat2 = output2.flatten().astype(np.float64)

    if flat1.shape != flat2.shape:
        return False, {"error": "Shape mismatch", "shape1": flat1.shape, "shape2": flat2.shape}

    # Compute differences
    abs_diff = np.abs(flat1 - flat2)
    rel_diff = abs_diff / (np.abs(flat1) + 1e-10)

    stats = {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "rtol": rtol,
        "atol": atol,
    }

    is_close = np.allclose(flat1, flat2, rtol=rtol, atol=atol)
    stats["is_close"] = is_close

    return is_close, stats


def generate_random_input(
    shape: Tuple[int, ...],
    dtype: str = "float32",
    low: float = 0.0,
    high: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random input tensor for testing.

    Args:
        shape: Tensor shape
        dtype: Data type
        low: Minimum value
        high: Maximum value
        seed: Random seed for reproducibility

    Returns:
        Random numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    if dtype in ("int32", "int64"):
        return np.random.randint(int(low), int(high) + 1, size=shape).astype(dtype)
    else:
        return np.random.uniform(low, high, size=shape).astype(dtype)


# =============================================================================
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """Simple progress tracker for conversion steps."""

    def __init__(self, total_steps: int, description: str = "Converting"):
        self.total = total_steps
        self.current = 0
        self.description = description

    def update(self, step_name: str) -> None:
        """Update progress with a step name."""
        self.current += 1
        progress = self.current / self.total * 100
        logger.info(f"[{progress:5.1f}%] {self.description}: {step_name}")

    def complete(self) -> None:
        """Mark conversion as complete."""
        logger.info(f"[100.0%] {self.description}: Complete!")


# =============================================================================
# CLI Helpers
# =============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """
    Configure logging for conversion scripts.

    Args:
        verbose: Enable debug-level logging
        log_file: Optional file to write logs to
    """
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def print_separator(char: str = "=", width: int = 60) -> None:
    """Print a separator line."""
    print(char * width)


def format_size(size_bytes: int) -> str:
    """Format a size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"
