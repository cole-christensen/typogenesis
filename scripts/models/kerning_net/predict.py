#!/usr/bin/env python3
"""
KerningNet Inference Script

Predict kerning values using a trained KerningNet model.
Supports single pair prediction, batch prediction for all critical pairs,
and full kerning table generation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.kerning_net.config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    get_all_critical_pairs,
    get_negative_kerning_pairs,
    get_zero_kerning_pairs,
)
from models.kerning_net.model import KerningNet, load_model


# =============================================================================
# Glyph Image Utilities
# =============================================================================

def load_glyph_image(
    path: Union[str, Path],
    target_size: int = 64,
) -> torch.Tensor:
    """
    Load and preprocess a glyph image.

    Args:
        path: Path to glyph image file.
        target_size: Target image size.

    Returns:
        Preprocessed tensor of shape (1, 1, H, W).
    """
    img = Image.open(path).convert("L")  # Grayscale
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    return tensor


def render_glyph_image(
    outline_data: Any,  # GlyphOutline or similar
    target_size: int = 64,
    padding: float = 0.1,
) -> torch.Tensor:
    """
    Render a glyph outline to an image tensor.

    This is a placeholder that should be integrated with the actual
    glyph rendering system from the Typogenesis app.

    Args:
        outline_data: Glyph outline data.
        target_size: Target image size.
        padding: Padding ratio around the glyph.

    Returns:
        Rendered tensor of shape (1, 1, H, W).
    """
    # TODO: Integrate with actual glyph rendering from Typogenesis
    # For now, return a placeholder
    raise NotImplementedError(
        "Glyph rendering not implemented. Use load_glyph_image() with "
        "pre-rendered glyph images, or integrate with Typogenesis renderer."
    )


# =============================================================================
# KerningNet Predictor Class
# =============================================================================

class KerningPredictor:
    """
    High-level interface for kerning prediction.

    Handles model loading, batching, and result formatting.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_config: Optional[ModelConfig] = None,
        dataset_config: Optional[DatasetConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the kerning predictor.

        Args:
            model_path: Path to trained model checkpoint.
            model_config: Model configuration.
            dataset_config: Dataset configuration (for de/normalization).
            inference_config: Inference configuration.
            device: Device to run inference on.
        """
        self.model_config = model_config or DEFAULT_MODEL_CONFIG
        self.dataset_config = dataset_config or DEFAULT_DATASET_CONFIG
        self.inference_config = inference_config or DEFAULT_INFERENCE_CONFIG

        # Determine device
        if device is None:
            device = self.inference_config.device
        self.device = self._get_device(device)

        # Load model
        if model_path is not None:
            self.model = load_model(
                str(model_path),
                self.model_config,
                self.device,
            )
        else:
            # Create uninitialized model (for testing)
            self.model = KerningNet(self.model_config)
            self.model.to(self.device)
            self.model.eval()

        # Cache for glyph embeddings
        self._embedding_cache: Dict[str, torch.Tensor] = {}

    def _get_device(self, preference: str) -> torch.device:
        """Get the device to use for inference."""
        if preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(preference)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def predict_single(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ) -> float:
        """
        Predict kerning for a single glyph pair.

        Args:
            left_image: Left glyph image tensor (1, 1, H, W) or (1, H, W).
            right_image: Right glyph image tensor.
            metrics: Optional font metrics tensor (4,).
            return_normalized: If True, return normalized value [-1, 1].

        Returns:
            Kerning value in UPM units (or normalized if specified).
        """
        # Ensure correct shape
        if left_image.dim() == 3:
            left_image = left_image.unsqueeze(0)
        if right_image.dim() == 3:
            right_image = right_image.unsqueeze(0)

        left_image = left_image.to(self.device)
        right_image = right_image.to(self.device)

        if metrics is not None:
            metrics = metrics.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(left_image, right_image, metrics)
            normalized_value = output.item()

        if return_normalized:
            return normalized_value

        # Denormalize to UPM units
        kerning = self.dataset_config.denormalize_kerning(normalized_value)

        if self.inference_config.round_to_int:
            kerning = round(kerning)

        return kerning

    def predict_batch(
        self,
        left_images: torch.Tensor,
        right_images: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ) -> List[float]:
        """
        Predict kerning for a batch of glyph pairs.

        Args:
            left_images: Batch of left glyph images (N, 1, H, W).
            right_images: Batch of right glyph images (N, 1, H, W).
            metrics: Optional batch of font metrics (N, 4).
            return_normalized: If True, return normalized values.

        Returns:
            List of kerning values.
        """
        left_images = left_images.to(self.device)
        right_images = right_images.to(self.device)

        if metrics is not None:
            metrics = metrics.to(self.device)

        with torch.no_grad():
            outputs = self.model(left_images, right_images, metrics)
            normalized_values = outputs.squeeze(-1).cpu().tolist()

        if return_normalized:
            return normalized_values

        # Denormalize
        kerning_values = [
            self.dataset_config.denormalize_kerning(v)
            for v in normalized_values
        ]

        if self.inference_config.round_to_int:
            kerning_values = [round(k) for k in kerning_values]

        return kerning_values

    def predict_from_paths(
        self,
        left_path: Union[str, Path],
        right_path: Union[str, Path],
        metrics: Optional[List[float]] = None,
    ) -> float:
        """
        Predict kerning from glyph image file paths.

        Args:
            left_path: Path to left glyph image.
            right_path: Path to right glyph image.
            metrics: Optional list of [left_advance, right_lsb, x_height, cap_height].

        Returns:
            Kerning value in UPM units.
        """
        left_img = load_glyph_image(left_path, self.model_config.image_size)
        right_img = load_glyph_image(right_path, self.model_config.image_size)

        metrics_tensor = None
        if metrics is not None:
            metrics_tensor = torch.tensor(metrics, dtype=torch.float32)

        return self.predict_single(left_img, right_img, metrics_tensor)

    def predict_critical_pairs(
        self,
        glyph_images: Dict[str, torch.Tensor],
        metrics: Optional[torch.Tensor] = None,
    ) -> Dict[Tuple[str, str], float]:
        """
        Predict kerning for all critical pairs.

        Args:
            glyph_images: Dictionary mapping character strings to image tensors.
            metrics: Optional font metrics tensor (4,).

        Returns:
            Dictionary mapping (left, right) pairs to kerning values.
        """
        critical_pairs = get_all_critical_pairs()
        results: Dict[Tuple[str, str], float] = {}

        # Filter to pairs we have images for
        available_pairs = [
            (left, right)
            for left, right in critical_pairs
            if left in glyph_images and right in glyph_images
        ]

        if not available_pairs:
            return results

        # Batch process for efficiency
        batch_size = self.inference_config.batch_size
        for i in range(0, len(available_pairs), batch_size):
            batch_pairs = available_pairs[i:i + batch_size]

            left_batch = torch.stack([glyph_images[p[0]] for p in batch_pairs])
            right_batch = torch.stack([glyph_images[p[1]] for p in batch_pairs])

            metrics_batch = None
            if metrics is not None:
                metrics_batch = metrics.unsqueeze(0).expand(len(batch_pairs), -1)

            kerning_values = self.predict_batch(left_batch, right_batch, metrics_batch)

            for pair, kerning in zip(batch_pairs, kerning_values):
                # Skip small values if configured
                if abs(kerning) >= self.inference_config.min_kerning_to_include:
                    results[pair] = kerning

        return results

    def predict_all_pairs(
        self,
        glyph_images: Dict[str, torch.Tensor],
        metrics: Optional[torch.Tensor] = None,
        characters: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, str], float]:
        """
        Predict kerning for all possible pairs of characters.

        Args:
            glyph_images: Dictionary mapping character strings to image tensors.
            metrics: Optional font metrics tensor (4,).
            characters: Optional list of characters to include. If None, uses all.

        Returns:
            Dictionary mapping (left, right) pairs to kerning values.
        """
        if characters is None:
            characters = list(glyph_images.keys())

        # Generate all pairs
        all_pairs = [
            (left, right)
            for left in characters
            for right in characters
            if left in glyph_images and right in glyph_images
        ]

        results: Dict[Tuple[str, str], float] = {}

        # Batch process
        batch_size = self.inference_config.batch_size
        for i in range(0, len(all_pairs), batch_size):
            batch_pairs = all_pairs[i:i + batch_size]

            left_batch = torch.stack([glyph_images[p[0]] for p in batch_pairs])
            right_batch = torch.stack([glyph_images[p[1]] for p in batch_pairs])

            metrics_batch = None
            if metrics is not None:
                metrics_batch = metrics.unsqueeze(0).expand(len(batch_pairs), -1)

            kerning_values = self.predict_batch(left_batch, right_batch, metrics_batch)

            for pair, kerning in zip(batch_pairs, kerning_values):
                if abs(kerning) >= self.inference_config.min_kerning_to_include:
                    results[pair] = kerning

        return results

    def generate_kerning_table(
        self,
        glyph_images: Dict[str, torch.Tensor],
        metrics: Optional[torch.Tensor] = None,
        critical_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate a kerning table suitable for font export.

        Args:
            glyph_images: Dictionary mapping character strings to image tensors.
            metrics: Optional font metrics tensor.
            critical_only: If True, only include critical pairs.

        Returns:
            List of kerning entries with 'left', 'right', 'value' keys.
        """
        if critical_only:
            pairs = self.predict_critical_pairs(glyph_images, metrics)
        else:
            pairs = self.predict_all_pairs(glyph_images, metrics)

        # Sort by absolute kerning value (most significant first)
        sorted_pairs = sorted(pairs.items(), key=lambda x: abs(x[1]), reverse=True)

        table = []
        for (left, right), value in sorted_pairs:
            table.append({
                "left": left,
                "right": right,
                "value": int(value) if self.inference_config.round_to_int else value,
            })

        return table


# =============================================================================
# Output Formatting
# =============================================================================

def format_table(
    results: Dict[Tuple[str, str], float],
    title: str = "Kerning Predictions",
) -> str:
    """Format results as a table string."""
    lines = [
        title,
        "=" * 40,
        f"{'Left':<8} {'Right':<8} {'Kerning':>10}",
        "-" * 40,
    ]

    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)

    for (left, right), value in sorted_results:
        # Escape special characters for display
        left_display = repr(left)[1:-1] if not left.isalnum() else left
        right_display = repr(right)[1:-1] if not right.isalnum() else right
        lines.append(f"{left_display:<8} {right_display:<8} {value:>10.1f}")

    lines.append("-" * 40)
    lines.append(f"Total pairs: {len(results)}")

    return "\n".join(lines)


def format_json(
    results: Dict[Tuple[str, str], float],
    round_values: bool = True,
) -> str:
    """Format results as JSON string."""
    data = {
        "kerning_pairs": [
            {
                "left": left,
                "right": right,
                "value": round(value) if round_values else value,
            }
            for (left, right), value in sorted(results.items())
        ],
        "count": len(results),
    }
    return json.dumps(data, indent=2)


def format_csv(
    results: Dict[Tuple[str, str], float],
    round_values: bool = True,
) -> str:
    """Format results as CSV string."""
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["left", "right", "kerning"])

    for (left, right), value in sorted(results.items()):
        writer.writerow([left, right, round(value) if round_values else value])

    return output.getvalue()


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict kerning values using trained KerningNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained model checkpoint"
    )

    # Input mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--pair", nargs=2, metavar=("LEFT", "RIGHT"),
        help="Predict kerning for a single pair (image paths)"
    )
    mode_group.add_argument(
        "--critical", action="store_true",
        help="Predict kerning for all critical pairs"
    )
    mode_group.add_argument(
        "--all", action="store_true",
        help="Predict kerning for all possible pairs"
    )

    # Input data
    parser.add_argument(
        "--glyph-dir", type=Path,
        help="Directory containing glyph images (named by character, e.g., A.png)"
    )
    parser.add_argument(
        "--metrics", nargs=4, type=float, metavar=("ADV", "LSB", "XHEIGHT", "CAPHEIGHT"),
        help="Font metrics: left_advance, right_lsb, x_height_ratio, cap_height_ratio"
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format", choices=["table", "json", "csv"], default="table",
        help="Output format"
    )
    parser.add_argument(
        "--min-kerning", type=int, default=2,
        help="Minimum kerning value to include"
    )

    # Device
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help="Device for inference"
    )

    return parser.parse_args()


def load_glyph_images_from_dir(
    directory: Path,
    target_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Load all glyph images from a directory."""
    images: Dict[str, torch.Tensor] = {}

    for img_path in directory.glob("*.png"):
        # Extract character from filename (e.g., "A.png" -> "A")
        char = img_path.stem
        if len(char) == 1:
            images[char] = load_glyph_image(img_path, target_size).squeeze(0)

    # Also check for special characters with descriptive names
    special_names = {
        "period": ".",
        "comma": ",",
        "quote": '"',
        "apostrophe": "'",
        "lparen": "(",
        "rparen": ")",
        "semicolon": ";",
        "colon": ":",
    }

    for name, char in special_names.items():
        special_path = directory / f"{name}.png"
        if special_path.exists():
            images[char] = load_glyph_image(special_path, target_size).squeeze(0)

    return images


def main() -> None:
    """Main entry point for inference script."""
    args = parse_args()

    # Configure inference
    inference_config = InferenceConfig(
        model_path=args.model,
        device=args.device,
        min_kerning_to_include=args.min_kerning,
        output_format=args.format,
    )

    # Create predictor
    print(f"Loading model from: {args.model}", file=sys.stderr)
    predictor = KerningPredictor(
        model_path=args.model,
        inference_config=inference_config,
        device=args.device,
    )
    print(f"Model loaded. Using device: {predictor.device}", file=sys.stderr)

    # Parse metrics
    metrics_tensor = None
    if args.metrics:
        metrics_tensor = torch.tensor(args.metrics, dtype=torch.float32)

    # Single pair prediction
    if args.pair:
        left_path, right_path = args.pair
        kerning = predictor.predict_from_paths(left_path, right_path, args.metrics)
        print(f"Kerning: {kerning}")
        return

    # Load glyph images for batch prediction
    if args.glyph_dir is None:
        print("Error: --glyph-dir required for critical/all pair prediction", file=sys.stderr)
        sys.exit(1)

    if not args.glyph_dir.exists():
        print(f"Error: Glyph directory not found: {args.glyph_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading glyph images from: {args.glyph_dir}", file=sys.stderr)
    glyph_images = load_glyph_images_from_dir(
        args.glyph_dir,
        predictor.model_config.image_size,
    )
    print(f"Loaded {len(glyph_images)} glyph images", file=sys.stderr)

    if len(glyph_images) < 2:
        print("Error: Need at least 2 glyph images for pair prediction", file=sys.stderr)
        sys.exit(1)

    # Predict
    if args.critical:
        print("Predicting critical pairs...", file=sys.stderr)
        results = predictor.predict_critical_pairs(glyph_images, metrics_tensor)
    else:  # args.all
        print("Predicting all pairs...", file=sys.stderr)
        results = predictor.predict_all_pairs(glyph_images, metrics_tensor)

    print(f"Generated {len(results)} kerning pairs", file=sys.stderr)

    # Format output
    if args.format == "table":
        output = format_table(results)
    elif args.format == "json":
        output = format_json(results)
    else:  # csv
        output = format_csv(results)

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        print(output)


# =============================================================================
# Verification Functions
# =============================================================================

def verify_model_predictions(
    predictor: KerningPredictor,
    glyph_images: Dict[str, torch.Tensor],
    metrics: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Verify that model predictions match expected behavior.

    Checks that:
    - Negative kerning pairs (AV, AT, etc.) have negative values
    - Zero kerning pairs (HH, II, etc.) have near-zero values

    Args:
        predictor: KerningPredictor instance.
        glyph_images: Dictionary of glyph images.
        metrics: Optional font metrics.

    Returns:
        Dictionary with verification results.
    """
    results = {
        "negative_pairs": {"expected": 0, "correct": 0, "pairs": []},
        "zero_pairs": {"expected": 0, "correct": 0, "pairs": []},
        "overall_correlation": 0.0,
    }

    # Check negative kerning pairs
    negative_pairs = get_negative_kerning_pairs()
    for left, right in negative_pairs:
        if left in glyph_images and right in glyph_images:
            results["negative_pairs"]["expected"] += 1

            left_img = glyph_images[left].unsqueeze(0)
            right_img = glyph_images[right].unsqueeze(0)
            kerning = predictor.predict_single(left_img, right_img, metrics)

            if kerning < 0:
                results["negative_pairs"]["correct"] += 1

            results["negative_pairs"]["pairs"].append({
                "pair": (left, right),
                "value": kerning,
                "correct": kerning < 0,
            })

    # Check zero kerning pairs
    zero_pairs = get_zero_kerning_pairs()
    for left, right in zero_pairs:
        if left in glyph_images and right in glyph_images:
            results["zero_pairs"]["expected"] += 1

            left_img = glyph_images[left].unsqueeze(0)
            right_img = glyph_images[right].unsqueeze(0)
            kerning = predictor.predict_single(left_img, right_img, metrics)

            # Consider "near zero" as within 20 UPM units
            is_near_zero = abs(kerning) < 20
            if is_near_zero:
                results["zero_pairs"]["correct"] += 1

            results["zero_pairs"]["pairs"].append({
                "pair": (left, right),
                "value": kerning,
                "correct": is_near_zero,
            })

    # Calculate accuracy
    neg_total = results["negative_pairs"]["expected"]
    neg_correct = results["negative_pairs"]["correct"]
    zero_total = results["zero_pairs"]["expected"]
    zero_correct = results["zero_pairs"]["correct"]

    results["negative_pairs"]["accuracy"] = neg_correct / neg_total if neg_total > 0 else 0
    results["zero_pairs"]["accuracy"] = zero_correct / zero_total if zero_total > 0 else 0

    return results


if __name__ == "__main__":
    main()
