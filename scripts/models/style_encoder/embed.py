#!/usr/bin/env python3
"""Embedding extraction utilities for StyleEncoder.

This module provides utilities for extracting style embeddings from glyph images
using a trained StyleEncoder model. It supports:
    - Single glyph embedding extraction
    - Batch embedding extraction
    - Full font embedding (averaged across glyphs)
    - Embedding similarity computation
    - Saving/loading embeddings to/from files

Usage:
    # Extract embedding for a single glyph
    python embed.py --image glyph.png --checkpoint model.pt

    # Extract embeddings for all glyphs in a font directory
    python embed.py --font_dir MyFont/ --checkpoint model.pt --output embeddings.npz

    # Extract embeddings for multiple fonts
    python embed.py --data_dir fonts/ --checkpoint model.pt --output all_embeddings.npz

    # Compute similarity between two fonts
    python embed.py --compare font_a.npz font_b.npz

Example (programmatic):
    >>> from scripts.models.style_encoder.embed import StyleEmbedder
    >>> embedder = StyleEmbedder("checkpoints/best.pt")
    >>>
    >>> # Single image
    >>> embedding = embedder.embed_image("glyph.png")
    >>>
    >>> # Full font (average)
    >>> font_embedding = embedder.embed_font("MyFont/")
    >>>
    >>> # Compare fonts
    >>> similarity = embedder.compute_similarity(font_a_embedding, font_b_embedding)
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import StyleEncoderConfig
from .model import StyleEncoder, create_style_encoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Type aliases
Tensor = torch.Tensor
NDArray = np.ndarray


class GlyphImageDataset(Dataset):
    """Simple dataset for batch processing glyph images.

    Attributes:
        image_paths: List of paths to glyph images
        transform: Image transformation pipeline
    """

    def __init__(
        self,
        image_paths: list[Path],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            image_paths: List of paths to glyph images
            transform: Optional transformation pipeline
        """
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        """Get image and its filename.

        Args:
            idx: Image index

        Returns:
            Tuple of (image_tensor, filename)
        """
        path = self.image_paths[idx]
        image = Image.open(path).convert("L")  # Grayscale
        image = self.transform(image)
        return image, path.name


class StyleEmbedder:
    """High-level interface for extracting style embeddings.

    Provides methods for embedding single images, font directories,
    and computing similarities between embeddings.

    Attributes:
        model: Loaded StyleEncoder model
        device: Computation device (CPU/GPU)
        transform: Image preprocessing pipeline
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[StyleEncoderConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize StyleEmbedder.

        Args:
            checkpoint_path: Path to trained model checkpoint.
                           If None, creates model with random weights.
            config: Model configuration. If None, uses default config
                   or config from checkpoint.
            device: Computation device. If None, uses CUDA if available.

        Example:
            >>> embedder = StyleEmbedder("best_model.pt")
            >>> embedding = embedder.embed_image("glyph.png")
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model
        if checkpoint_path is not None:
            self.model = create_style_encoder(
                config=config,
                checkpoint_path=str(checkpoint_path),
                device=self.device,
            )
        else:
            self.model = create_style_encoder(
                config=config or StyleEncoderConfig(),
                device=self.device,
            )

        self.model.eval()

        # Standard preprocessing
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        logger.info(f"StyleEmbedder initialized on {self.device}")

    @torch.no_grad()
    def embed_image(
        self,
        image: Union[str, Path, Image.Image, Tensor, NDArray],
    ) -> NDArray:
        """Extract embedding from a single glyph image.

        Args:
            image: Input image as:
                - Path to image file (str or Path)
                - PIL Image object
                - PyTorch tensor (C, H, W) or (H, W)
                - NumPy array (H, W) or (H, W, C)

        Returns:
            Style embedding as NumPy array, shape (embedding_dim,)

        Example:
            >>> embedding = embedder.embed_image("glyph_A.png")
            >>> print(embedding.shape)  # (128,)
        """
        # Convert to tensor
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("L")
            tensor = self.transform(image)
        elif isinstance(image, Image.Image):
            tensor = self.transform(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image).convert("L")
            elif image.ndim == 2:
                image = Image.fromarray(image.astype(np.uint8), mode="L")
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
            tensor = self.transform(image)
        elif isinstance(image, Tensor):
            if image.ndim == 2:
                image = image.unsqueeze(0)  # Add channel dim
            tensor = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)

        # Extract embedding
        embedding = self.model.encode(tensor)

        return embedding.cpu().numpy().squeeze()

    @torch.no_grad()
    def embed_batch(
        self,
        images: list[Union[str, Path, Image.Image]],
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> tuple[NDArray, list[str]]:
        """Extract embeddings from multiple images.

        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            num_workers: Number of data loader workers

        Returns:
            Tuple of (embeddings, filenames):
                - embeddings: NumPy array, shape (num_images, embedding_dim)
                - filenames: List of image filenames/identifiers

        Example:
            >>> images = list(Path("font/").glob("*.png"))
            >>> embeddings, names = embedder.embed_batch(images)
        """
        # Convert to paths
        paths = []
        for img in images:
            if isinstance(img, (str, Path)):
                paths.append(Path(img))
            elif isinstance(img, Image.Image):
                # Save temporarily (not ideal, but ensures consistent handling)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    img.save(f.name)
                    paths.append(Path(f.name))

        # Create dataset and loader
        dataset = GlyphImageDataset(paths, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_embeddings = []
        all_names = []

        for batch_images, names in loader:
            batch_images = batch_images.to(self.device)
            embeddings = self.model.encode(batch_images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_names.extend(names)

        return np.concatenate(all_embeddings, axis=0), all_names

    @torch.no_grad()
    def embed_font(
        self,
        font_dir: Union[str, Path],
        aggregation: str = "mean",
        return_per_glyph: bool = False,
    ) -> Union[NDArray, tuple[NDArray, dict[str, NDArray]]]:
        """Extract embedding for an entire font.

        Processes all glyph images in a font directory and optionally
        aggregates them into a single font-level embedding.

        Args:
            font_dir: Path to directory containing glyph images
            aggregation: How to aggregate glyph embeddings:
                - "mean": Average of all glyph embeddings
                - "median": Median of all glyph embeddings
                - "none": Return all glyph embeddings (same as return_per_glyph=True)
            return_per_glyph: If True, also return individual glyph embeddings

        Returns:
            If return_per_glyph is False:
                Font embedding, shape (embedding_dim,)
            If return_per_glyph is True:
                Tuple of (font_embedding, per_glyph_dict) where per_glyph_dict
                maps glyph filename to its embedding

        Example:
            >>> font_emb = embedder.embed_font("MyFont/")
            >>> font_emb, glyphs = embedder.embed_font("MyFont/", return_per_glyph=True)
        """
        font_dir = Path(font_dir)
        if not font_dir.exists():
            raise ValueError(f"Font directory does not exist: {font_dir}")

        # Find all glyph images
        image_paths = (
            list(font_dir.glob("*.png")) +
            list(font_dir.glob("*.jpg")) +
            list(font_dir.glob("*.jpeg"))
        )

        if not image_paths:
            raise ValueError(f"No image files found in {font_dir}")

        logger.info(f"Embedding {len(image_paths)} glyphs from {font_dir}")

        # Get all embeddings
        embeddings, names = self.embed_batch(image_paths)

        # Create per-glyph dictionary
        per_glyph = {name: emb for name, emb in zip(names, embeddings)}

        # Aggregate
        if aggregation == "mean":
            font_embedding = embeddings.mean(axis=0)
        elif aggregation == "median":
            font_embedding = np.median(embeddings, axis=0)
        elif aggregation == "none":
            return_per_glyph = True
            font_embedding = embeddings
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        if return_per_glyph:
            return font_embedding, per_glyph
        return font_embedding

    def compute_similarity(
        self,
        embedding_a: NDArray,
        embedding_b: NDArray,
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding_a: First embedding, shape (embedding_dim,)
            embedding_b: Second embedding, shape (embedding_dim,)

        Returns:
            Cosine similarity in range [-1, 1], where 1 means identical.

        Example:
            >>> emb_a = embedder.embed_font("FontA/")
            >>> emb_b = embedder.embed_font("FontB/")
            >>> similarity = embedder.compute_similarity(emb_a, emb_b)
            >>> print(f"Similarity: {similarity:.3f}")
        """
        # Normalize
        a_norm = embedding_a / (np.linalg.norm(embedding_a) + 1e-8)
        b_norm = embedding_b / (np.linalg.norm(embedding_b) + 1e-8)

        # Cosine similarity
        return float(np.dot(a_norm, b_norm))

    def compute_similarity_matrix(
        self,
        embeddings: NDArray,
    ) -> NDArray:
        """Compute pairwise similarity matrix for multiple embeddings.

        Args:
            embeddings: Embeddings, shape (num_samples, embedding_dim)

        Returns:
            Similarity matrix, shape (num_samples, num_samples)
        """
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized = embeddings / norms

        # Compute similarity matrix
        return np.dot(normalized, normalized.T)

    def find_similar_fonts(
        self,
        query_embedding: NDArray,
        font_embeddings: dict[str, NDArray],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find fonts most similar to a query embedding.

        Args:
            query_embedding: Query font embedding, shape (embedding_dim,)
            font_embeddings: Dictionary mapping font names to embeddings
            top_k: Number of top results to return

        Returns:
            List of (font_name, similarity) tuples, sorted by similarity (descending)

        Example:
            >>> query = embedder.embed_font("MyFont/")
            >>> similar = embedder.find_similar_fonts(query, font_db, top_k=5)
            >>> for name, sim in similar:
            ...     print(f"{name}: {sim:.3f}")
        """
        similarities = []
        for name, embedding in font_embeddings.items():
            sim = self.compute_similarity(query_embedding, embedding)
            similarities.append((name, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def interpolate_styles(
        self,
        embedding_a: NDArray,
        embedding_b: NDArray,
        num_steps: int = 5,
    ) -> NDArray:
        """Interpolate between two style embeddings.

        Creates a smooth transition from style A to style B.

        Args:
            embedding_a: Start embedding, shape (embedding_dim,)
            embedding_b: End embedding, shape (embedding_dim,)
            num_steps: Number of interpolation steps (including endpoints)

        Returns:
            Interpolated embeddings, shape (num_steps, embedding_dim)

        Example:
            >>> emb_a = embedder.embed_font("LightFont/")
            >>> emb_b = embedder.embed_font("BoldFont/")
            >>> interpolated = embedder.interpolate_styles(emb_a, emb_b, num_steps=5)
        """
        t_values = np.linspace(0, 1, num_steps)
        interpolated = []

        for t in t_values:
            interp = (1 - t) * embedding_a + t * embedding_b
            # Re-normalize
            interp = interp / (np.linalg.norm(interp) + 1e-8)
            interpolated.append(interp)

        return np.array(interpolated)


def save_embeddings(
    embeddings: dict[str, NDArray],
    output_path: Union[str, Path],
) -> None:
    """Save embeddings to a compressed NumPy file.

    Args:
        embeddings: Dictionary mapping names to embedding arrays
        output_path: Path to save file (will be .npz format)

    Example:
        >>> embeddings = {"FontA": emb_a, "FontB": emb_b}
        >>> save_embeddings(embeddings, "font_embeddings.npz")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path, **embeddings)
    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")


def load_embeddings(input_path: Union[str, Path]) -> dict[str, NDArray]:
    """Load embeddings from a NumPy file.

    Args:
        input_path: Path to .npz file

    Returns:
        Dictionary mapping names to embedding arrays

    Example:
        >>> embeddings = load_embeddings("font_embeddings.npz")
        >>> print(embeddings["FontA"].shape)  # (128,)
    """
    data = np.load(input_path)
    embeddings = {key: data[key] for key in data.files}
    logger.info(f"Loaded {len(embeddings)} embeddings from {input_path}")
    return embeddings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract style embeddings from glyph images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=Path,
        help="Single glyph image to embed",
    )
    input_group.add_argument(
        "--font_dir",
        type=Path,
        help="Directory containing glyph images for a single font",
    )
    input_group.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing multiple font subdirectories",
    )
    input_group.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two embedding files",
    )

    # Model options
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="Path to trained model checkpoint",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for embeddings (.npz format)",
    )

    # Processing options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "none"],
        help="How to aggregate glyph embeddings for a font",
    )

    # Other options
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for embedding extraction."""
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Handle compare mode (doesn't need model)
    if args.compare:
        emb_a = load_embeddings(args.compare[0])
        emb_b = load_embeddings(args.compare[1])

        # Get first embedding from each
        key_a = list(emb_a.keys())[0]
        key_b = list(emb_b.keys())[0]

        embedder = StyleEmbedder(args.checkpoint, device=device)
        sim = embedder.compute_similarity(emb_a[key_a], emb_b[key_b])

        print(f"Similarity between {key_a} and {key_b}: {sim:.4f}")
        return

    # Initialize embedder
    embedder = StyleEmbedder(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    results = {}

    if args.image:
        # Single image
        embedding = embedder.embed_image(args.image)
        results[args.image.stem] = embedding
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 10 dims): {embedding[:10]}")

    elif args.font_dir:
        # Single font directory
        embedding, per_glyph = embedder.embed_font(
            args.font_dir,
            aggregation=args.aggregation,
            return_per_glyph=True,
        )
        results[args.font_dir.name] = embedding
        print(f"Font embedding shape: {embedding.shape}")
        print(f"Number of glyphs: {len(per_glyph)}")

    elif args.data_dir:
        # Multiple fonts
        for font_dir in args.data_dir.iterdir():
            if not font_dir.is_dir():
                continue

            try:
                embedding = embedder.embed_font(
                    font_dir,
                    aggregation=args.aggregation,
                )
                results[font_dir.name] = embedding
                logger.info(f"Embedded {font_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to embed {font_dir.name}: {e}")

        print(f"Embedded {len(results)} fonts")

    # Save results
    if args.output and results:
        save_embeddings(results, args.output)


if __name__ == "__main__":
    main()
