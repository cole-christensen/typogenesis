"""Build dataset manifests and train/val/test splits for all three models.

Reads extracted glyph data and produces JSONL manifest files that the
PyTorch Dataset classes consume during training.

Usage:
    python -m data.prepare_datasets --data-dir data/extracted
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALL_CHARACTERS)}


def split_by_font_family(
    font_dirs: list[Path],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Split font directories into train/val/test by font family.

    Groups fonts by family (using metadata), then assigns entire families
    to splits to prevent data leakage.

    Args:
        font_dirs: List of extracted font directories.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set.
        test_frac: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name to list of font directories.
    """
    # Group by font family
    families = defaultdict(list)
    for font_dir in font_dirs:
        meta_path = font_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        # Extract family key for grouping related fonts together
        font_name = meta.get("font_name", font_dir.name)
        font_path = meta.get("font_path", "")

        # Try path-based grouping: if parent dir isn't a generic folder, use it
        # (works for Google Fonts: /fonts/roboto/Regular.ttf → "roboto")
        generic_dirs = {"Fonts", "Supplemental", "fonts", "ofl", "apache", ""}
        if font_path:
            parent = Path(font_path).parent
            parent_name = parent.name
            grandparent_name = parent.parent.name if parent.parent else ""
            if parent_name not in generic_dirs:
                family_key = parent_name
            elif grandparent_name and grandparent_name not in generic_dirs:
                family_key = grandparent_name
            else:
                # Fallback: strip style suffixes from font name
                style_suffixes = {"Bold", "Italic", "Regular", "Light", "Medium", "Thin",
                                  "Heavy", "Black", "Condensed", "Narrow", "Wide",
                                  "Oblique", "Roman", "Plain"}
                parts = font_name.split()
                family_parts = []
                for part in parts:
                    if part in style_suffixes:
                        break
                    family_parts.append(part)
                family_key = " ".join(family_parts) if family_parts else font_name
        else:
            family_key = font_name.split()[0] if font_name else font_dir.name
        families[family_key].append(font_dir)

    family_names = sorted(families.keys())
    rng = random.Random(seed)
    rng.shuffle(family_names)

    n = len(family_names)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_families = family_names[:n_train]
    val_families = family_names[n_train:n_train + n_val]
    test_families = family_names[n_train + n_val:]

    splits = {"train": [], "val": [], "test": []}
    for fam in train_families:
        splits["train"].extend(families[fam])
    for fam in val_families:
        splits["val"].extend(families[fam])
    for fam in test_families:
        splits["test"].extend(families[fam])

    logger.info(
        f"Split {n} families -> train={len(train_families)} ({len(splits['train'])} fonts), "
        f"val={len(val_families)} ({len(splits['val'])} fonts), "
        f"test={len(test_families)} ({len(splits['test'])} fonts)"
    )

    return splits


def build_glyph_manifest(
    font_dirs: list[Path],
    data_dir: Path,
    image_size: int = 64,
) -> list[dict]:
    """Build manifest entries for the GlyphDataset.

    Each entry is one glyph image with its character index and font ID.

    Args:
        font_dirs: List of extracted font directories.
        data_dir: Root data directory (for relative paths).
        image_size: Image size to reference.

    Returns:
        List of manifest entry dicts.
    """
    entries = []
    for font_dir in font_dirs:
        meta_path = font_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        font_id = meta.get("file_hash", font_dir.name)
        chars = meta.get("available_characters", [])

        for char in chars:
            char_name = f"U+{ord(char):04X}"
            img_path = font_dir / "glyphs" / f"{char_name}_{image_size}.png"
            if img_path.exists():
                entries.append({
                    "image": str(img_path.relative_to(data_dir)),
                    "char": char,
                    "char_index": CHAR_TO_IDX.get(char, -1),
                    "font_id": font_id,
                })

    return entries


def build_style_manifest(
    font_dirs: list[Path],
    data_dir: Path,
    image_size: int = 64,
) -> list[dict]:
    """Build manifest entries for the StyleDataset.

    Each entry is a font with all its glyph image paths, used for
    contrastive learning (same-font glyphs = positive pairs).

    Args:
        font_dirs: List of extracted font directories.
        data_dir: Root data directory (for relative paths).
        image_size: Image size to reference.

    Returns:
        List of manifest entry dicts.
    """
    entries = []
    for font_dir in font_dirs:
        meta_path = font_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        font_id = meta.get("file_hash", font_dir.name)
        chars = meta.get("available_characters", [])

        glyph_paths = []
        for char in chars:
            char_name = f"U+{ord(char):04X}"
            img_path = font_dir / "glyphs" / f"{char_name}_{image_size}.png"
            if img_path.exists():
                glyph_paths.append(str(img_path.relative_to(data_dir)))

        if len(glyph_paths) >= 10:  # Minimum glyphs for contrastive learning
            entries.append({
                "font_id": font_id,
                "font_name": meta.get("font_name", ""),
                "glyphs": glyph_paths,
                "num_glyphs": len(glyph_paths),
            })

    return entries


def build_kerning_manifest(
    font_dirs: list[Path],
    data_dir: Path,
    image_size: int = 64,
    include_zero_pairs: bool = True,
    max_zero_ratio: float = 1.0,
) -> list[dict]:
    """Build manifest entries for the KerningDataset.

    Each entry is a glyph pair with its kerning value. Includes both
    kerned pairs (from font tables) and unkerned pairs (as negatives).

    Args:
        font_dirs: List of extracted font directories.
        data_dir: Root data directory (for relative paths).
        image_size: Image size to reference.
        include_zero_pairs: Whether to include pairs with zero kerning.
        max_zero_ratio: Maximum ratio of zero-kerning pairs to non-zero pairs.

    Returns:
        List of manifest entry dicts.
    """
    entries = []
    rng = random.Random(42)

    for font_dir in font_dirs:
        meta_path = font_dir / "metadata.json"
        kerning_path = font_dir / "kerning.json"
        if not meta_path.exists() or not kerning_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        font_id = meta.get("file_hash", font_dir.name)
        upm = meta.get("metrics", {}).get("units_per_em", 1000)
        kerning_pairs = json.loads(kerning_path.read_text())
        chars = set(meta.get("available_characters", []))

        nonzero_entries = []
        kerned_pairs_set = set()

        for pair in kerning_pairs:
            left = pair["left"]
            right = pair["right"]
            value = pair["value"]

            if left not in chars or right not in chars:
                continue

            left_name = f"U+{ord(left):04X}"
            right_name = f"U+{ord(right):04X}"
            left_path = font_dir / "glyphs" / f"{left_name}_{image_size}.png"
            right_path = font_dir / "glyphs" / f"{right_name}_{image_size}.png"

            if left_path.exists() and right_path.exists():
                nonzero_entries.append({
                    "left_image": str(left_path.relative_to(data_dir)),
                    "right_image": str(right_path.relative_to(data_dir)),
                    "left_char": left,
                    "right_char": right,
                    "kerning": value,
                    "units_per_em": upm,
                    "font_id": font_id,
                })
                kerned_pairs_set.add((left, right))

        entries.extend(nonzero_entries)

        # Add zero-kerning pairs as negatives
        if include_zero_pairs and nonzero_entries:
            char_list = sorted(chars & set(ALL_CHARACTERS))
            max_zeros = int(len(nonzero_entries) * max_zero_ratio)
            zero_count = 0

            # Sample random pairs that don't have kerning
            for _ in range(max_zeros * 3):  # Over-sample then trim
                if zero_count >= max_zeros:
                    break
                left = rng.choice(char_list)
                right = rng.choice(char_list)
                if (left, right) in kerned_pairs_set:
                    continue

                left_name = f"U+{ord(left):04X}"
                right_name = f"U+{ord(right):04X}"
                left_path = font_dir / "glyphs" / f"{left_name}_{image_size}.png"
                right_path = font_dir / "glyphs" / f"{right_name}_{image_size}.png"

                if left_path.exists() and right_path.exists():
                    entries.append({
                        "left_image": str(left_path.relative_to(data_dir)),
                        "right_image": str(right_path.relative_to(data_dir)),
                        "left_char": left,
                        "right_char": right,
                        "kerning": 0,
                        "units_per_em": upm,
                        "font_id": font_id,
                    })
                    zero_count += 1

    return entries


def write_manifest(entries: list[dict], output_path: Path) -> None:
    """Write manifest entries to a JSONL file.

    Args:
        entries: List of manifest entry dicts.
        output_path: Path to write the JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Wrote {len(entries)} entries to {output_path}")


def prepare_datasets(
    data_dir: Path,
    image_size: int = 64,
    seed: int = 42,
) -> dict:
    """Build all dataset manifests from extracted font data.

    Args:
        data_dir: Directory containing extracted font data.
        image_size: Image size to reference in manifests.
        seed: Random seed for split reproducibility.

    Returns:
        Summary dict with manifest paths and entry counts.
    """
    # Find all extracted font directories
    font_dirs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ]
    logger.info(f"Found {len(font_dirs)} extracted fonts in {data_dir}")

    if not font_dirs:
        logger.warning("No extracted fonts found. Run extract_glyphs.py first.")
        return {"error": "no fonts found"}

    # Split by font family
    splits = split_by_font_family(font_dirs, seed=seed)

    summary = {"manifests": {}}

    for split_name, split_dirs in splits.items():
        # Glyph manifest
        glyph_entries = build_glyph_manifest(split_dirs, data_dir, image_size)
        glyph_path = data_dir / f"glyph_{split_name}.jsonl"
        write_manifest(glyph_entries, glyph_path)
        summary["manifests"][f"glyph_{split_name}"] = {
            "path": str(glyph_path),
            "count": len(glyph_entries),
        }

        # Style manifest
        style_entries = build_style_manifest(split_dirs, data_dir, image_size)
        style_path = data_dir / f"style_{split_name}.jsonl"
        write_manifest(style_entries, style_path)
        summary["manifests"][f"style_{split_name}"] = {
            "path": str(style_path),
            "count": len(style_entries),
        }

        # Kerning manifest
        kerning_entries = build_kerning_manifest(split_dirs, data_dir, image_size)
        kerning_path = data_dir / f"kerning_{split_name}.jsonl"
        write_manifest(kerning_entries, kerning_path)
        summary["manifests"][f"kerning_{split_name}"] = {
            "path": str(kerning_path),
            "count": len(kerning_entries),
        }

    summary_path = data_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Dataset preparation complete: {json.dumps(summary, indent=2)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build dataset manifests from extracted fonts")
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing extracted font data",
    )
    parser.add_argument(
        "--image-size", type=int, default=64,
        help="Image size to reference in manifests",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split reproducibility",
    )
    args = parser.parse_args()

    prepare_datasets(args.data_dir, args.image_size, args.seed)


if __name__ == "__main__":
    main()
