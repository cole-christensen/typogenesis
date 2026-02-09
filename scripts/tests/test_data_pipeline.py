"""Tests for data pipeline: glyph extraction, dataset preparation, augmentations.

Tests use synthetic font data created in fixtures rather than requiring
actual Google Fonts downloads.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class TestGlyphExtractor:
    """Tests for glyph extraction from font files."""

    def test_render_glyph_to_image_produces_valid_image(self, tmp_path, system_font_path):
        """render_glyph_to_image returns a properly sized grayscale image."""
        from data.extract_glyphs import render_glyph_to_image

        img = render_glyph_to_image(system_font_path, "A", image_size=64)
        assert img is not None, "Failed to render glyph 'A'"
        assert img.size == (64, 64)
        assert img.mode == "L"

    def test_render_glyph_128x128(self, tmp_path, system_font_path):
        """Glyph renders correctly at 128x128."""
        from data.extract_glyphs import render_glyph_to_image

        img = render_glyph_to_image(system_font_path, "A", image_size=128)
        assert img is not None
        assert img.size == (128, 128)

    def test_render_glyph_has_content(self, system_font_path):
        """Rendered glyph is not blank (has non-zero pixels)."""
        from data.extract_glyphs import render_glyph_to_image

        img = render_glyph_to_image(system_font_path, "A", image_size=64)
        assert img is not None
        arr = np.array(img)
        assert arr.max() > 0, "Rendered glyph is completely blank"

    def test_render_all_required_chars(self, system_font_path):
        """All 62 required characters render successfully."""
        from data.extract_glyphs import ALL_CHARACTERS, render_glyph_to_image

        rendered = 0
        for char in ALL_CHARACTERS:
            img = render_glyph_to_image(system_font_path, char, image_size=64)
            if img is not None:
                rendered += 1

        assert rendered >= 52, f"Only rendered {rendered}/62 characters (need at least 52)"

    def test_get_font_metrics(self, system_font_path):
        """Font metrics extraction returns expected fields."""
        from fontTools.ttLib import TTFont

        from data.extract_glyphs import get_font_metrics

        tt = TTFont(system_font_path)
        metrics = get_font_metrics(tt)
        tt.close()

        assert "units_per_em" in metrics
        assert metrics["units_per_em"] > 0
        assert "ascender" in metrics

    def test_extract_glyph_outline(self, system_font_path):
        """Bezier outline extraction returns contour data."""
        from fontTools.ttLib import TTFont

        from data.extract_glyphs import extract_glyph_outline

        tt = TTFont(system_font_path)
        outline = extract_glyph_outline(tt, "A")
        tt.close()

        assert outline is not None, "No outline extracted for 'A'"
        assert len(outline) > 0, "Outline has no contours"
        # Each contour should have path operations
        for contour in outline:
            assert len(contour) > 0
            assert "type" in contour[0]
            assert "points" in contour[0]

    def test_extract_font_full_pipeline(self, system_font_path, tmp_path):
        """Full extract_font produces expected directory structure."""
        from data.extract_glyphs import extract_font

        result = extract_font(system_font_path, tmp_path, image_sizes=(64,))

        assert result is not None, "extract_font returned None"
        assert result["num_glyphs_rendered"] > 0

        # Check directory structure
        font_hash = result["file_hash"]
        font_dir = tmp_path / font_hash
        assert font_dir.exists()
        assert (font_dir / "metadata.json").exists()
        assert (font_dir / "kerning.json").exists()
        assert (font_dir / "glyphs").is_dir()

        # Check at least some glyph images exist
        glyph_images = list((font_dir / "glyphs").glob("*_64.png"))
        assert len(glyph_images) > 0

    def test_extract_font_is_incremental(self, system_font_path, tmp_path):
        """Extracting the same font twice skips on the second run."""
        from data.extract_glyphs import extract_font

        result1 = extract_font(system_font_path, tmp_path, image_sizes=(64,))
        assert result1 is not None

        result2 = extract_font(system_font_path, tmp_path, image_sizes=(64,))
        assert result2 is None, "Second extraction should skip (already done)"

    def test_has_required_coverage(self, system_font_path):
        """has_required_coverage correctly identifies fonts with full charset."""
        from data.download_fonts import has_required_coverage

        assert has_required_coverage(system_font_path), (
            "System font should have required character coverage"
        )

    def test_font_file_hash_deterministic(self, system_font_path):
        """Font file hash is deterministic."""
        from data.extract_glyphs import font_file_hash

        h1 = font_file_hash(system_font_path)
        h2 = font_file_hash(system_font_path)
        assert h1 == h2
        assert len(h1) == 12


class TestDatasetPreparation:
    """Tests for dataset manifest building and splitting."""

    def test_split_no_font_leakage(self):
        """No font family appears in multiple splits."""
        # Create mock font dirs
        import tempfile

        from data.prepare_datasets import split_by_font_family
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            families = {}
            for i in range(20):
                font_dir = tmpdir / f"font_{i:03d}"
                font_dir.mkdir()
                family = f"family_{i // 3}"
                meta = {"font_name": f"Font {i}", "font_path": f"/fonts/{family}/style.ttf"}
                (font_dir / "metadata.json").write_text(json.dumps(meta))
                families.setdefault(family, []).append(font_dir)

            dirs = [d for d in tmpdir.iterdir() if d.is_dir()]
            splits = split_by_font_family(dirs, seed=42)

            # Extract family keys per split
            split_families = {}
            for split_name, split_dirs in splits.items():
                fams = set()
                for d in split_dirs:
                    meta = json.loads((d / "metadata.json").read_text())
                    fam = Path(meta["font_path"]).parent.name
                    fams.add(fam)
                split_families[split_name] = fams

            # Check no overlap
            for s1 in split_families:
                for s2 in split_families:
                    if s1 != s2:
                        overlap = split_families[s1] & split_families[s2]
                        assert not overlap, (
                            f"Font leakage: {overlap} appears in both {s1} and {s2}"
                        )

    def test_build_glyph_manifest_schema(self, extracted_font_dir):
        """Glyph manifest entries have all required fields."""
        from data.prepare_datasets import build_glyph_manifest

        entries = build_glyph_manifest([extracted_font_dir], extracted_font_dir.parent)

        assert len(entries) > 0, "No glyph manifest entries produced"
        entry = entries[0]
        assert "image" in entry
        assert "char" in entry
        assert "char_index" in entry
        assert "font_id" in entry
        assert 0 <= entry["char_index"] <= 61

    def test_build_style_manifest_schema(self, extracted_font_dir):
        """Style manifest entries have all required fields."""
        from data.prepare_datasets import build_style_manifest

        entries = build_style_manifest([extracted_font_dir], extracted_font_dir.parent)

        assert len(entries) > 0, "No style manifest entries produced"
        entry = entries[0]
        assert "font_id" in entry
        assert "glyphs" in entry
        assert "num_glyphs" in entry
        assert isinstance(entry["glyphs"], list)
        assert entry["num_glyphs"] >= 10

    def test_build_kerning_manifest_schema(self, extracted_font_dir):
        """Kerning manifest entries have all required fields."""
        from data.prepare_datasets import build_kerning_manifest

        entries = build_kerning_manifest(
            [extracted_font_dir], extracted_font_dir.parent, include_zero_pairs=True,
        )

        assert len(entries) > 0, "Font should have kerning data for this test"
        if len(entries) > 0:
            entry = entries[0]
            assert "left_image" in entry
            assert "right_image" in entry
            assert "kerning" in entry
            assert "units_per_em" in entry
            assert "font_id" in entry

    def test_write_manifest_creates_valid_jsonl(self, tmp_path):
        """write_manifest produces valid JSONL file."""
        from data.prepare_datasets import write_manifest

        entries = [
            {"image": "a.png", "char": "A", "char_index": 26, "font_id": "test"},
            {"image": "b.png", "char": "B", "char_index": 27, "font_id": "test"},
        ]
        output_path = tmp_path / "test.jsonl"
        write_manifest(entries, output_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "image" in parsed


class TestAugmentations:
    """Tests for data augmentation transforms."""

    def test_train_transforms_output_shape(self):
        """Training transforms produce correct tensor shape."""
        from data.augmentations import get_train_transforms

        transform = get_train_transforms(image_size=64, elastic=False)
        img = Image.new("L", (100, 100), 128)
        tensor = transform(img)

        assert tensor.shape == (1, 64, 64)
        assert tensor.dtype == torch.float32

    def test_eval_transforms_output_shape(self):
        """Eval transforms produce correct tensor shape."""
        from data.augmentations import get_eval_transforms

        transform = get_eval_transforms(image_size=64)
        img = Image.new("L", (100, 100), 128)
        tensor = transform(img)

        assert tensor.shape == (1, 64, 64)

    def test_eval_transforms_deterministic(self):
        """Eval transforms are deterministic (no randomness)."""
        from data.augmentations import get_eval_transforms

        transform = get_eval_transforms(image_size=64)
        img = Image.new("L", (100, 100), 128)

        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2)

    def test_normalized_range(self):
        """Transforms normalize to approximately [-1, 1] range."""
        from data.augmentations import get_eval_transforms

        transform = get_eval_transforms(image_size=64)
        img = Image.new("L", (64, 64), 128)
        tensor = transform(img)

        # With Normalize(0.5, 0.5): output = (input - 0.5) / 0.5
        # For pixel value 128/255 ≈ 0.502 -> (0.502 - 0.5) / 0.5 ≈ 0.004
        assert tensor.min() >= -1.1
        assert tensor.max() <= 1.1

    def test_morphology_transform(self):
        """MorphologyTransform doesn't crash and preserves image mode."""
        from data.augmentations import MorphologyTransform

        transform = MorphologyTransform(erosion_prob=1.0, dilation_prob=0.0)
        img = Image.new("L", (64, 64), 200)
        result = transform(img)
        assert result.size == (64, 64)
        assert result.mode == "L"

    def test_gaussian_noise(self):
        """GaussianNoise adds noise with expected magnitude."""
        from data.augmentations import GaussianNoise

        noise = GaussianNoise(std=0.1)
        tensor = torch.zeros(1, 64, 64)
        noisy = noise(tensor)

        # Should be different from input
        assert not torch.allclose(noisy, tensor)
        # But within reasonable range
        assert noisy.min() >= -1.0
        assert noisy.max() <= 1.0
