"""Tests for data pipeline: font downloading, glyph extraction, dataset preparation.

This is a stub - real tests will be implemented in Phase 7B when the data
pipeline modules (download_fonts.py, extract_glyphs.py, prepare_datasets.py)
are built.
"""

import pytest


@pytest.mark.data
class TestFontDownloader:
    """Tests for Google Fonts downloader.

    This is a stub. Will test:
    - Downloads at least N fonts
    - Filters to Latin-script fonts with A-Z, a-z, 0-9 coverage
    - Skips already-downloaded fonts (incremental)
    - Produces expected directory structure
    """

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_download_creates_font_files(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_download_incremental_skips_existing(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_download_filters_latin_fonts(self):
        pass


@pytest.mark.data
class TestGlyphExtractor:
    """Tests for glyph extraction from font files.

    This is a stub. Will test:
    - Extracts all 62 characters (A-Z, a-z, 0-9) from each font
    - Renders to correct image sizes (64x64, 128x128)
    - Extracts bezier outlines as JSON
    - Extracts font metrics (UPM, ascender, descender, x-height)
    - Extracts kerning tables (GPOS + kern)
    """

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_extract_produces_glyph_images(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_extract_glyph_image_dimensions(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_extract_bezier_outlines(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_extract_font_metrics(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_extract_kerning_pairs(self):
        pass


@pytest.mark.data
class TestDatasetPreparation:
    """Tests for dataset manifest building and train/val/test splitting.

    This is a stub. Will test:
    - Produces valid JSONL manifests for all three model types
    - Train/val/test split is 80/10/10 by font family
    - No font family appears in multiple splits (no data leakage)
    - Manifest entries have all required fields
    """

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_manifest_schema_glyph(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_manifest_schema_style(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_manifest_schema_kerning(self):
        pass

    @pytest.mark.skip(reason="Stub: data pipeline not yet implemented (Phase 7B)")
    def test_split_no_font_leakage(self):
        pass
