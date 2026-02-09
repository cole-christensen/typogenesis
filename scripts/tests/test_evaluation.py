"""Tests for evaluation metrics and visualization.

Tests the actual metric computation functions with synthetic data.
"""

import torch
from PIL import Image


class TestStyleEncoderMetrics:
    """Tests for StyleEncoder evaluation metrics."""

    def test_same_font_similarity_high_for_identical(self):
        """Same-font similarity is 1.0 for identical embeddings per font."""
        from evaluation.metrics import same_font_similarity

        # Two fonts, 3 glyphs each, identical within font
        embed_a = torch.randn(1, 128)
        embed_a = embed_a / embed_a.norm()
        embed_b = torch.randn(1, 128)
        embed_b = embed_b / embed_b.norm()

        embeddings = torch.cat([
            embed_a.expand(3, -1),  # Font 0: 3 identical glyphs
            embed_b.expand(3, -1),  # Font 1: 3 identical glyphs
        ])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        sim = same_font_similarity(embeddings, labels)
        assert sim > 0.99, f"Expected ~1.0 for identical embeddings, got {sim}"

    def test_cross_font_similarity_low_for_orthogonal(self):
        """Cross-font similarity is near 0 for orthogonal embeddings."""
        from evaluation.metrics import cross_font_similarity

        torch.manual_seed(42)
        # Create 2 fonts with very different embeddings
        embeddings = torch.randn(6, 128)
        # Make font 0 and font 1 embeddings quite different
        embeddings[:3] = torch.randn(1, 128).expand(3, -1) + torch.randn(3, 128) * 0.01
        embeddings[3:] = -embeddings[:3]  # Opposite direction
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        sim = cross_font_similarity(embeddings, labels)
        # Opposite embeddings should have negative similarity
        assert sim < 0.0, f"Expected negative for opposite embeddings, got {sim}"

    def test_retrieval_accuracy_perfect_clustering(self):
        """Retrieval accuracy is 1.0 when fonts are perfectly clustered."""
        from evaluation.metrics import retrieval_accuracy

        # Create 3 fonts with very distinct embeddings
        embeddings = torch.zeros(9, 128)
        embeddings[0:3, 0] = 1.0   # Font 0 in dimension 0
        embeddings[3:6, 1] = 1.0   # Font 1 in dimension 1
        embeddings[6:9, 2] = 1.0   # Font 2 in dimension 2
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

        result = retrieval_accuracy(embeddings, labels, top_k=(1, 5))
        assert result["top_1"] == 1.0
        assert result["top_5"] == 1.0


class TestKerningNetMetrics:
    """Tests for KerningNet evaluation metrics."""

    def test_kerning_mae_zero_for_perfect(self):
        """MAE is 0 when predictions exactly match targets."""
        from evaluation.metrics import kerning_mae

        preds = torch.tensor([0.1, -0.2, 0.0, 0.3])
        targets = preds.clone()

        mae = kerning_mae(preds, targets, max_kerning=200.0)
        assert mae < 0.01, f"Expected ~0 MAE for perfect predictions, got {mae}"

    def test_kerning_mae_correct_scale(self):
        """MAE is correctly scaled to font units."""
        from evaluation.metrics import kerning_mae

        preds = torch.tensor([0.1])
        targets = torch.tensor([0.0])

        mae = kerning_mae(preds, targets, max_kerning=200.0)
        # Error is 0.1 * 200 = 20 font units
        assert abs(mae - 20.0) < 0.1, f"Expected 20.0 font units, got {mae}"

    def test_kerning_direction_accuracy_correct(self):
        """Direction accuracy is 1.0 when all signs match."""
        from evaluation.metrics import kerning_direction_accuracy

        preds = torch.tensor([-0.5, 0.3, -0.1, 0.0])
        targets = torch.tensor([-0.3, 0.1, -0.8, 0.0])

        acc = kerning_direction_accuracy(preds, targets)
        assert acc == 1.0, f"Expected 1.0 for matching signs, got {acc}"

    def test_kerning_direction_accuracy_wrong(self):
        """Direction accuracy catches wrong signs."""
        from evaluation.metrics import kerning_direction_accuracy

        preds = torch.tensor([0.5, -0.5])  # Both wrong sign
        targets = torch.tensor([-0.5, 0.5])

        acc = kerning_direction_accuracy(preds, targets)
        assert acc == 0.0, f"Expected 0.0 for all wrong signs, got {acc}"

    def test_critical_pair_mae(self):
        """Critical pair MAE correctly filters and computes."""
        from evaluation.metrics import critical_pair_mae

        preds = {("A", "V"): -0.3, ("A", "W"): -0.2, ("X", "Y"): 0.1}
        targets = {("A", "V"): -0.5, ("A", "W"): -0.2, ("X", "Y"): 0.0}

        mae = critical_pair_mae(preds, targets, max_kerning=200.0)
        # Only AV and AW are critical pairs
        # AV error: |(-0.3) - (-0.5)| * 200 = 40
        # AW error: |(-0.2) - (-0.2)| * 200 = 0
        # Mean: 20.0
        assert abs(mae - 20.0) < 0.1, f"Expected 20.0, got {mae}"


class TestGlyphDiffusionMetrics:
    """Tests for GlyphDiffusion evaluation metrics."""

    def test_fid_identical_distributions(self):
        """FID is ~0 for identical distributions."""
        from evaluation.metrics import compute_fid

        torch.manual_seed(42)
        features = torch.randn(100, 64)

        fid = compute_fid(features, features.clone())
        assert fid < 1.0, f"Expected FID ~0 for identical distributions, got {fid}"

    def test_fid_different_distributions(self):
        """FID is positive for different distributions."""
        from evaluation.metrics import compute_fid

        torch.manual_seed(42)
        real = torch.randn(100, 64)
        generated = torch.randn(100, 64) + 5.0  # Shifted mean

        fid = compute_fid(real, generated)
        assert fid > 10.0, f"Expected large FID for different distributions, got {fid}"


class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_sample_grid(self, tmp_path):
        """Sample grid creates a valid PNG file."""
        from evaluation.visualize import plot_sample_grid

        images = torch.randn(12, 1, 64, 64)
        labels = [chr(65 + i) for i in range(12)]
        output = tmp_path / "grid.png"

        plot_sample_grid(images, labels, output)
        assert output.exists()
        img = Image.open(output)
        assert img.size[0] > 0

    def test_plot_training_curve(self, tmp_path):
        """Training curve creates a valid PNG file."""
        from evaluation.visualize import plot_training_curve

        losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35]
        output = tmp_path / "curve.png"

        plot_training_curve(losses, output)
        assert output.exists()

    def test_plot_kerning_comparison(self, tmp_path):
        """Kerning comparison creates a valid PNG file."""
        from evaluation.visualize import plot_kerning_comparison

        preds = [-50.0, -30.0, 0.0, -20.0, 10.0]
        targets = [-60.0, -25.0, 0.0, -15.0, 5.0]
        labels = ["AV", "AW", "AB", "To", "AT"]
        output = tmp_path / "kerning.png"

        plot_kerning_comparison(preds, targets, labels, output)
        assert output.exists()

    def test_plot_embedding_space(self, tmp_path):
        """Embedding space plot creates a valid PNG file."""
        from evaluation.visualize import plot_embedding_space

        torch.manual_seed(42)
        embeddings = torch.randn(50, 128)
        labels = torch.arange(5).repeat(10)
        output = tmp_path / "embeddings.png"

        plot_embedding_space(embeddings, labels, output, method="pca")
        assert output.exists()
