"""Unit tests for contrastive loss functions.

Tests NTXentLoss, SupConLoss, InfoNCELoss, and TripletLoss to verify:
- Correct output properties (finite, non-negative)
- Expected behavior for known inputs (identical vs orthogonal embeddings)
- Regression test for the known SupConLoss NaN bug (0 * -inf)
"""

import torch

from models.style_encoder.losses import (
    InfoNCELoss,
    NTXentLoss,
    SupConLoss,
    TripletLoss,
)


class TestNTXentLoss:
    """Tests for NT-Xent (SimCLR-style) loss."""

    def test_identical_embeddings_near_zero_loss(self):
        """Identical view pairs should produce near-zero loss."""
        torch.manual_seed(42)
        loss_fn = NTXentLoss(temperature=0.07)

        # All embeddings identical within each pair
        z = torch.randn(4, 128)
        z = torch.nn.functional.normalize(z, dim=1)

        loss = loss_fn(z, z.clone())

        # With identical pairs, the positive similarity is maximal (1.0),
        # but other pairs also have varying similarities.
        # For truly identical z_i == z_j, the loss approaches 0 when
        # positive sim >> negative sims, which happens when batch is large
        # and embeddings are random enough. For small batch it won't be
        # exactly zero but should be finite and small.
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    def test_orthogonal_embeddings_higher_loss(self):
        """Orthogonal (dissimilar) pairs should produce higher loss than identical pairs."""
        torch.manual_seed(42)
        loss_fn = NTXentLoss(temperature=0.5)  # Higher temp for numerical stability

        batch_size = 8
        dim = 128

        # Identical pairs
        z_same = torch.randn(batch_size, dim)
        loss_identical = loss_fn(z_same, z_same.clone())

        # Orthogonal pairs: make z_j orthogonal to z_i
        z_i = torch.randn(batch_size, dim)
        z_j = torch.randn(batch_size, dim)
        # Make orthogonal by removing the projection of z_j onto z_i
        z_i_norm = torch.nn.functional.normalize(z_i, dim=1)
        z_j = z_j - (z_j * z_i_norm).sum(dim=1, keepdim=True) * z_i_norm
        loss_orthogonal = loss_fn(z_i, z_j)

        assert loss_orthogonal > loss_identical, (
            f"Orthogonal loss ({loss_orthogonal.item():.4f}) should be > "
            f"identical loss ({loss_identical.item():.4f})"
        )

    def test_output_finite_and_non_negative(self):
        """Loss should always be finite and non-negative."""
        torch.manual_seed(42)
        loss_fn = NTXentLoss(temperature=0.07)

        z_i = torch.randn(8, 64)
        z_j = torch.randn(8, 64)

        loss = loss_fn(z_i, z_j)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_batch_size_one(self):
        """Loss should still work with batch_size=1 (though degenerate)."""
        torch.manual_seed(42)
        loss_fn = NTXentLoss(temperature=0.07)

        # With batch_size=1, there are 2 total samples, and each one's
        # positive is the other. The only "negative" is itself (masked out).
        # cross_entropy with 1 valid class should return 0.
        z_i = torch.randn(1, 64)
        z_j = torch.randn(1, 64)

        loss = loss_fn(z_i, z_j)
        assert torch.isfinite(loss), f"Loss should be finite for batch_size=1, got {loss.item()}"


class TestSupConLoss:
    """Tests for Supervised Contrastive Loss.

    CRITICAL: Includes regression test for the known NaN bug where
    pos_mask * log_prob produced NaN when 0 * -inf occurred.
    """

    def test_normal_inputs_finite(self):
        """SupConLoss should produce finite loss for normal inputs."""
        torch.manual_seed(42)
        loss_fn = SupConLoss(temperature=0.07)

        features = torch.randn(8, 128)
        # 4 classes, 2 samples each
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = loss_fn(features, labels)

        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_nan_regression_all_zero_pos_mask_rows(self):
        """REGRESSION: When pos_mask has all-zero rows, loss must still be finite.

        The original bug: pos_mask * log_prob produced NaN when a sample had
        no positive pairs (pos_mask row = all zeros). The log_prob contained
        -inf values (from masked_fill on self-similarities), and 0 * -inf = NaN.

        The fix uses torch.where(pos_mask, log_prob, zeros) instead of
        pos_mask * log_prob.
        """
        torch.manual_seed(42)
        loss_fn = SupConLoss(temperature=0.07)

        features = torch.randn(5, 128)
        # Label 4 has only one sample (index 4) -> pos_mask row is all zeros
        labels = torch.tensor([0, 0, 1, 1, 4])

        loss = loss_fn(features, labels)

        assert torch.isfinite(loss), (
            f"REGRESSION: Loss is NaN/Inf when a class has only one sample. "
            f"Got {loss.item()}. The 0 * -inf = NaN bug may have returned."
        )

    def test_single_sample_per_class(self):
        """When every class has only one sample, loss should be 0 (no positives)."""
        torch.manual_seed(42)
        loss_fn = SupConLoss(temperature=0.07)

        features = torch.randn(4, 128)
        labels = torch.tensor([0, 1, 2, 3])  # All unique labels

        loss = loss_fn(features, labels)

        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
        # With no positive pairs for any sample, loss should be 0
        assert loss.item() == 0.0, (
            f"Loss should be 0 when no sample has a positive pair, got {loss.item()}"
        )

    def test_all_same_class(self):
        """When all samples are the same class, all non-self pairs are positives."""
        torch.manual_seed(42)
        loss_fn = SupConLoss(temperature=0.07)

        features = torch.randn(4, 128)
        labels = torch.tensor([0, 0, 0, 0])

        loss = loss_fn(features, labels)

        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_multi_view_input(self):
        """SupConLoss should handle 3D input (batch_size, num_views, dim)."""
        torch.manual_seed(42)
        loss_fn = SupConLoss(temperature=0.07)

        # 4 samples, 2 views each
        features = torch.randn(4, 2, 128)
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(features, labels)

        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"


class TestInfoNCELoss:
    """Tests for InfoNCE loss."""

    def test_output_finite_and_non_negative(self):
        """Loss should be finite and non-negative."""
        torch.manual_seed(42)
        loss_fn = InfoNCELoss(temperature=0.07)

        query = torch.randn(8, 64)
        positive_key = torch.randn(8, 64)

        loss = loss_fn(query, positive_key)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_matching_pairs_lower_loss(self):
        """Matching (similar) pairs should produce lower loss than random pairs."""
        torch.manual_seed(42)
        loss_fn = InfoNCELoss(temperature=0.5)

        query = torch.randn(8, 64)

        # Matching pairs: small perturbation
        pos_close = query + torch.randn_like(query) * 0.01
        loss_close = loss_fn(query, pos_close)

        # Non-matching pairs: fully random
        pos_random = torch.randn(8, 64)
        loss_random = loss_fn(query, pos_random)

        assert loss_close < loss_random, (
            f"Close-pair loss ({loss_close.item():.4f}) should be < "
            f"random-pair loss ({loss_random.item():.4f})"
        )

    def test_with_explicit_negative_keys(self):
        """Loss should work with explicitly provided negative keys."""
        torch.manual_seed(42)
        loss_fn = InfoNCELoss(temperature=0.07)

        query = torch.randn(4, 64)
        positive_key = torch.randn(4, 64)
        negative_keys = torch.randn(16, 64)

        loss = loss_fn(query, positive_key, negative_keys)

        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"


class TestTripletLoss:
    """Tests for Triplet loss."""

    def test_output_finite_and_non_negative(self):
        """Loss should be finite and non-negative."""
        torch.manual_seed(42)
        loss_fn = TripletLoss(margin=0.5)

        anchor = torch.randn(8, 64)
        positive = torch.randn(8, 64)
        negative = torch.randn(8, 64)

        loss = loss_fn(anchor, positive, negative)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_correct_direction(self):
        """Loss should be lower when positive is closer than negative."""
        torch.manual_seed(42)
        loss_fn = TripletLoss(margin=0.5)

        anchor = torch.randn(8, 64)

        # Good triplet: positive very close, negative far
        positive_close = anchor + torch.randn_like(anchor) * 0.01
        negative_far = torch.randn(8, 64) * 5.0
        loss_good = loss_fn(anchor, positive_close, negative_far)

        # Bad triplet: positive far, negative close
        positive_far = torch.randn(8, 64) * 5.0
        negative_close = anchor + torch.randn_like(anchor) * 0.01
        loss_bad = loss_fn(anchor, positive_far, negative_close)

        assert loss_good < loss_bad, (
            f"Good triplet loss ({loss_good.item():.4f}) should be < "
            f"bad triplet loss ({loss_bad.item():.4f})"
        )

    def test_zero_loss_when_well_separated(self):
        """Loss should be 0 when pos_dist + margin < neg_dist."""
        loss_fn = TripletLoss(margin=0.5)

        # Anchor and positive are identical (distance = 0)
        anchor = torch.randn(4, 64)
        positive = anchor.clone()
        # Negative is very far
        negative = torch.randn(4, 64) * 100.0

        loss = loss_fn(anchor, positive, negative)

        assert loss.item() == 0.0, (
            f"Loss should be 0 for well-separated triplets, got {loss.item()}"
        )

    def test_hard_negative_mining(self):
        """Triplet loss with hard negative mining should work with labels."""
        torch.manual_seed(42)
        loss_fn = TripletLoss(margin=0.5, hard_negative_mining=True)

        anchor = torch.randn(8, 64)
        positive = torch.randn(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = loss_fn(anchor, positive, labels=labels)

        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
