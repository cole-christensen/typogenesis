"""Contrastive loss functions for StyleEncoder training.

This module implements various contrastive loss functions for training
the StyleEncoder to produce similar embeddings for same-font glyphs
and dissimilar embeddings for different-font glyphs.

Implemented losses:
    - NT-Xent (Normalized Temperature-scaled Cross Entropy) - SimCLR style
    - Triplet Loss with optional hard negative mining
    - InfoNCE (Noise Contrastive Estimation)

Example:
    >>> from scripts.models.style_encoder.losses import NTXentLoss, TripletLoss
    >>>
    >>> # NT-Xent loss for SimCLR-style training
    >>> criterion = NTXentLoss(temperature=0.07)
    >>> loss = criterion(z_i, z_j)  # Two augmented views
    >>>
    >>> # Triplet loss
    >>> criterion = TripletLoss(margin=0.5, hard_negative_mining=True)
    >>> loss = criterion(anchor, positive, negative)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type alias for clarity
Tensor = torch.Tensor


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    The NT-Xent loss is used in SimCLR and treats the pair of augmented
    views from the same image as positive pairs, while all other pairs
    in the batch are treated as negative pairs.

    For a batch of N samples with 2 views each (2N total):
    - Each sample has 1 positive pair (its other augmented view)
    - Each sample has 2(N-1) negative pairs (all other samples' views)

    Loss = -log(exp(sim(z_i, z_j)/tau) / sum(exp(sim(z_i, z_k)/tau)))

    where z_i and z_j are the two views of the same sample.

    Attributes:
        temperature: Temperature parameter tau for scaling similarities
        reduction: How to reduce the batch loss ("mean", "sum", or "none")
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
    ) -> None:
        """Initialize NT-Xent loss.

        Args:
            temperature: Temperature parameter for scaling. Lower values make
                        the model focus more on hard negatives.
            reduction: Reduction method for batch loss.

        Raises:
            ValueError: If temperature is not positive or reduction is invalid.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute NT-Xent loss.

        Args:
            z_i: First augmented view embeddings, shape (batch_size, embedding_dim)
            z_j: Second augmented view embeddings, shape (batch_size, embedding_dim)
                 z_i[k] and z_j[k] should be from the same original sample.

        Returns:
            NT-Xent loss value.
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenate both views: [z_i; z_j] -> (2*batch_size, embedding_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix: (2*batch_size, 2*batch_size)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Create mask for positive pairs
        # For sample k: positive pairs are (k, k+batch_size) and (k+batch_size, k)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

        # Labels: for sample k in first half, positive is at k+batch_size
        #         for sample k in second half, positive is at k-batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])

        # Mask out self-similarities
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Compute cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)

        return loss


class TripletLoss(nn.Module):
    """Triplet Loss with optional hard negative mining.

    Triplet loss encourages the anchor-positive distance to be smaller
    than the anchor-negative distance by at least a margin.

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    With hard negative mining, we select the hardest negative for each
    anchor-positive pair (the negative closest to the anchor).

    Attributes:
        margin: Minimum margin between positive and negative distances
        hard_negative_mining: Whether to use hard negative mining
        reduction: How to reduce the batch loss
    """

    def __init__(
        self,
        margin: float = 0.5,
        hard_negative_mining: bool = True,
        reduction: str = "mean",
    ) -> None:
        """Initialize Triplet loss.

        Args:
            margin: Margin value for triplet loss.
            hard_negative_mining: If True, mine hard negatives from the batch.
            reduction: Reduction method for batch loss.

        Raises:
            ValueError: If margin is not positive or reduction is invalid.
        """
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be positive, got {margin}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.margin = margin
        self.hard_negative_mining = hard_negative_mining
        self.reduction = reduction

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute triplet loss.

        Can be used in two modes:
        1. Explicit triplets: Provide anchor, positive, negative directly
        2. Batch mining: Provide anchor, positive, labels for hard mining

        Args:
            anchor: Anchor embeddings, shape (batch_size, embedding_dim)
            positive: Positive embeddings, shape (batch_size, embedding_dim)
            negative: Negative embeddings, shape (batch_size, embedding_dim).
                     If None and hard_negative_mining is True, negatives are
                     mined from the batch using labels.
            labels: Font labels for each sample, shape (batch_size,).
                   Required if negative is None and hard_negative_mining is True.

        Returns:
            Triplet loss value.

        Raises:
            ValueError: If neither negative nor labels provided when needed.
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        if negative is not None:
            # Explicit triplets provided
            negative = F.normalize(negative, p=2, dim=1)

            # Compute distances (1 - cosine_sim for normalized vectors = euclidean^2 / 2)
            pos_dist = (anchor - positive).pow(2).sum(dim=1)
            neg_dist = (anchor - negative).pow(2).sum(dim=1)

        elif self.hard_negative_mining and labels is not None:
            # Mine hard negatives from batch
            negative, neg_dist = self._mine_hard_negatives(anchor, labels)
            pos_dist = (anchor - positive).pow(2).sum(dim=1)

        else:
            raise ValueError(
                "Must provide either 'negative' embeddings or 'labels' for hard mining"
            )

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _mine_hard_negatives(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Mine hard negatives from the batch.

        For each sample, finds the closest sample from a different class
        (semi-hard or hard negative).

        Args:
            embeddings: Embeddings, shape (batch_size, embedding_dim)
            labels: Class labels, shape (batch_size,)

        Returns:
            Tuple of (hard_negatives, distances):
                - hard_negatives: Hard negative embeddings, shape (batch_size, embedding_dim)
                - distances: Distances to hard negatives, shape (batch_size,)
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Create mask for valid negatives (different labels)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~labels_equal

        # Set invalid pairs to large distance
        dist_matrix = dist_matrix.masked_fill(~neg_mask, float("inf"))

        # Find hard negatives (minimum distance among valid negatives)
        min_dist, min_indices = dist_matrix.min(dim=1)

        # Handle case where no valid negatives exist (all same label)
        no_negatives = min_dist == float("inf")
        if no_negatives.any():
            # Fall back to random negatives for those samples
            for i in torch.where(no_negatives)[0]:
                valid_indices = torch.where(neg_mask[i])[0]
                if len(valid_indices) > 0:
                    rand_idx = valid_indices[torch.randint(len(valid_indices), (1,))]
                    min_indices[i] = rand_idx
                    min_dist[i] = dist_matrix[i, rand_idx]

        hard_negatives = embeddings[min_indices]
        return hard_negatives, min_dist.pow(2)  # Return squared distance


class InfoNCELoss(nn.Module):
    """InfoNCE (Noise Contrastive Estimation) Loss.

    InfoNCE loss estimates mutual information between views by contrasting
    positive pairs against K negative samples. It's similar to NT-Xent but
    can handle asymmetric batch structures.

    Loss = -log(exp(sim(q, k+)/tau) / (exp(sim(q, k+)/tau) + sum(exp(sim(q, k-)/tau))))

    where q is the query, k+ is the positive key, and k- are negative keys.

    Attributes:
        temperature: Temperature parameter for scaling similarities
        reduction: How to reduce the batch loss
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
    ) -> None:
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling.
            reduction: Reduction method for batch loss.

        Raises:
            ValueError: If temperature is not positive or reduction is invalid.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        query: Tensor,
        positive_key: Tensor,
        negative_keys: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute InfoNCE loss.

        Args:
            query: Query embeddings, shape (batch_size, embedding_dim)
            positive_key: Positive key embeddings, shape (batch_size, embedding_dim)
                         query[k] and positive_key[k] form a positive pair.
            negative_keys: Negative key embeddings, shape (num_negatives, embedding_dim).
                          If None, uses all other samples in batch as negatives.

        Returns:
            InfoNCE loss value.
        """
        batch_size = query.shape[0]
        device = query.device

        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)

        # Positive similarities: (batch_size,)
        pos_sim = (query * positive_key).sum(dim=1) / self.temperature

        if negative_keys is None:
            # Use all other samples in batch as negatives
            # Negative similarities: (batch_size, batch_size)
            neg_sim = torch.mm(query, positive_key.t()) / self.temperature

            # Mask out positive pairs (diagonal)
            mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            neg_sim = neg_sim.masked_fill(mask, float("-inf"))

            # Log-sum-exp over negatives
            neg_logsumexp = torch.logsumexp(neg_sim, dim=1)

        else:
            # Use provided negative keys
            negative_keys = F.normalize(negative_keys, p=2, dim=1)

            # Negative similarities: (batch_size, num_negatives)
            neg_sim = torch.mm(query, negative_keys.t()) / self.temperature

            # Log-sum-exp over negatives
            neg_logsumexp = torch.logsumexp(neg_sim, dim=1)

        # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_logsumexp)))
        # = -pos_sim + log(exp(pos_sim) + exp(neg_logsumexp))
        # = -pos_sim + logsumexp(pos_sim, neg_logsumexp)
        total_logsumexp = torch.logsumexp(
            torch.stack([pos_sim, neg_logsumexp], dim=1), dim=1
        )
        loss = -pos_sim + total_logsumexp

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Extends contrastive learning to leverage label information, where all
    samples from the same class are treated as positives.

    Loss = sum_{i} -1/|P(i)| * sum_{p in P(i)} log(exp(sim(i,p)/tau) / sum_{a in A(i)} exp(sim(i,a)/tau))

    where P(i) is the set of positives for sample i (same class) and A(i) is
    all samples except i.

    Attributes:
        temperature: Temperature parameter for scaling similarities
        base_temperature: Base temperature for normalization (usually same as temperature)
        reduction: How to reduce the batch loss
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        reduction: str = "mean",
    ) -> None:
        """Initialize Supervised Contrastive loss.

        Args:
            temperature: Temperature parameter for scaling.
            base_temperature: Base temperature for loss normalization.
            reduction: Reduction method for batch loss.

        Raises:
            ValueError: If temperatures are not positive or reduction is invalid.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if base_temperature <= 0:
            raise ValueError(f"base_temperature must be positive, got {base_temperature}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: Embeddings, shape (batch_size, embedding_dim) or
                     (batch_size, num_views, embedding_dim) for multi-view.
            labels: Class labels, shape (batch_size,)

        Returns:
            Supervised contrastive loss value.
        """
        device = features.device

        if features.dim() == 3:
            # Multi-view: (batch_size, num_views, embedding_dim)
            batch_size, num_views = features.shape[:2]
            features = features.view(batch_size * num_views, -1)
            labels = labels.repeat_interleave(num_views)
        else:
            batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        # Create mask for positive pairs (same label, excluding self)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = torch.eye(len(labels), dtype=torch.bool, device=device)
        pos_mask = labels_equal & ~self_mask

        # Count positives per sample
        pos_count = pos_mask.sum(dim=1)

        # Mask out self-similarities
        sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))

        # Log-softmax over all non-self samples
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

        # Mean log-probability over positive pairs
        # Handle samples with no positives (set loss to 0)
        loss = -(pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)

        # Scale by temperature ratio
        loss = loss * (self.temperature / self.base_temperature)

        # Zero out loss for samples with no positives
        loss = loss.masked_fill(pos_count == 0, 0)

        if self.reduction == "mean":
            # Mean over samples with positives
            return loss.sum() / (pos_count > 0).sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """Factory function to create loss function by name.

    Args:
        loss_type: Type of loss function. Supported: "nt_xent", "triplet",
                  "infonce", "supcon"
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        Initialized loss module.

    Raises:
        ValueError: If unsupported loss type is specified.

    Example:
        >>> loss_fn = get_loss_function("nt_xent", temperature=0.1)
        >>> loss_fn = get_loss_function("triplet", margin=0.3)
    """
    loss_types = {
        "nt_xent": NTXentLoss,
        "triplet": TripletLoss,
        "infonce": InfoNCELoss,
        "supcon": SupConLoss,
    }

    if loss_type not in loss_types:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. "
            f"Supported: {list(loss_types.keys())}"
        )

    return loss_types[loss_type](**kwargs)
