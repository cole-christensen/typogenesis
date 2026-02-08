"""Training smoke tests: single-step training for all three models.

Verifies that:
- Loss computation works
- Backward pass produces gradients
- Optimizer step decreases loss (or at least doesn't crash)
"""

import torch
import torch.nn as nn


class TestGlyphDiffusionTraining:
    """Smoke tests for GlyphDiffusion training loop."""

    def test_single_training_step(self, batch_size, image_size):
        """One training step completes without errors and loss is finite."""
        from models.glyph_diffusion import (
            FlowMatchingLoss,
            FlowMatchingSchedule,
            GlyphDiffusionModel,
            ModelConfig,
            prepare_training_batch,
        )

        torch.manual_seed(42)
        config = ModelConfig()
        model = GlyphDiffusionModel(config)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = FlowMatchingLoss()
        schedule = FlowMatchingSchedule()

        # Synthetic batch
        x_0 = torch.randn(batch_size, 1, image_size, image_size)
        char_indices = torch.randint(0, 62, (batch_size,))
        style_embed = torch.randn(batch_size, 128)

        # Prepare noisy input and target
        x_t, timesteps, noise, target_velocity = prepare_training_batch(
            x_0, schedule, torch.device("cpu")
        )

        # Forward pass
        predicted_velocity = model(x_t, timesteps, char_indices, style_embed)
        loss = loss_fn(predicted_velocity, target_velocity)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0, "Loss should be positive for random inputs"

    def test_loss_decreases_over_steps(self, image_size):
        """Loss decreases after multiple training steps on same batch."""
        from models.glyph_diffusion import (
            FlowMatchingLoss,
            FlowMatchingSchedule,
            GlyphDiffusionModel,
            ModelConfig,
            prepare_training_batch,
        )

        torch.manual_seed(42)
        batch_size = 8
        model = GlyphDiffusionModel(ModelConfig())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = FlowMatchingLoss()
        schedule = FlowMatchingSchedule()

        x_0 = torch.randn(batch_size, 1, image_size, image_size)
        char_indices = torch.randint(0, 62, (batch_size,))
        style_embed = torch.randn(batch_size, 128)

        losses = []
        for _ in range(10):
            x_t, timesteps, noise, target_velocity = prepare_training_batch(
                x_0, schedule, torch.device("cpu")
            )
            predicted = model(x_t, timesteps, char_indices, style_embed)
            loss = loss_fn(predicted, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (last < first)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


class TestStyleEncoderTraining:
    """Smoke tests for StyleEncoder contrastive training."""

    def test_single_training_step_ntxent(self, batch_size):
        """One training step with NT-Xent loss completes without errors."""
        from models.style_encoder import NTXentLoss, StyleEncoder, StyleEncoderConfig

        torch.manual_seed(42)
        config = StyleEncoderConfig(pretrained=False)
        model = StyleEncoder(config)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss_fn = NTXentLoss(temperature=0.07)

        # NT-Xent expects two views: z_i and z_j from the same samples
        # Simulate by creating two augmented views of the same font glyphs
        view_i = torch.randn(batch_size, 1, 64, 64)
        view_j = view_i + torch.randn_like(view_i) * 0.1  # Slightly augmented

        # Forward pass with projection for both views
        _, proj_i = model(view_i, return_projection=True)
        _, proj_j = model(view_j, return_projection=True)
        loss = loss_fn(proj_i, proj_j)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_decreases_over_steps(self):
        """Loss decreases after multiple training steps."""
        from models.style_encoder import NTXentLoss, StyleEncoder, StyleEncoderConfig

        torch.manual_seed(42)
        batch_size = 8
        config = StyleEncoderConfig(pretrained=False)
        model = StyleEncoder(config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = NTXentLoss(temperature=0.07)

        # Two views of the same font glyphs (SimCLR-style)
        view_i = torch.randn(batch_size, 1, 64, 64)
        view_j = view_i + torch.randn_like(view_i) * 0.1

        losses = []
        for _ in range(10):
            _, proj_i = model(view_i, return_projection=True)
            _, proj_j = model(view_j, return_projection=True)
            loss = loss_fn(proj_i, proj_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


class TestKerningNetTraining:
    """Smoke tests for KerningNet training."""

    def test_single_training_step(self, batch_size, image_size):
        """One training step completes without errors."""
        from models.kerning_net import KerningNet, KerningNetConfig

        torch.manual_seed(42)
        model = KerningNet(KerningNetConfig())
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss_fn = nn.SmoothL1Loss()

        left = torch.randn(batch_size, 1, image_size, image_size)
        right = torch.randn(batch_size, 1, image_size, image_size)
        target_kerning = torch.randn(batch_size, 1) * 0.1  # Small kerning values

        predicted = model(left, right)
        loss = loss_fn(predicted, target_kerning)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_decreases_over_steps(self, image_size):
        """Loss decreases after multiple training steps on same batch."""
        from models.kerning_net import KerningNet, KerningNetConfig

        torch.manual_seed(42)
        batch_size = 16
        model = KerningNet(KerningNetConfig())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.SmoothL1Loss()

        left = torch.randn(batch_size, 1, image_size, image_size)
        right = torch.randn(batch_size, 1, image_size, image_size)
        target = torch.randn(batch_size, 1) * 0.1

        losses = []
        for _ in range(20):
            predicted = model(left, right)
            loss = loss_fn(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
