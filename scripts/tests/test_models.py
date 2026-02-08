"""Tests for model architectures: forward pass, output shapes, gradient flow.

Tests all three models:
- GlyphDiffusionModel: Flow-matching UNet
- StyleEncoder: ResNet-18 contrastive encoder
- KerningNet: Siamese CNN regressor
"""

import torch


class TestGlyphDiffusionModel:
    """Tests for GlyphDiffusionModel forward pass and architecture."""

    def test_forward_pass_default_config(
        self, glyph_diffusion_model, synthetic_glyphs, timesteps, char_indices, style_embeddings
    ):
        """Model produces output with same shape as input."""
        with torch.no_grad():
            output = glyph_diffusion_model(
                synthetic_glyphs, timesteps, char_indices, style_embeddings
            )
        assert output.shape == synthetic_glyphs.shape

    def test_forward_pass_with_mask(
        self, glyph_diffusion_model, synthetic_glyphs, timesteps, char_indices, style_embeddings
    ):
        """Model accepts optional mask conditioning."""
        mask = torch.zeros_like(synthetic_glyphs)
        with torch.no_grad():
            output = glyph_diffusion_model(
                synthetic_glyphs, timesteps, char_indices, style_embeddings, mask=mask
            )
        assert output.shape == synthetic_glyphs.shape

    def test_output_shape_batch_1(self, glyph_diffusion_model):
        """Model works with batch size 1."""
        x = torch.randn(1, 1, 64, 64)
        t = torch.rand(1)
        c = torch.randint(0, 62, (1,))
        s = torch.randn(1, 128)
        with torch.no_grad():
            output = glyph_diffusion_model(x, t, c, s)
        assert output.shape == (1, 1, 64, 64)

    def test_gradient_flow(self, synthetic_glyphs, timesteps, char_indices, style_embeddings):
        """Gradients flow through all parameters."""
        from models.glyph_diffusion import GlyphDiffusionModel, ModelConfig

        model = GlyphDiffusionModel(ModelConfig())
        model.train()
        output = model(synthetic_glyphs, timesteps, char_indices, style_embeddings)
        loss = output.sum()
        loss.backward()

        params_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        # At least 80% of parameters should receive gradients
        assert params_with_grad > total_params * 0.8, (
            f"Only {params_with_grad}/{total_params} parameters received gradients"
        )

    def test_parameter_count(self, glyph_diffusion_model):
        """Model has expected parameter count (~17.5M)."""
        num_params = glyph_diffusion_model.num_parameters()
        assert num_params > 1_000_000, f"Model too small: {num_params:,} params"
        assert num_params < 100_000_000, f"Model too large: {num_params:,} params"

    def test_deterministic_output(
        self, glyph_diffusion_model, synthetic_glyphs, timesteps, char_indices, style_embeddings
    ):
        """Same inputs produce same outputs in eval mode."""
        with torch.no_grad():
            out1 = glyph_diffusion_model(
                synthetic_glyphs, timesteps, char_indices, style_embeddings
            )
            out2 = glyph_diffusion_model(
                synthetic_glyphs, timesteps, char_indices, style_embeddings
            )
        assert torch.allclose(out1, out2, atol=1e-6)


class TestStyleEncoder:
    """Tests for StyleEncoder forward pass and architecture."""

    def test_forward_pass_embedding_only(self, style_encoder_model, synthetic_glyphs):
        """Model produces 128-dim embeddings."""
        with torch.no_grad():
            embedding = style_encoder_model(synthetic_glyphs)
        assert embedding.shape == (synthetic_glyphs.shape[0], 128)

    def test_forward_pass_with_projection(self, style_encoder_model, synthetic_glyphs):
        """Model returns both embedding and projection when requested."""
        with torch.no_grad():
            embedding, projection = style_encoder_model(
                synthetic_glyphs, return_projection=True
            )
        assert embedding.shape == (synthetic_glyphs.shape[0], 128)
        assert projection.shape == (synthetic_glyphs.shape[0], 256)

    def test_embedding_normalized(self, style_encoder_model, synthetic_glyphs):
        """Embeddings are L2-normalized (unit vectors)."""
        with torch.no_grad():
            embedding = style_encoder_model(synthetic_glyphs)
        norms = torch.norm(embedding, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_batch_size_1(self, style_encoder_model):
        """Model works with batch size 1."""
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            embedding = style_encoder_model(x)
        assert embedding.shape == (1, 128)

    def test_gradient_flow(self, synthetic_glyphs):
        """Gradients flow through all parameters."""
        from models.style_encoder import StyleEncoder, StyleEncoderConfig

        model = StyleEncoder(StyleEncoderConfig(pretrained=False))
        model.train()
        embedding, projection = model(synthetic_glyphs, return_projection=True)
        loss = projection.sum()
        loss.backward()

        params_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert params_with_grad > total_params * 0.8

    def test_parameter_count(self, style_encoder_model):
        """Model has expected parameter count (~11.7M)."""
        num_params = sum(p.numel() for p in style_encoder_model.parameters())
        assert num_params > 1_000_000, f"Model too small: {num_params:,} params"
        assert num_params < 50_000_000, f"Model too large: {num_params:,} params"

    def test_encode_method(self, style_encoder_model, synthetic_glyphs):
        """encode() convenience method works the same as forward."""
        with torch.no_grad():
            embedding_forward = style_encoder_model(synthetic_glyphs)
            embedding_encode = style_encoder_model.encode(synthetic_glyphs)
        assert torch.allclose(embedding_forward, embedding_encode, atol=1e-6)


class TestKerningNet:
    """Tests for KerningNet forward pass and architecture."""

    def test_forward_pass(self, kerning_net_model, batch_size, image_size):
        """Model produces scalar kerning predictions."""
        left = torch.randn(batch_size, 1, image_size, image_size)
        right = torch.randn(batch_size, 1, image_size, image_size)
        with torch.no_grad():
            kerning = kerning_net_model(left, right)
        assert kerning.shape == (batch_size, 1)

    def test_batch_size_1(self, kerning_net_model):
        """Model works with batch size 1."""
        left = torch.randn(1, 1, 64, 64)
        right = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            kerning = kerning_net_model(left, right)
        assert kerning.shape == (1, 1)

    def test_shared_encoder(self, kerning_net_model):
        """Left and right glyphs use the same encoder (Siamese)."""
        left = torch.randn(2, 1, 64, 64)
        right = left.clone()  # Same images
        with torch.no_grad():
            left_feat = kerning_net_model.encoder(left)
            right_feat = kerning_net_model.encoder(right)
        assert torch.allclose(left_feat, right_feat, atol=1e-6)

    def test_gradient_flow(self, batch_size, image_size):
        """Gradients flow through all parameters."""
        from models.kerning_net import KerningNet, KerningNetConfig

        model = KerningNet(KerningNetConfig())
        model.train()
        left = torch.randn(batch_size, 1, image_size, image_size)
        right = torch.randn(batch_size, 1, image_size, image_size)
        kerning = model(left, right)
        loss = kerning.sum()
        loss.backward()

        params_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert params_with_grad > total_params * 0.8

    def test_parameter_count(self, kerning_net_model):
        """Model has expected parameter count (~1-2M)."""
        num_params = kerning_net_model.num_parameters()
        assert num_params > 100_000, f"Model too small: {num_params:,} params"
        assert num_params < 10_000_000, f"Model too large: {num_params:,} params"

    def test_deterministic_output(self, kerning_net_model):
        """Same inputs produce same outputs in eval mode."""
        left = torch.randn(2, 1, 64, 64)
        right = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            out1 = kerning_net_model(left, right)
            out2 = kerning_net_model(left, right)
        assert torch.allclose(out1, out2, atol=1e-6)
