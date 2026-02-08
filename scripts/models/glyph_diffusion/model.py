"""
UNet model for GlyphDiffusion with conditioning.

This module implements a UNet architecture for flow-matching diffusion,
with support for:
- Time embedding (sinusoidal)
- Character conditioning (learned embeddings)
- Style conditioning (128-dim vector via AdaIN)
- Optional mask conditioning for partial completion
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding as used in Transformers and diffusion models.

    Maps scalar timesteps to high-dimensional embeddings using sin/cos functions
    at different frequencies.
    """

    def __init__(self, embed_dim: int, max_period: int = 10000):
        """Initialize time embedding.

        Args:
            embed_dim: Dimension of the output embedding.
            max_period: Maximum period for sinusoidal frequencies.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute time embeddings.

        Args:
            timesteps: Tensor of shape (batch_size,) with timestep values in [0, 1].

        Returns:
            Embeddings of shape (batch_size, embed_dim).
        """
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        # Scale timesteps to [0, max_period]
        args = timesteps[:, None].float() * freqs[None, :] * self.max_period
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeEmbedMLP(nn.Module):
    """MLP that projects time embeddings to conditioning dimension."""

    def __init__(self, time_embed_dim: int, out_dim: int):
        """Initialize time embedding MLP.

        Args:
            time_embed_dim: Dimension of input time embedding.
            out_dim: Output dimension for conditioning.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_embed: torch.Tensor) -> torch.Tensor:
        """Project time embedding.

        Args:
            t_embed: Time embedding of shape (batch_size, time_embed_dim).

        Returns:
            Projected embedding of shape (batch_size, out_dim).
        """
        return self.mlp(t_embed)


class CharacterEmbedding(nn.Module):
    """Learned embedding for character identity.

    Maps character indices to dense vectors that capture character identity
    for conditioning the generation.
    """

    def __init__(self, num_characters: int, embed_dim: int):
        """Initialize character embedding.

        Args:
            num_characters: Number of unique characters (62 for a-z, A-Z, 0-9).
            embed_dim: Dimension of character embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_characters, embed_dim)

    def forward(self, char_indices: torch.Tensor) -> torch.Tensor:
        """Get character embeddings.

        Args:
            char_indices: Tensor of shape (batch_size,) with character indices.

        Returns:
            Embeddings of shape (batch_size, embed_dim).
        """
        return self.embedding(char_indices)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization for style conditioning.

    Projects style embedding to scale and shift parameters,
    then applies instance normalization with learned affine parameters.
    """

    def __init__(self, num_features: int, style_dim: int):
        """Initialize AdaIN layer.

        Args:
            num_features: Number of channels to normalize.
            style_dim: Dimension of style embedding.
        """
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_linear = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply adaptive instance normalization.

        Args:
            x: Feature map of shape (batch, channels, height, width).
            style: Style embedding of shape (batch, style_dim).

        Returns:
            Normalized and styled feature map.
        """
        style_params = self.style_linear(style)
        gamma, beta = style_params.chunk(2, dim=-1)
        # Reshape for broadcasting: (batch, channels, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        # Apply instance norm then affine transform
        normalized = self.instance_norm(x)
        return gamma * normalized + beta


class ResidualBlock(nn.Module):
    """Residual block with time and style conditioning.

    Each block consists of:
    1. Group norm + SiLU + Conv
    2. Time embedding addition
    3. Group norm + SiLU + Dropout + Conv
    4. AdaIN for style conditioning
    5. Skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        style_embed_dim: int,
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        """Initialize residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_embed_dim: Dimension of time embedding.
            style_embed_dim: Dimension of style embedding.
            dropout: Dropout probability.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.adain = AdaIN(out_channels, style_embed_dim)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        style_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width).
            t_embed: Time embedding of shape (batch, time_embed_dim).
            style_embed: Style embedding of shape (batch, style_embed_dim).

        Returns:
            Output tensor of shape (batch, out_channels, height, width).
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t = self.time_proj(t_embed)
        h = h + t[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Apply style conditioning via AdaIN
        h = self.adain(h, style_embed)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing global dependencies.

    Uses multi-head self-attention with a residual connection.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 8,
    ):
        """Initialize attention block.

        Args:
            channels: Number of channels.
            num_heads: Number of attention heads.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Output tensor of same shape.
        """
        batch, channels, height, width = x.shape
        h = self.norm(x)

        # Reshape to sequence: (batch, height*width, channels)
        h = rearrange(h, "b c h w -> b (h w) c")

        # Self-attention
        h, _ = self.attention(h, h, h)

        # Reshape back: (batch, channels, height, width)
        h = rearrange(h, "b (h w) c -> b c h w", h=height, w=width)

        return x + h


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""

    def __init__(self, channels: int):
        """Initialize downsample layer.

        Args:
            channels: Number of channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by factor of 2.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Downsampled tensor of shape (batch, channels, height//2, width//2).
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer using nearest neighbor + convolution."""

    def __init__(self, channels: int):
        """Initialize upsample layer.

        Args:
            channels: Number of channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by factor of 2.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Upsampled tensor of shape (batch, channels, height*2, width*2).
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """UNet architecture for flow-matching diffusion.

    The UNet processes images through:
    1. Input convolution (with optional mask concatenation)
    2. Encoder (downsampling path with residual blocks and attention)
    3. Middle block (residual + attention)
    4. Decoder (upsampling path with skip connections)
    5. Output convolution

    Conditioning is applied via:
    - Time: Sinusoidal embedding added to residual blocks
    - Character: Learned embedding added to time embedding
    - Style: AdaIN in each residual block
    - Mask: Concatenated to input (optional)
    """

    def __init__(self, config: ModelConfig):
        """Initialize UNet.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        image_size = config.image_size
        in_channels = config.in_channels
        if config.use_mask_conditioning:
            in_channels += 1  # Add mask channel

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.time_mlp = TimeEmbedMLP(config.time_embed_dim, config.time_embed_dim)

        # Character embedding (added to time embedding)
        # num_characters + 1 to accommodate the null class index used for
        # classifier-free guidance (index num_characters = unconditional)
        self.char_embed = CharacterEmbedding(
            config.num_characters + 1, config.char_embed_dim
        )
        self.char_proj = nn.Linear(config.char_embed_dim, config.time_embed_dim)

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, config.base_channels, kernel_size=3, padding=1)

        # Build channel schedule
        channels = [config.base_channels * m for m in config.channel_multipliers]
        num_levels = len(channels)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()

        prev_channels = config.base_channels
        current_resolution = image_size

        for level, ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                level_blocks.append(
                    ResidualBlock(
                        prev_channels,
                        ch,
                        config.time_embed_dim,
                        config.style_embed_dim,
                        config.dropout,
                    )
                )
                prev_channels = ch

                # Add attention if at right resolution
                if current_resolution in config.attention_resolutions:
                    level_blocks.append(AttentionBlock(ch, config.num_heads))

            self.encoder_blocks.append(level_blocks)

            # Downsample (except at last level)
            if level < num_levels - 1:
                self.encoder_downsamples.append(Downsample(ch))
                current_resolution //= 2
            else:
                self.encoder_downsamples.append(nn.Identity())

        # Middle
        self.middle_block1 = ResidualBlock(
            channels[-1],
            channels[-1],
            config.time_embed_dim,
            config.style_embed_dim,
            config.dropout,
        )
        self.middle_attention = AttentionBlock(channels[-1], config.num_heads)
        self.middle_block2 = ResidualBlock(
            channels[-1],
            channels[-1],
            config.time_embed_dim,
            config.style_embed_dim,
            config.dropout,
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for level in reversed(range(num_levels)):
            ch = channels[level]
            level_blocks = nn.ModuleList()

            for block_idx in range(config.num_res_blocks + 1):
                # Skip from encoder level has ch channels
                skip_ch = ch
                in_ch = prev_channels + skip_ch if block_idx == 0 else ch

                level_blocks.append(
                    ResidualBlock(
                        in_ch,
                        ch,
                        config.time_embed_dim,
                        config.style_embed_dim,
                        config.dropout,
                    )
                )
                prev_channels = ch

                # Add attention if at right resolution
                if current_resolution in config.attention_resolutions:
                    level_blocks.append(AttentionBlock(ch, config.num_heads))

            self.decoder_blocks.append(level_blocks)

            # Upsample (except at first level)
            if level > 0:
                self.decoder_upsamples.append(Upsample(ch))
                current_resolution *= 2
            else:
                self.decoder_upsamples.append(nn.Identity())

        # Output
        # After decoder, we concatenate the initial conv_in skip (base_channels)
        # with the decoder output (channels[0]), so input is channels[0] + base_channels
        self.norm_out = nn.GroupNorm(8, config.base_channels)
        self.conv_out = nn.Conv2d(
            config.base_channels, config.out_channels, kernel_size=3, padding=1
        )

        # Final projection to base channels, accounting for initial skip concatenation
        self.final_proj = nn.Conv2d(
            channels[0] + config.base_channels, config.base_channels, kernel_size=1
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        char_indices: torch.Tensor,
        style_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through UNet.

        Args:
            x: Noisy input of shape (batch, 1, height, width).
            timesteps: Timesteps in [0, 1] of shape (batch,).
            char_indices: Character indices of shape (batch,).
            style_embed: Style embeddings of shape (batch, style_embed_dim).
            mask: Optional mask for partial completion of shape (batch, 1, height, width).

        Returns:
            Velocity prediction of shape (batch, 1, height, width).
        """
        # Concatenate mask if provided
        if self.config.use_mask_conditioning:
            if mask is None:
                mask = torch.zeros_like(x)
            x = torch.cat([x, mask], dim=1)

        # Compute conditioning embeddings
        t_embed = self.time_embed(timesteps)
        t_embed = self.time_mlp(t_embed)

        c_embed = self.char_embed(char_indices)
        c_embed = self.char_proj(c_embed)

        # Combine time and character embeddings
        cond_embed = t_embed + c_embed

        # Input conv
        h = self.conv_in(x)

        # Encoder with skip connections
        skips = [h]
        for level_blocks, downsample in zip(
            self.encoder_blocks, self.encoder_downsamples
        ):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, cond_embed, style_embed)
                else:  # AttentionBlock
                    h = block(h)
            skips.append(h)
            h = downsample(h)

        # Middle
        h = self.middle_block1(h, cond_embed, style_embed)
        h = self.middle_attention(h)
        h = self.middle_block2(h, cond_embed, style_embed)

        # Decoder with skip connections
        for level_blocks, upsample in zip(
            self.decoder_blocks, self.decoder_upsamples
        ):
            for i, block in enumerate(level_blocks):
                if isinstance(block, ResidualBlock):
                    if i == 0:
                        # Concatenate skip connection
                        skip = skips.pop()
                        # Handle size mismatch from downsampling
                        if h.shape[2:] != skip.shape[2:]:
                            skip = F.interpolate(skip, size=h.shape[2:], mode="nearest")
                        h = torch.cat([h, skip], dim=1)
                    h = block(h, cond_embed, style_embed)
                else:  # AttentionBlock
                    h = block(h)
            h = upsample(h)

        # Consume the initial conv_in skip connection
        initial_skip = skips.pop()
        if h.shape[2:] != initial_skip.shape[2:]:
            initial_skip = F.interpolate(
                initial_skip, size=h.shape[2:], mode="nearest"
            )
        h = torch.cat([h, initial_skip], dim=1)

        # Output
        h = self.final_proj(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class GlyphDiffusionModel(nn.Module):
    """Wrapper combining UNet with conditioning modules.

    This is the main model class used for training and inference.
    It handles:
    - Building the UNet from config
    - Processing conditioning inputs
    - Velocity prediction for flow matching
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize GlyphDiffusion model.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        super().__init__()
        self.config = config or ModelConfig()
        self.unet = UNet(self.config)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        char_indices: torch.Tensor,
        style_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity field for flow matching.

        Args:
            x: Noisy input of shape (batch, 1, height, width).
            timesteps: Timesteps in [0, 1] of shape (batch,).
            char_indices: Character indices of shape (batch,).
            style_embed: Style embeddings of shape (batch, style_embed_dim).
            mask: Optional mask for partial completion.

        Returns:
            Predicted velocity of shape (batch, 1, height, width).
        """
        return self.unet(x, timesteps, char_indices, style_embed, mask)

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: If True, only count trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(config: Optional[ModelConfig] = None) -> GlyphDiffusionModel:
    """Factory function to create a GlyphDiffusion model.

    Args:
        config: Model configuration. Uses defaults if not provided.

    Returns:
        Initialized model.
    """
    return GlyphDiffusionModel(config)


if __name__ == "__main__":
    # Test model creation and forward pass
    config = ModelConfig()
    model = create_model(config)
    print(f"Model parameters: {model.num_parameters():,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, config.image_size, config.image_size)
    timesteps = torch.rand(batch_size)
    char_indices = torch.randint(0, config.num_characters, (batch_size,))
    style_embed = torch.randn(batch_size, config.style_embed_dim)
    mask = torch.zeros(batch_size, 1, config.image_size, config.image_size)

    with torch.no_grad():
        output = model(x, timesteps, char_indices, style_embed, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("Forward pass successful!")
