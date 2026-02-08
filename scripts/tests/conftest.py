"""Shared pytest fixtures for Typogenesis ML tests.

Provides fixtures for:
- Device selection (CPU/GPU)
- Synthetic glyph image generation
- Model factory functions with default configs
- Temporary directories for test outputs
"""

import sys
from pathlib import Path

import pytest
import torch

# Add scripts/ to path so we can import models as packages
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


@pytest.fixture
def device():
    """Select test device: GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Small batch size for fast tests."""
    return 4


@pytest.fixture
def image_size():
    """Default image size for glyph tests."""
    return 64


@pytest.fixture
def synthetic_glyphs(batch_size, image_size):
    """Generate synthetic grayscale glyph images.

    Returns tensor of shape (batch_size, 1, image_size, image_size)
    with values in [-1, 1] (normalized).
    """
    torch.manual_seed(42)
    return torch.randn(batch_size, 1, image_size, image_size)


@pytest.fixture
def char_indices(batch_size):
    """Random character indices for conditioning (0-61)."""
    torch.manual_seed(42)
    return torch.randint(0, 62, (batch_size,))


@pytest.fixture
def style_embeddings(batch_size):
    """Random 128-dim style embeddings."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 128)


@pytest.fixture
def timesteps(batch_size):
    """Random timesteps in [0, 1] for flow matching."""
    torch.manual_seed(42)
    return torch.rand(batch_size)


@pytest.fixture
def glyph_diffusion_model():
    """Create GlyphDiffusion model with default config."""
    from models.glyph_diffusion import GlyphDiffusionModel, ModelConfig

    config = ModelConfig()
    model = GlyphDiffusionModel(config)
    model.eval()
    return model


@pytest.fixture
def style_encoder_model():
    """Create StyleEncoder model with default config (no pretrained weights for speed)."""
    from models.style_encoder import StyleEncoder, StyleEncoderConfig

    config = StyleEncoderConfig(pretrained=False)
    model = StyleEncoder(config)
    model.eval()
    return model


@pytest.fixture
def kerning_net_model():
    """Create KerningNet model with default config."""
    from models.kerning_net import KerningNet, KerningNetConfig

    config = KerningNetConfig()
    model = KerningNet(config)
    model.eval()
    return model


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir
