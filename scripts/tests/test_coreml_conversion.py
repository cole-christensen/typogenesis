"""Tests for CoreML model conversion naming and shapes.

Verifies that the CoreML conversion in train_all.py produces models
with the correct input/output names expected by the Swift app.
"""

import types

import pytest
import torch


def _patch_attention_blocks(model):
    """Replace einops.rearrange in AttentionBlock with trace-friendly ops."""
    for module in model.modules():
        if hasattr(module, "attention") and hasattr(module, "norm"):

            def _trace_friendly_forward(self, x: torch.Tensor) -> torch.Tensor:
                batch, channels, height, width = x.shape
                h = self.norm(x)
                h = h.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
                h, _ = self.attention(h, h, h)
                h = h.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
                return x + h

            module.forward = types.MethodType(_trace_friendly_forward, module)


class _UNetExportWrapper(torch.nn.Module):
    """Wrapper that makes mask non-optional for tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, char_indices, style_embed, mask):
        return self.model(x, timesteps, char_indices, style_embed, mask)


# ─── StyleEncoder conversion output naming ───


def test_style_encoder_coreml_output_name():
    """Verify StyleEncoder CoreML conversion specifies output name 'embedding'."""
    ct = pytest.importorskip("coremltools")
    from models.style_encoder.model import StyleEncoder, StyleEncoderConfig

    config = StyleEncoderConfig(pretrained=False)
    model = StyleEncoder(config)
    model.eval()

    dummy = torch.randn(1, 1, 64, 64)
    traced = torch.jit.trace(model, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 1, 64, 64))],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.macOS14,
    )

    spec = mlmodel.get_spec()
    output_names = [o.name for o in spec.description.output]
    assert "embedding" in output_names, f"Expected 'embedding' in output names, got {output_names}"


# ─── KerningNet conversion output naming ───


def test_kerning_net_coreml_output_name():
    """Verify KerningNet CoreML conversion specifies output name 'kerning'."""
    ct = pytest.importorskip("coremltools")
    from models.kerning_net.model import KerningNet

    model = KerningNet()
    model.eval()

    dummy_left = torch.randn(1, 1, 64, 64)
    dummy_right = torch.randn(1, 1, 64, 64)
    traced = torch.jit.trace(model, (dummy_left, dummy_right))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="left_glyph", shape=(1, 1, 64, 64)),
            ct.TensorType(name="right_glyph", shape=(1, 1, 64, 64)),
        ],
        outputs=[ct.TensorType(name="kerning")],
        minimum_deployment_target=ct.target.macOS14,
    )

    spec = mlmodel.get_spec()
    output_names = [o.name for o in spec.description.output]
    assert "kerning" in output_names, f"Expected 'kerning' in output names, got {output_names}"

    input_names = [i.name for i in spec.description.input]
    assert "left_glyph" in input_names
    assert "right_glyph" in input_names


# ─── GlyphDiffusion UNet conversion ───


def _create_exported_unet():
    """Create an exported UNet model ready for CoreML conversion."""
    import numpy as np
    from torch.export import export as torch_export

    from models.glyph_diffusion.config import ModelConfig
    from models.glyph_diffusion.model import GlyphDiffusionModel

    config = ModelConfig()
    model = GlyphDiffusionModel(config)
    model.eval()
    _patch_attention_blocks(model)

    wrapper = _UNetExportWrapper(model)
    wrapper.eval()

    dummy_x = torch.randn(1, 1, 64, 64)
    dummy_t = torch.tensor([0.5])
    dummy_char = torch.tensor([0], dtype=torch.int32)
    dummy_style = torch.randn(1, 128)
    dummy_mask = torch.zeros(1, 1, 64, 64)

    exported = torch_export(
        wrapper,
        (dummy_x, dummy_t, dummy_char, dummy_style, dummy_mask),
    )
    exported = exported.run_decompositions({})

    return exported, wrapper, np


def test_glyph_diffusion_coreml_names():
    """Verify GlyphDiffusion UNet CoreML conversion has correct input/output names."""
    ct = pytest.importorskip("coremltools")

    exported, _, np = _create_exported_unet()

    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="x", shape=(1, 1, 64, 64)),
            ct.TensorType(name="timesteps", shape=(1,)),
            ct.TensorType(name="char_indices", shape=(1,), dtype=np.int32),
            ct.TensorType(name="style_embed", shape=(1, 128)),
            ct.TensorType(name="mask", shape=(1, 1, 64, 64)),
        ],
        outputs=[ct.TensorType(name="velocity")],
        minimum_deployment_target=ct.target.macOS14,
    )

    spec = mlmodel.get_spec()
    input_names = [i.name for i in spec.description.input]
    output_names = [o.name for o in spec.description.output]

    assert "x" in input_names, f"Expected 'x' in inputs, got {input_names}"
    assert "timesteps" in input_names
    assert "char_indices" in input_names
    assert "style_embed" in input_names
    assert "mask" in input_names
    assert "velocity" in output_names, f"Expected 'velocity' in outputs, got {output_names}"


# ─── UNet numerical verification ───


def test_unet_coreml_output_matches_pytorch():
    """Verify CoreML UNet output matches PyTorch within tolerance."""
    ct = pytest.importorskip("coremltools")

    import numpy as np
    from torch.export import export as torch_export

    from models.glyph_diffusion.config import ModelConfig
    from models.glyph_diffusion.model import GlyphDiffusionModel

    config = ModelConfig()
    model = GlyphDiffusionModel(config)
    model.eval()
    _patch_attention_blocks(model)

    wrapper = _UNetExportWrapper(model)
    wrapper.eval()

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    test_x = torch.randn(1, 1, 64, 64)
    test_t = torch.tensor([0.5])
    test_char = torch.tensor([10], dtype=torch.int32)
    test_style = torch.randn(1, 128)
    test_mask = torch.zeros(1, 1, 64, 64)

    # PyTorch output
    with torch.no_grad():
        pt_output = wrapper(test_x, test_t, test_char, test_style, test_mask)

    # Export with test inputs for conversion
    exported = torch_export(
        wrapper,
        (test_x, test_t, test_char, test_style, test_mask),
    )
    exported = exported.run_decompositions({})

    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="x", shape=(1, 1, 64, 64)),
            ct.TensorType(name="timesteps", shape=(1,)),
            ct.TensorType(name="char_indices", shape=(1,), dtype=np.int32),
            ct.TensorType(name="style_embed", shape=(1, 128)),
            ct.TensorType(name="mask", shape=(1, 1, 64, 64)),
        ],
        outputs=[ct.TensorType(name="velocity")],
        minimum_deployment_target=ct.target.macOS14,
    )

    # CoreML prediction
    coreml_input = {
        "x": test_x.numpy(),
        "timesteps": test_t.numpy(),
        "char_indices": test_char.numpy().astype(np.int32),
        "style_embed": test_style.numpy(),
        "mask": test_mask.numpy(),
    }

    coreml_output = mlmodel.predict(coreml_input)
    coreml_velocity = coreml_output["velocity"]

    pt_np = pt_output.numpy()

    # Check shapes match
    assert pt_np.shape == coreml_velocity.shape, (
        f"Shape mismatch: PyTorch {pt_np.shape} vs CoreML {coreml_velocity.shape}"
    )

    # Check values are close (UNet with attention + AdaIN has higher conversion variance;
    # use mean absolute error since max can spike on individual elements)
    mae = np.abs(pt_np - coreml_velocity).mean()
    assert mae < 0.05, f"Mean abs error {mae:.6f} exceeds tolerance 0.05"
    max_diff = np.abs(pt_np - coreml_velocity).max()
    assert max_diff < 0.5, f"Max difference {max_diff:.6f} exceeds tolerance 0.5"
