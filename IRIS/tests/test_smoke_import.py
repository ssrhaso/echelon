"""Smoke test: the CPU env is sane and IRIS modules import.

Proves the toolchain is wired (torch CPU, einops) and the IRIS source tree is
importable from src/. Deliberately avoids importing
the gym/ale-dependent env modules and avoids instantiating LPIPS (which would
fetch VGG weights over the network).
"""


def test_torch_is_cpu_only():
    import torch

    assert not torch.cuda.is_available(), "CPU validation env must not see a GPU"
    x = torch.randn(2, 3)
    assert (x + x).shape == (2, 3)


def test_import_iris_tokenizer_modules():
    # Core tokenizer pieces we will modify in later commits.
    from models.tokenizer import Encoder, Decoder, EncoderDecoderConfig, Tokenizer

    assert all(callable(c) for c in (Encoder, Decoder, EncoderDecoderConfig, Tokenizer))


def test_import_iris_world_model_modules():
    from models.world_model import WorldModel
    from models.transformer import Transformer, TransformerConfig

    assert all(callable(c) for c in (WorldModel, Transformer, TransformerConfig))


def test_build_stock_encoder_decoder_on_cpu():
    """The real 64->4 downsampling stack builds and runs on CPU -> 4x4 spatial."""
    import torch
    from conftest import make_encoder_decoder

    encoder, _ = make_encoder_decoder()
    x = torch.randn(1, 3, 64, 64)
    z = encoder(x)
    # ch_mult has 5 entries => 4 downsamples => 64/2**4 = 4 => 4x4 = 16 positions.
    assert z.shape[-2:] == (4, 4), z.shape
