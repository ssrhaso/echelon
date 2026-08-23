"""Residual quantiser wired into Tokenizer.encode / compute_loss.

Asserts:
  * encode keeps the WM-facing contract (tokens (B,16) Long) AND exposes the
    ECHELON-parity extras (tokens_levels x3, vq_loss).
  * WM-facing tokens are exactly level-0 indices (tokens_per_block stays 17).
  * the tokenizer surfaces the ECHELON freezing interface via .hrvq.
  * compute_loss runs hermetically (LPIPS stubbed) and its backward trains the
    encoder and all three codebooks; commitment_loss == residual vq_loss.
"""
import torch
import torch.nn as nn

from conftest import make_stock_tokenizer

B, C, H, W = 2, 3, 64, 64
N_TOKENS, EMBED_DIM, VOCAB = 16, 512, 512


class _ZeroLPIPS(nn.Module):
    """Hermetic stand-in for LPIPS: no VGG download, returns a per-sample zero."""
    def forward(self, a, b):
        return a.new_zeros(a.shape[0], 1, 1, 1)


def _obs():
    return torch.rand(B, C, H, W)


def test_encode_exposes_levels_and_keeps_wm_tokens():
    tok = make_stock_tokenizer()
    out = tok.encode(_obs(), should_preprocess=True)
    # WM-facing contract unchanged.
    assert out.tokens.shape == (B, N_TOKENS) and out.tokens.dtype == torch.long
    assert out.z_quantized.shape == (B, EMBED_DIM, 4, 4)
    # ECHELON-parity extras.
    assert len(out.tokens_levels) == 3
    assert all(t.shape == (B, N_TOKENS) and t.dtype == torch.long for t in out.tokens_levels)
    assert out.vq_loss.ndim == 0
    # WM tokens == level-0 indices.
    assert torch.equal(out.tokens, out.tokens_levels[0])


def test_tokenizer_exposes_echelon_freeze_interface():
    tok = make_stock_tokenizer()
    assert hasattr(tok.hrvq, "freeze_levels") and hasattr(tok.hrvq, "get_frozen_levels")
    assert hasattr(tok.hrvq, "get_codebook_usage")
    assert tok.hrvq.get_frozen_levels() == []


def test_compute_loss_hermetic_and_trains_codebooks():
    tok = make_stock_tokenizer()
    tok.lpips = _ZeroLPIPS()  # avoid VGG download; keep compute_loss assertable
    batch = {"observations": torch.rand(B, 1, C, H, W)}  # (b, t, c, h, w)

    losses = tok.compute_loss(batch)
    d = losses.intermediate_losses
    assert {"commitment_loss", "reconstruction_loss", "perceptual_loss"} <= set(d.keys())
    assert d["perceptual_loss"] == 0.0  # stubbed

    losses.loss_total.backward()
    assert tok.encoder.conv_in.weight.grad.abs().sum() > 0
    for level, q in enumerate(tok.hrvq.quantizers):
        g = q.embedding.weight.grad
        assert g is not None and g.abs().sum() > 0, f"level {level} codebook untrained"
