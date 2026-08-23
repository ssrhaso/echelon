"""Freeze mechanism + requires_grad wiring.

The CPU proxy for the STE / encoder-drift guarantee: after freezing a codebook
level (and the encoder), a backward pass must yield ZERO gradient on exactly
those parameters and NONZERO gradient everywhere else that should still train.
"""
import torch
import torch.nn as nn

from conftest import make_stock_tokenizer


class _ZeroLPIPS(nn.Module):
    def forward(self, a, b):
        return a.new_zeros(a.shape[0], 1, 1, 1)


def _tokenizer_with_loss_backward(freeze_levels, freeze_encoder):
    tok = make_stock_tokenizer()
    tok.lpips = _ZeroLPIPS()
    if freeze_levels:
        tok.hrvq.freeze_levels(freeze_levels)
    if freeze_encoder:
        tok.freeze_encoder()
    batch = {"observations": torch.rand(2, 1, 3, 64, 64)}
    tok.compute_loss(batch).loss_total.backward()
    return tok


def test_freeze_levels_sets_state_and_interface():
    tok = make_stock_tokenizer()
    tok.hrvq.freeze_levels([1])
    assert tok.hrvq.get_frozen_levels() == [1]
    assert tok.hrvq.quantizers[1].frozen is True
    assert tok.hrvq.quantizers[1].embedding.weight.requires_grad is False
    assert tok.hrvq.quantizers[0].embedding.weight.requires_grad is True


def test_frozen_level_and_encoder_get_zero_grad_others_nonzero():
    tok = _tokenizer_with_loss_backward(freeze_levels=[1], freeze_encoder=True)

    # ZERO grad on exactly the frozen params (requires_grad False -> grad stays None).
    assert tok.hrvq.quantizers[1].embedding.weight.grad is None, "frozen level 1 got grad"
    assert tok.encoder.conv_in.weight.grad is None, "frozen encoder got grad"
    assert tok.pre_quant_conv.weight.grad is None, "frozen pre_quant_conv got grad"

    # NONZERO grad on everything that must still train.
    for level in (0, 2):
        g = tok.hrvq.quantizers[level].embedding.weight.grad
        assert g is not None and g.abs().sum() > 0, f"unfrozen level {level} got no grad"
    assert tok.decoder.conv_out.weight.grad.abs().sum() > 0, "decoder got no grad"
    assert tok.post_quant_conv.weight.grad.abs().sum() > 0, "post_quant_conv got no grad"


def test_frozen_level_contributes_zero_commitment_loss():
    tok = make_stock_tokenizer()
    z = torch.randn(2, 16, 512)
    tok.hrvq.freeze_levels([2])
    out = tok.hrvq(z)
    assert float(out["commitment_losses"][2].detach()) == 0.0
    assert float(out["commitment_losses"][0].detach()) > 0.0


def test_unfreeze_restores_training():
    tok = make_stock_tokenizer()
    tok.hrvq.freeze_levels([0])
    tok.hrvq.quantizers[0].unfreeze()
    assert tok.hrvq.quantizers[0].embedding.weight.requires_grad is True
    assert tok.hrvq.get_frozen_levels() == []
