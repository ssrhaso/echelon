"""Unit tests: ResidualQuantizer (gradient-style, 3 levels) in isolation.

Proves the three properties the port requires of the cascade:
  * shape: input (B, 16, 512) -> summed z_q (B, 16, 512)
  * STE identity: grad of z_q w.r.t. input is pass-through (~1)
  * 3 distinct commitment losses are returned

All CPU, no tokenizer/world-model wiring involved.
"""
import torch

from models.tokenizer.residual_quantizer import ResidualQuantizer

B, N, D = 2, 16, 512


def _rq():
    return ResidualQuantizer(embed_dim=D, num_codes=[512, 512, 512],
                             commitment_costs=[1.0, 1.0, 1.0])


def test_summed_output_shape_matches_input():
    rq = _rq()
    z = torch.randn(B, N, D)
    out = rq(z)
    assert out["z_q"].shape == (B, N, D)
    assert len(out["z_q_levels"]) == 3 and all(zl.shape == (B, N, D) for zl in out["z_q_levels"])
    assert len(out["indices"]) == 3 and all(idx.shape == (B, N) for idx in out["indices"])
    assert all(idx.dtype == torch.long for idx in out["indices"])


def test_straight_through_is_identity_passthrough():
    """d(z_q)/d(z_e) must be identity: z_q = z_e + (sum_codes - z_e).detach()."""
    rq = _rq()
    z = torch.randn(B, N, D, requires_grad=True)
    out = rq(z)
    # Seed a known upstream gradient and check it passes straight through.
    g = torch.randn(B, N, D)
    out["z_q"].backward(g)
    assert z.grad is not None
    assert torch.allclose(z.grad, g, atol=1e-6), "STE is not a pass-through"


def test_forward_value_equals_sum_of_codes():
    """Forward value of z_q equals the (detached) sum of per-level code vectors."""
    rq = _rq()
    z = torch.randn(B, N, D)
    out = rq(z)
    summed = sum(out["z_q_levels"]).detach()
    assert torch.allclose(out["z_q"], summed, atol=1e-6)


def test_three_distinct_commitment_losses():
    rq = _rq()
    z = torch.randn(B, N, D)
    out = rq(z)
    losses = out["commitment_losses"]
    assert len(losses) == 3
    vals = [float(l.detach()) for l in losses]
    assert all(v >= 0 for v in vals)
    # Residual magnitudes shrink across levels, so the three losses differ.
    assert len(set(round(v, 8) for v in vals)) == 3, f"losses not distinct: {vals}"
    assert torch.allclose(out["vq_loss"], sum(losses))


def test_codebooks_receive_gradient_from_commitment():
    """Each level's nn.Embedding gets a nonzero grad from the commitment loss."""
    rq = _rq()
    z = torch.randn(B, N, D)
    out = rq(z)
    out["vq_loss"].backward()
    for level, q in enumerate(rq.quantizers):
        g = q.embedding.weight.grad
        assert g is not None and g.abs().sum() > 0, f"level {level} codebook got no grad"
