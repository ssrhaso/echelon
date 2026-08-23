"""Levels can carry signal: CPU proxy for cascade health.

This proves L0/L1/L2 are WIRED to learn: each level receives a nonzero, distinct
gradient, and a tiny overfit loop actually moves every level's codebook and
reduces the commitment loss.

It does NOT prove the levels will stay alive under real training. Residual
collapse (L1/L2 going dead) is a dynamics property only observable on GPU with a
real encoder and data.
"""
import torch

from models.tokenizer.residual_quantizer import ResidualQuantizer

D = 512


def _rq():
    return ResidualQuantizer(embed_dim=D, num_codes=[512, 512, 512],
                             commitment_costs=[1.0, 1.0, 1.0])


def test_each_level_receives_nonzero_and_distinct_gradient():
    rq = _rq()
    torch.manual_seed(1)
    z = torch.randn(8, 16, D)
    rq(z)["vq_loss"].backward()

    grad_norms = []
    for level, q in enumerate(rq.quantizers):
        g = q.embedding.weight.grad
        assert g is not None and g.abs().sum() > 0, f"level {level} got no gradient"
        grad_norms.append(float(g.norm()))

    # Distinct: the three levels see different residual scales, so their codebook
    # gradients have different magnitudes (not a degenerate shared signal).
    assert len(set(round(n, 6) for n in grad_norms)) == 3, f"grad norms not distinct: {grad_norms}"


def test_tiny_overfit_moves_every_level_and_reduces_loss():
    rq = _rq()
    torch.manual_seed(2)
    z = torch.randn(16, 16, D)  # fixed tiny batch to overfit

    init_weights = [q.embedding.weight.detach().clone() for q in rq.quantizers]
    opt = torch.optim.Adam(rq.parameters(), lr=1e-2)

    first_loss = None
    for step in range(25):
        opt.zero_grad()
        loss = rq(z)["vq_loss"]
        loss.backward()
        opt.step()
        if step == 0:
            first_loss = float(loss.detach())
    last_loss = float(rq(z)["vq_loss"].detach())

    assert last_loss < first_loss, f"loss did not decrease: {first_loss} -> {last_loss}"
    for level, (q, w0) in enumerate(zip(rq.quantizers, init_weights)):
        moved = (q.embedding.weight.detach() - w0).abs().sum()
        assert moved > 0, f"level {level} codebook did not move under overfit"
