"""Per-level codebook usage method.

Verifies get_codebook_usage returns a live-code fraction and perplexity per
level, with the right ranges and keys. This is the CPU proxy for the cascade
liveness monitored on GPU.
"""
import torch

from models.tokenizer.residual_quantizer import ResidualQuantizer

D = 512


def _rq():
    return ResidualQuantizer(embed_dim=D, num_codes=[512, 512, 512],
                             commitment_costs=[1.0, 1.0, 1.0])


def test_usage_keys_and_ranges():
    rq = _rq()
    z = torch.randn(8, 16, D)
    stats = rq.get_codebook_usage(rq(z)["indices"])
    for level in range(3):
        u = stats[f"usage_{level}"]
        p = stats[f"perplexity_{level}"]
        assert 0.0 < u <= 1.0, f"usage_{level} out of range: {u}"
        assert 1.0 <= p <= 512.0 + 1e-3, f"perplexity_{level} out of range: {p}"


def test_perplexity_collapses_to_one_for_single_code():
    """If every position maps to one code, perplexity -> 1 and usage -> 1/512."""
    rq = _rq()
    indices = [torch.zeros(4, 16, dtype=torch.long) for _ in range(3)]
    stats = rq.get_codebook_usage(indices)
    for level in range(3):
        assert abs(stats[f"perplexity_{level}"] - 1.0) < 1e-4
        assert abs(stats[f"usage_{level}"] - 1.0 / 512) < 1e-9


def test_perplexity_higher_for_more_diverse_codes():
    rq = _rq()
    collapsed = [torch.zeros(16, 16, dtype=torch.long) for _ in range(3)]
    diverse = [torch.arange(256).repeat(1).reshape(16, 16) % 512 for _ in range(3)]
    p_collapsed = rq.get_codebook_usage(collapsed)["perplexity_0"]
    p_diverse = rq.get_codebook_usage(diverse)["perplexity_0"]
    assert p_diverse > p_collapsed
