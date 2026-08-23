"""Decode path sums level embeddings.

WorldModelEnv.decode_obs_tokens now calls Tokenizer.tokens_to_z, which sums
level embeddings instead of the stock single-codebook lookup. world_model_env.py
itself imports gym (an env-only dependency not installed in the CPU env), so we
exercise the summing helper + decode directly. Verifies:
  * 3-level decode shape == single-level decode shape (== stock (B,3,64,64))
  * the WM sequence length is unchanged (encode still yields 16 obs tokens)
"""
import torch

from conftest import make_stock_tokenizer

B, C, H, W = 2, 3, 64, 64
N_TOKENS, EMBED_DIM, VOCAB = 16, 512, 512


def _tokens():
    return torch.randint(0, VOCAB, (B, N_TOKENS))


def test_tokens_to_z_shape_single_and_three_levels_match():
    tok = make_stock_tokenizer()
    z_single = tok.tokens_to_z(_tokens())                       # level-0 only (WM path)
    z_three = tok.tokens_to_z([_tokens(), _tokens(), _tokens()])  # full cascade
    assert z_single.shape == (B, EMBED_DIM, 4, 4)
    assert z_three.shape == z_single.shape


def test_decode_from_summed_embeddings_matches_stock_shape():
    tok = make_stock_tokenizer()
    z = tok.tokens_to_z(_tokens())
    rec = tok.decode(z, should_postprocess=True)
    assert rec.shape == (B, C, H, W)
    # And the full-cascade path decodes to the identical shape.
    rec3 = tok.decode(tok.tokens_to_z([_tokens(), _tokens(), _tokens()]), should_postprocess=True)
    assert rec3.shape == rec.shape


def test_three_level_sum_equals_sum_of_level_lookups():
    """tokens_to_z over 3 levels == sum of the three per-level embedding lookups."""
    from einops import rearrange
    tok = make_stock_tokenizer()
    levels = [_tokens(), _tokens(), _tokens()]
    z = tok.tokens_to_z(levels)
    expected = sum(tok.hrvq.quantizers[l].embedding(levels[l]) for l in range(3))
    expected = rearrange(expected, 'b (h w) e -> b e h w', h=4)
    assert torch.allclose(z, expected, atol=1e-6)


def test_wm_sequence_length_unchanged():
    """The WM still consumes 16 obs tokens per frame (tokens_per_block stays 17)."""
    tok = make_stock_tokenizer()
    out = tok.encode(torch.rand(B, C, H, W), should_preprocess=True)
    assert out.tokens.shape[-1] == N_TOKENS
