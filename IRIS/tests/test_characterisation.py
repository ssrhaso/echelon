"""Characterisation tests: STOCK single-level IRIS tokenizer.

These lock the pre-port behaviour so the 3-level residual quantiser can be
proven shape- and grad-compatible with what the world model already consumes.
If a change breaks any of these invariants unintentionally, these tests go red.

Stock invariants captured here:
  * encode -> tokens (B, 16) Long; z and z_quantized (B, 512, 4, 4)
  * decode((B,512,4,4)) -> (B, 3, 64, 64)
  * round-trip encode_decode -> (B, 3, 64, 64)
  * one backward pass produces grads on BOTH encoder and codebook (embedding)
"""
import torch

from conftest import make_stock_tokenizer

B, C, H, W = 2, 3, 64, 64
N_TOKENS = 16          # 4x4 spatial grid
EMBED_DIM = 512
VOCAB = 512


def _obs():
    return torch.rand(B, C, H, W)  # channels-first, in [0,1]


def test_encode_shapes():
    tok = make_stock_tokenizer()
    out = tok.encode(_obs(), should_preprocess=True)
    assert out.tokens.shape == (B, N_TOKENS)
    assert out.tokens.dtype == torch.long
    assert out.z.shape == (B, EMBED_DIM, 4, 4)
    assert out.z_quantized.shape == (B, EMBED_DIM, 4, 4)
    assert int(out.tokens.min()) >= 0 and int(out.tokens.max()) < VOCAB


def test_decode_shape():
    tok = make_stock_tokenizer()
    out = tok.encode(_obs(), should_preprocess=True)
    rec = tok.decode(out.z_quantized, should_postprocess=False)
    assert rec.shape == (B, C, H, W)


def test_round_trip_shape():
    tok = make_stock_tokenizer()
    rec = tok.encode_decode(_obs(), should_preprocess=True, should_postprocess=True)
    assert rec.shape == (B, C, H, W)


def test_forward_returns_triplet_shapes():
    tok = make_stock_tokenizer()
    z, z_q, rec = tok(_obs(), should_preprocess=True, should_postprocess=False)
    assert z.shape == (B, EMBED_DIM, 4, 4)
    assert z_q.shape == (B, EMBED_DIM, 4, 4)
    assert rec.shape == (B, C, H, W)


def test_single_backward_grads_encoder_and_codebook():
    """One backward pass must give grad to encoder AND every codebook AND decoder.

    Post-port the commitment loss is the residual quantiser's vq_loss (sum of the
    3 per-level beta=1.0 commitment losses). It is the ONLY path that reaches the
    codebooks (the decoder STE detaches them), so a nonzero grad on each level's
    nn.Embedding proves all three codebooks are being trained. Shapes asserted
    here are IDENTICAL to the stock tokenizer (characterisation invariant).
    """
    tok = make_stock_tokenizer()
    obs = tok.preprocess_input(_obs())
    out = tok.encode(obs, should_preprocess=False)
    decoder_input = out.z + (out.z_quantized - out.z).detach()
    rec = tok.decode(decoder_input, should_postprocess=False)

    # Same shapes as the stock single-level tokenizer.
    assert out.z.shape == (B, EMBED_DIM, 4, 4)
    assert out.z_quantized.shape == (B, EMBED_DIM, 4, 4)
    assert out.tokens.shape == (B, N_TOKENS) and out.tokens.dtype == torch.long
    assert rec.shape == (B, C, H, W)

    reconstruction = (obs - rec).abs().mean()
    (out.vq_loss + reconstruction).backward()

    enc_grad = tok.encoder.conv_in.weight.grad
    dec_grad = tok.decoder.conv_out.weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0, "encoder got no grad"
    assert dec_grad is not None and dec_grad.abs().sum() > 0, "decoder got no grad"
    for level, q in enumerate(tok.hrvq.quantizers):
        g = q.embedding.weight.grad
        assert g is not None and g.abs().sum() > 0, f"level {level} codebook got no grad"
