"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder
from .residual_quantizer import ResidualQuantizer
from utils import LossWithIntermediateLosses


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor          # summed RAW codes (grad-connected), (.., E, h, w)
    tokens: torch.LongTensor                # WM-facing level-0 tokens (.., 16) -- keeps tokens_per_block=17
    tokens_levels: List[torch.LongTensor]   # ECHELON-parity: per-level indices [(.,16) x 3]
    vq_loss: torch.FloatTensor              # sum of the 3 per-level commitment losses


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder,
                 with_lpips: bool = True,
                 hrvq_num_codes: Optional[List[int]] = None,
                 hrvq_commitment_costs: Optional[List[float]] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        # PORT: the stock single-level nn.Embedding codebook is replaced by a
        # 3-level gradient-style residual quantiser (ECHELON's cascade, IRIS
        # internals). Per-level code count defaults to vocab_size so the
        # WM-facing token vocabulary (level 0) is unchanged at 512.
        hrvq_num_codes = hrvq_num_codes if hrvq_num_codes is not None else [vocab_size, vocab_size, vocab_size]
        hrvq_commitment_costs = hrvq_commitment_costs if hrvq_commitment_costs is not None else [1.0, 1.0, 1.0]
        self.hrvq = ResidualQuantizer(embed_dim=embed_dim, num_codes=hrvq_num_codes,
                                      commitment_costs=hrvq_commitment_costs)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.lpips = LPIPS().eval() if with_lpips else None

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        outputs = self.encode(observations, should_preprocess=False)
        # Straight-through on the summed codes (forward = summed codes, grad to encoder).
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess=False)

        # PORT: the commitment loss is now the SUM of the 3 per-level residual
        # commitment losses (beta=1.0 each), computed inside the residual
        # quantiser, rather than the stock single-codebook term.
        commitment_loss = outputs.vq_loss

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape

        # 3-level residual quantisation over the (b, h*w, e) = (b, 16, 512) grid.
        z_seq = rearrange(z, 'b e h w -> b (h w) e')
        rq = self.hrvq(z_seq)
        # Summed RAW codes (grad-connected to the codebooks); STE applied by callers.
        z_q = rearrange(sum(rq["z_q_levels"]), 'b (h w) e -> b e h w', h=h, w=w).contiguous()
        tokens_levels = rq["indices"]          # list of (b, 16)
        # WM-facing tokens are LEVEL 0 only -> tokens stay 16, tokens_per_block=17,
        # WM transformer untouched. Levels 1-2 refine reconstruction only.
        tokens = tokens_levels[0]
        vq_loss = rq["vq_loss"]

        # Reshape to original batch prefix.
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)
        tokens_levels = [t.reshape(*shape[:-3], -1) for t in tokens_levels]

        return TokenizerEncoderOutput(z, z_q, tokens, tokens_levels, vq_loss)

    def tokens_to_z(self, tokens) -> torch.Tensor:
        """Map discrete tokens back to a spatial latent z (B, E, h, w) by SUMMING
        level embeddings -- the residual-cascade replacement for the stock
        single ``self.embedding(tokens)`` lookup in WorldModelEnv.decode_obs_tokens.

        ``tokens`` is either a single LongTensor (B, K) -- the world model only
        predicts the level-0 stream, so the sum is over level 0 alone -- or a
        list of per-level LongTensors [(B,K) x L] for full-cascade reconstruction.
        Either way the output shape is identical, which is what keeps decode and
        the WM sequence length unchanged.
        """
        if torch.is_tensor(tokens):
            tokens = [tokens]
        k = tokens[0].shape[-1]
        h = int(round(k ** 0.5))
        z_q = sum(self.hrvq.quantizers[level].embedding(tokens[level]) for level in range(len(tokens)))  # (B, K, E)
        return rearrange(z_q, 'b (h w) e -> b e h w', h=h)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def freeze_encoder(self) -> None:
        """Freeze the encoder CNN and pre_quant_conv (mirrors main.py --freeze_encoder).

        Used with codebook transfer: when frozen levels carry transferred codes,
        the encoder that feeds them is held fixed too so the graft is not undone
        by encoder drift. The decoder / post_quant_conv stay trainable.
        """
        self.encoder.requires_grad_(False)
        self.pre_quant_conv.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        self.encoder.requires_grad_(True)
        self.pre_quant_conv.requires_grad_(True)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
