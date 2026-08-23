"""3-level residual vector quantiser for the IRIS + HRVQ port.

This reimplements ECHELON's 3-level residual cascade (reference:
nnet/modules/twister/hrvq/vq.py:188-247) in IRIS's NATIVE gradient-codebook
style.

TWO-MECHANISM SEAM (read before merging back into ECHELON)
----------------------------------------------------------
ECHELON's quantiser (VectorQuantizerEMA) updates its codebook with an
exponential-moving-average buffer and revives dead codes; its commitment loss
only nudges the encoder. IRIS instead trains the codebook with GRADIENT descent
through an nn.Embedding: the "codebook term" ``(z.detach() - z_q)**2`` carries
the gradient that moves the codes, exactly as in IRIS's stock
Tokenizer.compute_loss (beta=1.0). We keep that native style here -- NO EMA, NO
dead-code revival. The collapse risk that EMA+revival hides is therefore a
GPU-side finding, not silently patched.

To let the eventual ECHELON merge share its freezing-ladder driver and logging,
this module exposes the SAME PUBLIC INTERFACE as ECHELON's HRVQ even though the
internals differ: a per-level ``.frozen`` flag, ``freeze_levels(list)``,
``get_frozen_levels()``, ``get_codebook_usage(indices)``, ``decode_from_indices``
and ``decode_partial``. Only the quantiser internals fork; the surface matches.
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientVQ(nn.Module):
    """Single-level vector quantiser with a GRADIENT-trained nn.Embedding codebook.

    Native IRIS style: the codebook is learned by backprop through the commitment
    loss, not by EMA. ``.frozen`` mirrors ECHELON's flag -- when set, the level
    contributes zero commitment loss (and, paired with requires_grad_(False) in
    freeze_levels, receives no gradient).
    """

    def __init__(self, num_codes: int, embed_dim: int, commitment_cost: float = 1.0) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_codes, embed_dim)
        # Same init scheme as IRIS's stock Tokenizer codebook.
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.frozen = False

    def freeze(self) -> None:
        # Two-part freeze: zero this level's commitment loss AND stop gradient to
        # its codebook. (In ECHELON, frozen=True suffices because the EMA codebook
        # is buffer-updated; here the codebook is an nn.Parameter, so we must also
        # flip requires_grad to truly hold it fixed.)
        self.frozen = True
        self.embedding.weight.requires_grad_(False)

    def unfreeze(self) -> None:
        self.frozen = False
        self.embedding.weight.requires_grad_(True)

    def forward(self, z: torch.Tensor):
        """Quantise ``z`` of shape (*, embed_dim).

        Returns (z_q_st, indices, commitment_loss) where z_q_st is the
        straight-through quantised tensor (forward = codebook vector, gradient =
        pass-through to z), indices is the LongTensor of code ids with z's batch
        prefix, and commitment_loss is a scalar (zero when frozen).
        """
        shape = z.shape
        z_flat = z.reshape(-1, self.embed_dim)

        # L2 nearest-code search (argmin of ||z - e||^2).
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = dist.argmin(dim=-1)
        z_q = self.embedding(indices)  # (N, D), grad-connected to the codebook

        if self.frozen:
            commitment_loss = z_flat.new_zeros(())
        else:
            # IRIS-native VQVAE loss (beta=1.0 by default):
            #   codebook term moves the codes toward the encoder output,
            #   commitment term moves the encoder output toward the codes.
            commitment_loss = (
                (z_flat.detach() - z_q).pow(2).mean()
                + self.commitment_cost * (z_flat - z_q.detach()).pow(2).mean()
            )

        z_q_st = z_flat + (z_q - z_flat).detach()
        return z_q_st.reshape(shape), indices.reshape(shape[:-1]), commitment_loss


class ResidualQuantizer(nn.Module):
    """3-level hierarchical residual VQ (gradient codebooks).

    Residual chain (identical structure to ECHELON's HRVQ):
        L0 quantises z_e            -> z_q0, residual = z_e - z_q0.detach()
        L1 quantises residual       -> z_q1, residual = residual - z_q1.detach()
        L2 quantises residual       -> z_q2
        z_q = STE_on_sum(z_q0 + z_q1 + z_q2)   (gradient flows to z_e only)

    Each level keeps its own commitment loss; the codebooks are trained by the
    gradient through those losses (no EMA).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_codes: List[int] = [512, 512, 512],
        commitment_costs: List[float] = [1.0, 1.0, 1.0],
    ) -> None:
        super().__init__()
        assert len(num_codes) == len(commitment_costs)
        self.embed_dim = embed_dim
        self.num_levels = len(num_codes)
        self.quantizers = nn.ModuleList(
            [GradientVQ(num_codes[i], embed_dim, commitment_costs[i]) for i in range(self.num_levels)]
        )

    def forward(self, z_e: torch.Tensor) -> Dict:
        """Residual-quantise z_e of shape (*, embed_dim).

        Returns dict:
            "z_q":               (*, embed_dim) straight-through summed quantised
            "z_q_levels":        [z_q0, z_q1, z_q2] raw codebook vectors (*, embed_dim)
            "indices":           [idx0, idx1, idx2] each LongTensor (*,)
            "vq_loss":           scalar total commitment loss (sum over levels)
            "commitment_losses": [l0, l1, l2] per-level scalars
        """
        z_q_levels: List[torch.Tensor] = []
        indices_all: List[torch.Tensor] = []
        commitment_losses: List[torch.Tensor] = []

        residual = z_e
        for level in range(self.num_levels):
            _, idx, loss = self.quantizers[level](residual)
            z_q_raw = self.quantizers[level].embedding(idx)  # raw (grad-connected) code vectors
            z_q_levels.append(z_q_raw)
            indices_all.append(idx)
            commitment_losses.append(loss)
            if level < self.num_levels - 1:
                # Detach so the next level's residual does not backprop through
                # earlier codebooks (matches ECHELON's residual chain).
                residual = residual - z_q_raw.detach()

        z_q_sum = sum(z_q_levels)
        # Straight-through on the SUM: forward value is the summed codes, gradient
        # is identity to z_e (the encoder). Codebooks learn via the commitment
        # losses above, not through this path.
        z_q_st = z_e + (z_q_sum - z_e).detach()

        vq_loss = sum(commitment_losses)
        return {
            "z_q": z_q_st,
            "z_q_levels": z_q_levels,
            "indices": indices_all,
            "vq_loss": vq_loss,
            "commitment_losses": commitment_losses,
        }

    # ------------------------------------------------------------------ #
    # ECHELON-parity interface (same surface, gradient internals).        #
    # freeze_levels / get_frozen_levels wired here; requires_grad gating  #
    # and the usage method land in later commits per the port plan.       #
    # ------------------------------------------------------------------ #
    def freeze_levels(self, levels: List[int]) -> None:
        for level in levels:
            assert 0 <= level < self.num_levels, f"Level {level} out of range [0, {self.num_levels})"
            self.quantizers[level].freeze()

    def get_frozen_levels(self) -> List[int]:
        return [i for i in range(self.num_levels) if self.quantizers[i].frozen]

    @torch.no_grad()
    def get_codebook_usage(self, indices: List[torch.Tensor]) -> Dict[str, float]:
        """Per-level codebook usage on a batch (same surface as ECHELON's HRVQ).

        For each level returns:
            usage_{l}       live-code fraction = unique codes used / num_codes
            perplexity_{l}  exp(entropy of the code-usage distribution)
        Perplexity in [1, num_codes]; near 1 means collapse onto one code, near
        num_codes means uniform usage. This is the CPU-observable proxy for
        cascade liveness on GPU.
        """
        stats: Dict[str, float] = {}
        for level in range(self.num_levels):
            idx = indices[level].reshape(-1)
            num_codes = self.quantizers[level].num_codes
            counts = torch.bincount(idx, minlength=num_codes).float()
            probs = counts / counts.sum()
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            stats[f"usage_{level}"] = idx.unique().numel() / num_codes
            stats[f"perplexity_{level}"] = float(perplexity)
        return stats

    def decode_from_indices(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """Codebook lookup + sum across all levels -> (*, embed_dim)."""
        z_q = torch.zeros_like(self.quantizers[0].embedding(indices[0]))
        for level in range(self.num_levels):
            z_q = z_q + self.quantizers[level].embedding(indices[level])
        return z_q

    def decode_partial(self, indices: List[torch.Tensor], up_to_level: int) -> torch.Tensor:
        """Sum levels 0..up_to_level inclusive (for cascade reconstruction)."""
        assert 0 <= up_to_level < self.num_levels
        z_q = torch.zeros_like(self.quantizers[0].embedding(indices[0]))
        for level in range(up_to_level + 1):
            z_q = z_q + self.quantizers[level].embedding(indices[level])
        return z_q
