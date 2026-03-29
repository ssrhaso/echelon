# Copyright 2025, Hasaan Ahmad.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spatial HRVQ: VectorQuantizerEMA + multi-level residual HRVQ.

VectorQuantizerEMA: copied from fvq/vq.py with default embed_dim=256.
HRVQ: supports num_codes=[512,512,512] for 3-level residual quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Single-level vector quantizer with EMA codebook updates and dead code revival."""

    def __init__(
        self,
        num_codes: int,
        embed_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        revival_interval: int = 100,
        revival_threshold: float = 1.0,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.revival_interval = revival_interval
        self.revival_threshold = revival_threshold

        # Codebook Embeddings : (K, D)
        embedding = torch.randn(num_codes, embed_dim)
        self.register_buffer('embedding', embedding)

        # EMA Tracking Buffers
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embedding_sum", embedding.clone())
        self.register_buffer("update_count", torch.tensor(0))

    def _ema_update(
        self, z_flat: torch.Tensor,
        indices: torch.Tensor
    ) -> None:
        """Update Codebook Embeddings via EMA."""
        encodings = F.one_hot(indices, self.num_codes).float()

        cluster_size = encodings.sum(0)
        embedding_sum = encodings.t() @ z_flat

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embedding_sum.mul_(self.ema_decay).add_(embedding_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_codes * self.epsilon) * n
        )

        self.embedding.copy_(self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1))

        self.update_count += 1

    def _revive_dead_codes(
        self,
        z_flat: torch.Tensor
    ) -> int:
        """Replace Dead Codebook entries with random encoder outputs."""
        if self.update_count % self.revival_interval != 0:
            return 0

        dead_mask = self.ema_cluster_size < self.revival_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0:
            rand_indices = torch.randint(0, z_flat.shape[0], (num_dead,), device=z_flat.device)
            self.embedding[dead_mask] = z_flat[rand_indices].detach()
            self.ema_cluster_size[dead_mask] = self.revival_threshold
            self.ema_embedding_sum[dead_mask] = self.embedding[dead_mask] * self.revival_threshold

        return num_dead

    def forward(
        self,
        z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input tensor.

        Args:
            z: (*, embed_dim) continuous embeddings
        Returns:
            z_q_st: (*, embed_dim) straight-through quantized
            indices: (*,) codebook indices
            commitment_loss: scalar
            perplexity: scalar
        """
        shape = z.shape
        z_flat = z.reshape(-1, self.embed_dim)

        with torch.no_grad():
            distances = (
                z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_flat @ self.embedding.T
                + self.embedding.pow(2).sum(1, keepdim=True).T
            )

            indices = distances.argmin(dim=1)
            z_q = self.embedding[indices]

        if self.training:
            self._ema_update(z_flat.detach(), indices)
            self._revive_dead_codes(z_flat.detach())

        commitment_loss = self.commitment_cost * F.mse_loss(z_flat, z_q.detach())

        z_q_st = z_flat + (z_q - z_flat).detach()

        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        z_q_st = z_q_st.reshape(shape)
        indices = indices.reshape(shape[:-1])

        return z_q_st, indices, commitment_loss, perplexity


class HRVQ(nn.Module):
    """Hierarchical Residual Vector Quantization.

    Supports num_codes=[512] (single level, for ablation) and
    num_codes=[512,512,512] (3 levels, full spatial HRVQ).

    Residual chain:
        L0 quantizes z_e              -> z_q0, residual0 = z_e - z_q0
        L1 quantizes residual0        -> z_q1, residual1 = residual0 - z_q1
        L2 quantizes residual1        -> z_q2
        z_q = z_q0 + z_q1 + z_q2 (straight-through on the sum)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_codes: list[int] = [512, 512, 512],
        commitment_costs: list[float] = [0.25, 0.5, 1.0],
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        assert len(num_codes) == len(commitment_costs)
        self.embed_dim = embed_dim
        self.num_levels = len(num_codes)

        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                num_codes=num_codes[i],
                embed_dim=embed_dim,
                commitment_cost=commitment_costs[i],
                ema_decay=ema_decay,
                epsilon=epsilon,
            )
            for i in range(self.num_levels)
        ])

    def forward(self, z_e: torch.Tensor) -> dict:
        """Hierarchical residual quantization.

        Args:
            z_e: (*, 256) continuous embeddings (any batch prefix)
        Returns dict:
            "z_q": (*, 256) summed quantized with straight-through
            "z_q_levels": [z_q0, z_q1, z_q2] each (*, 256), raw codebook vectors
            "indices": [idx0, idx1, idx2] each (*,) LongTensor
            "vq_loss": scalar total commitment loss
            "perplexities": [perp0, perp1, perp2] per-level scalars
        """
        z_q_levels = []
        indices_all = []
        total_vq_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        perplexities = []

        residual = z_e
        for level in range(self.num_levels):
            # Quantize the current residual
            z_q_st_level, indices_level, loss_level, perp_level = self.quantizers[level](residual)
            # Raw z_q for this level (detached codebook lookup, NOT straight-through)
            # We need the raw quantized vector for residual computation
            z_q_raw = self.quantizers[level].embedding[indices_level]
            z_q_levels.append(z_q_raw)
            indices_all.append(indices_level)
            total_vq_loss = total_vq_loss + loss_level
            perplexities.append(perp_level)

            # Compute residual for next level using raw z_q (detached from codebook)
            if level < self.num_levels - 1:
                residual = residual - z_q_raw.detach()

        # Straight-through on the SUM: gradient flows to z_e only
        z_q_sum = sum(zq.detach() for zq in z_q_levels)
        z_q_st = z_e + (z_q_sum - z_e).detach()

        return {
            "z_q": z_q_st,
            "z_q_levels": z_q_levels,
            "indices": indices_all,
            "vq_loss": total_vq_loss,
            "perplexities": perplexities,
        }

    @torch.no_grad()
    def encode(self, z_e: torch.Tensor) -> list[torch.Tensor]:
        """Encode to indices only (no grad)."""
        result = self.forward(z_e)
        return result["indices"]

    def decode_from_indices(self, indices: list[torch.Tensor]) -> torch.Tensor:
        """Codebook lookup + sum across levels. Returns (*, embed_dim)."""
        z_q = torch.zeros_like(self.quantizers[0].embedding[indices[0]])
        for level in range(self.num_levels):
            z_q = z_q + self.quantizers[level].embedding[indices[level]]
        return z_q

    def decode_partial(self, indices: list[torch.Tensor], up_to_level: int) -> torch.Tensor:
        """Sum levels 0..up_to_level. For cascade reconstruction."""
        assert 0 <= up_to_level < self.num_levels
        z_q = torch.zeros_like(self.quantizers[0].embedding[indices[0]])
        for level in range(up_to_level + 1):
            z_q = z_q + self.quantizers[level].embedding[indices[level]]
        return z_q

    @torch.no_grad()
    def get_codebook_usage(self, indices: list[torch.Tensor]) -> dict:
        """Per-level unique codes, usage %, perplexity."""
        stats = {}
        for level in range(self.num_levels):
            idx = indices[level].reshape(-1)
            unique_codes = idx.unique().numel()
            total_codes = self.quantizers[level].num_codes
            counts = torch.bincount(idx, minlength=total_codes).float()
            probs = counts / counts.sum()
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            stats[f"usage_{level}"] = unique_codes / total_codes
            stats[f"perplexity_{level}"] = perplexity.item()
        return stats
