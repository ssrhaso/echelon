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

"""EMA vector quantizer and the multi-level residual HRVQ built from it."""

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

        # Freeze flag: when True, skip EMA updates and zero commitment loss
        self.frozen = False

    def freeze(self):
        """Disable EMA updates and commitment loss."""
        self.frozen = True

    def unfreeze(self):
        """Re-enable EMA updates and commitment loss."""
        self.frozen = False

    def _ema_update(
        self, z_flat: torch.Tensor,
        indices: torch.Tensor
    ) -> None:
        """Update the codebook embeddings via EMA."""
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
        """Replace dead codebook entries with random encoder outputs."""
        if self.update_count % self.revival_interval != 0:
            return 0

        dead_mask = self.ema_cluster_size < self.revival_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0:
            rand_indices = torch.randint(0, z_flat.shape[0], (num_dead,), device=z_flat.device)
            # Masked index-assignment needs matching dtypes, and the codebook
            # buffer stays fp32 while z_flat may be bf16 under autocast.
            self.embedding[dead_mask] = z_flat[rand_indices].detach().to(self.embedding.dtype)
            self.ema_cluster_size[dead_mask] = self.revival_threshold
            self.ema_embedding_sum[dead_mask] = self.embedding[dead_mask] * self.revival_threshold

        return num_dead

    def forward(
        self,
        z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize z.

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

        if self.training and not self.frozen:
            self._ema_update(z_flat.detach(), indices)
            self._revive_dead_codes(z_flat.detach())

        if self.frozen:
            commitment_loss = torch.tensor(0.0, device=z_flat.device, dtype=z_flat.dtype)
        else:
            commitment_loss = self.commitment_cost * F.mse_loss(z_flat, z_q.detach())

        z_q_st = z_flat + (z_q - z_flat).detach()

        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        z_q_st = z_q_st.reshape(shape)
        indices = indices.reshape(shape[:-1])

        return z_q_st, indices, commitment_loss, perplexity


class HRVQ(nn.Module):
    """Hierarchical residual vector quantization.

    Level l quantizes the residual left by levels 0..l-1, and the output is the
    sum over levels with straight-through gradients to z_e.
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
            "residual_errors": [e0, e1, e2] per-level ||r_l - z_q_l||^2 / ||r_l||^2
        """
        z_q_levels = []
        indices_all = []
        total_vq_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        perplexities = []
        residual_errors = []

        residual = z_e
        for level in range(self.num_levels):
            z_q_st_level, indices_level, loss_level, perp_level = self.quantizers[level](residual)
            # Raw lookup, not straight-through: the residual needs the
            # undifferentiated quantized vector.
            z_q_raw = self.quantizers[level].embedding[indices_level]
            z_q_levels.append(z_q_raw)
            indices_all.append(indices_level)
            total_vq_loss = total_vq_loss + loss_level
            perplexities.append(perp_level)

            # Relative residual error at this level, log-only.
            with torch.no_grad():
                num = (residual.detach() - z_q_raw.detach()).pow(2).sum(dim=-1)
                den = residual.detach().pow(2).sum(dim=-1).clamp_min(1e-8)
                residual_errors.append((num / den).mean())

            # Residual passed to the next level.
            if level < self.num_levels - 1:
                residual = residual - z_q_raw.detach()

        # Straight-through on the sum, so gradient flows to z_e only.
        z_q_sum = sum(zq.detach() for zq in z_q_levels)
        z_q_st = z_e + (z_q_sum - z_e).detach()

        return {
            "z_q": z_q_st,
            "z_q_levels": z_q_levels,
            "indices": indices_all,
            "vq_loss": total_vq_loss,
            "perplexities": perplexities,
            "residual_errors": residual_errors,
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

    def freeze_levels(self, levels: list[int]):
        """Freeze the given levels."""
        for level in levels:
            assert 0 <= level < self.num_levels, f"Level {level} out of range [0, {self.num_levels})"
            self.quantizers[level].freeze()

    def get_frozen_levels(self) -> list[int]:
        """Indices of the frozen levels."""
        return [i for i in range(self.num_levels) if self.quantizers[i].frozen]
