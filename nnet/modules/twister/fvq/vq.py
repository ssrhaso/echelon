# Copyright 2026, Hasaan Ahmad.
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

"""Flat single-level VQ-EMA: one codebook, no residual levels."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """Single-level vector quantizer with EMA codebook updates and dead code revival."""

    def __init__(
        self,
        num_codes: int,
        embed_dim: int,
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
        """Update the codebook embeddings via EMA."""

        encodings = F.one_hot(indices, self.num_codes).float()
        
        # Per-code count and embedding sum over this batch
        cluster_size = encodings.sum(0)              # (K,)
        embedding_sum = encodings.t() @ z_flat      # (K, D)
        
        # EMA Updates
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embedding_sum.mul_(self.ema_decay).add_(embedding_sum, alpha=1 - self.ema_decay)
        
        # Laplace smoothing to avoid division by zero
        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.epsilon) 
            / (n + self.num_codes * self.epsilon) * n
        )

        # Update codebook
        self.embedding.copy_(self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1))

        self.update_count += 1

    def _revive_dead_codes(
        self, 
        z_flat: torch.Tensor
    ) -> int:
        """Replace dead codebook entries with random encoder outputs."""

        if self.update_count % self.revival_interval != 0:
            return 0

        # A code is dead when its usage falls below the threshold
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
        """Quantize z."""

        shape = z.shape
        z_flat = z.reshape(-1, self.embed_dim)  # (N, D)
        
        # Pairwise distances, under no_grad since they only feed the argmin
        with torch.no_grad():
            distances = (
                z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_flat @ self.embedding.T
                + self.embedding.pow(2).sum(1, keepdim=True).T
            )  # (N, K)

            indices = distances.argmin(dim=1)   # (N,)
            z_q = self.embedding[indices]       # (N, D)

        # EMA codebook update, on detached inputs to keep it out of the graph
        if self.training:
            self._ema_update(z_flat.detach(), indices)
            self._revive_dead_codes(z_flat.detach())
            
        # Commitment loss keeps encoder outputs close to the codebook
        commitment_loss = self.commitment_cost * F.mse_loss(z_flat, z_q.detach())
        
        # Straight-through estimator, so gradient flows to z
        z_q_st = z_flat + (z_q - z_flat).detach()  # (N, D)
        
        # Perplexity, the exponentiated assignment entropy
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(dim = 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Back to the original batch dims
        z_q_st = z_q_st.reshape(shape)
        indices = indices.reshape(shape[:-1])

        return z_q_st, indices, commitment_loss, perplexity


class HRVQ(nn.Module):
    """Single-level quantizer wrapped in the HRVQ interface, with no hierarchy."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_codes: list[int] = [512],
        commitment_costs: list[float] = [0.25],
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        assert len(num_codes) == 1, "flat VQ has a single level"
        self.embed_dim = embed_dim
        self.num_levels = 1

        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                num_codes=num_codes[0],
                embed_dim=embed_dim,
                commitment_cost=commitment_costs[0],
                ema_decay=ema_decay,
                epsilon=epsilon,
            )
        ])

    def forward(
        self, 
        z_e: torch.Tensor
    ) -> dict:
        """Single-level quantization."""

        z_q, indices, vq_loss, perplexity = self.quantizers[0](z_e)

        return {
            "z_q": z_q,
            "z_q_levels": [z_q],
            "indices": [indices],
            "vq_loss": vq_loss,
            "perplexities": [perplexity],
        }

    @torch.no_grad()
    def encode(
        self, 
        z_e: torch.Tensor
    ) -> list[torch.Tensor]:
        """Encode to indices."""

        result = self.forward(z_e)
        return result["indices"]

    def decode_from_indices(
        self, 
        indices: list[torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct z_q from indices."""

        return self.quantizers[0].embedding[indices[0]]

    def decode_partial(
        self, 
        indices: list[torch.Tensor], 
        up_to_level: int
    ) -> torch.Tensor:
        """Decode levels 0..up_to_level, of which only level 0 exists here."""

        assert up_to_level >= 0, "flat VQ only has level 0"

        return self.decode_from_indices(indices)

    @torch.no_grad()
    def get_codebook_usage(
        self, 
        indices: list[torch.Tensor]
    ) -> dict:
        """Codebook usage and perplexity over a batch of indices."""

        idx = indices[0].reshape(-1)
        unique_codes = idx.unique().numel()
        total_codes = self.quantizers[0].num_codes
        
        # Perplexity from the empirical distribution
        counts = torch.bincount(idx, minlength=total_codes).float()
        probs = counts / counts.sum()
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        
        return {
            "usage_0": unique_codes / total_codes,
            "perplexity_0": perplexity.item(),
        }
