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

""" 
FLAT SINGLE LEVEL VQ-EMA 

No Residual, No Structure, No Cascade, just one codebook.
"""
# PyTorch
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
        # Initialise from uniform (VQ-VAE paper)
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
        """ Update Codebook Embeddings via EMA """
        
        # ONEHOT ENCODING Assignments (N, K)
        encodings = F.one_hot(indices, self.num_codes).float()
        
        # Per code count and embedding sum (for this batch)
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

        # Update Codebook 
        self.embedding.copy_(self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1))
        
        # Increment Counter 
        self.update_count += 1
        
        pass

    def _revive_dead_codes(
        self, 
        z_flat: torch.Tensor
    ) -> int:
        """ Replace Dead Codebook entries with random encoder outputs """
        
        if self.update_count % self.revival_interval != 0:
            return 0

        # IF DEAD (usage < threshold)
        dead_mask = self.ema_cluster_size < self.revival_threshold
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            # SAMPLE random encoder outputs as replacements
            rand_indices = torch.randint(0, z_flat.shape[0], (num_dead,), device=z_flat.device)
            self.embedding[dead_mask] = z_flat[rand_indices].detach()
            self.ema_cluster_size[dead_mask] = self.revival_threshold
            self.ema_embedding_sum[dead_mask] = self.embedding[dead_mask] * self.revival_threshold
            
        return num_dead

    def forward(
        self, 
        z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input tensor """
        
        shape = z.shape
        z_flat = z.reshape(-1, self.embed_dim)  # (N, D)
        
        # PAIRWISE DISTANCES 
        distances = (
            z.flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.T
            + self.embedding.pow(2).sum(1, keepdim=True).T
        )  # (N, K)
        
        # Nearest Codebook Update (Training Only)
        indices = distances.argmin(dim=1)   # (N,)
        z_q = self.embedding[indices]       # (N, D) Quantized Vector  
        
        # EMA Codebook Update (Training Only)
        if self.training:
            self._ema_update(z_flat, indices)
            self._revive_dead_codes(z_flat)
            
        # Commitment Loss 
        # (Encourage encoder outputs to stay close to codebook vectors)
        commitment_loss = self.commitment_cost * F.mse_loss(z_flat, z_q.detach())
        
        # Straight-Through estimator: gradient flows through z_q as if it were z
        z_q_st = z_flat + (z_q - z_flat).detach()  # (N, D)
        
        # Perplexity exp(entropy) - measures effective codebook usage
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(dim = 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # RESHAPE BACK to original batch dimensions
        z_q_st = z_q_st.reshape(shape)
        indices = indices.reshape(shape[:-1])
        
        return z_q_st, indices, commitment_loss, perplexity

        pass


class HRVQ(nn.Module):
    """Hierarchical Residual Vector Quantization with 3 codebook levels."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_codes: list[int] = [512, 512, 512],
        commitment_costs: list[float] = [0.25, 0.5, 1.0],
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        pass

    def forward(
        self, 
        z_e: torch.Tensor
    ) -> dict:
        """Hierarchical residual quantization """
        
        pass

    @torch.no_grad()
    def encode(
        self, 
        z_e: torch.Tensor
    ) -> list[torch.Tensor]:
        """Encode to indices only """
        
        pass

    def decode_from_indices(
        self, 
        indices: list[torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct z_q from indices. Returns summed z_q."""
        
        pass

    def decode_partial(
        self, 
        indices: list[torch.Tensor], 
        up_to_level: int
    ) -> torch.Tensor:
        """Decode using only levels 0..up_to_level"""
        
        pass

    @torch.no_grad()
    def get_codebook_usage(
        self, 
        indices: list[torch.Tensor]
    ) -> dict:
        """Per-level codebook usage statistics"""
        
        pass


if __name__ == "__main__":
    pass
