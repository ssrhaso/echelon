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

"""Spatial HRVQ Encoder: CNN -> spatial tokens -> shared 3-level HRVQ -> aggregate.

Replaces fvq/encoder.py. Same CNN, but stops before flattening to produce
16 spatial tokens of 256-dim each, applies shared HRVQ, then aggregates
to (32, 32) stoch for TSSM compatibility.
"""

import torch.nn as nn

from nnet import modules
from .vq import HRVQ


class SpatialHRVQEncoder(nn.Module):

    def __init__(
        self,
        dim_input_cnn=3,
        dim_cnn=32,
        act_fun=nn.SiLU,
        weight_init="dreamerv3_normal",
        bias_init="zeros",
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3}},
        image_size=(64, 64),
        stoch_size=32,
        discrete=32,
        dist_weight_init="xavier_uniform",
        dist_bias_init="zeros",
        # Spatial HRVQ params
        num_positions=16,
        position_dim=256,
        hrvq_num_codes: list = None,
        hrvq_commitment_costs: list = None,
        hrvq_ema_decay: float = 0.99,
        hrvq_epsilon: float = 1e-5,
    ):
        super(SpatialHRVQEncoder, self).__init__()

        self.dim_input_cnn = dim_input_cnn
        self.dim_cnn = dim_cnn
        self.image_size = image_size
        self.stoch_size = stoch_size
        self.discrete = discrete
        self.num_positions = num_positions
        self.position_dim = position_dim  # = 8 * dim_cnn = 256

        if hrvq_num_codes is None:
            hrvq_num_codes = [512, 512, 512]
        if hrvq_commitment_costs is None:
            hrvq_commitment_costs = [0.25, 0.5, 1.0]

        # CNN: 64 -> 32 -> 16 -> 8 -> 4, output (N, 256, 4, 4)
        self.cnn = modules.ConvNeuralNetwork(
            dim_input=self.dim_input_cnn,
            dim_layers=[dim_cnn, 2 * dim_cnn, 4 * dim_cnn, 8 * dim_cnn],
            kernel_size=4,
            strides=2,
            act_fun=act_fun,
            padding="same",
            weight_init=weight_init,
            bias_init=bias_init,
            norm=cnn_norm,
            channels_last=False,
            bias=cnn_norm is None
        )

        # No pre_vq_proj: CNN outputs 256 channels = position_dim = embed_dim

        # Shared 3-level HRVQ (256-dim per position)
        self.hrvq = HRVQ(
            embed_dim=position_dim,
            num_codes=hrvq_num_codes,
            commitment_costs=hrvq_commitment_costs,
            ema_decay=hrvq_ema_decay,
            epsilon=hrvq_epsilon,
        )

        # Aggregate spatial tokens to stoch: Linear(16*256=4096 -> 32*32=1024)
        self.spatial_aggregate = modules.Linear(
            in_features=num_positions * position_dim,
            out_features=stoch_size * discrete,
            weight_init=dist_weight_init,
            bias_init=dist_bias_init,
        )

    def forward_cnn(self, x):
        """CNN feature extraction — STOP before flattening.

        Args:
            x: (B, L, C, H, W) or (B, C, H, W)
        Returns:
            spatial_features: (*, 16, 256) spatial grid, 256-dim per position
        """
        shape = x.shape
        x = x.reshape((-1,) + shape[-3:])                    # (N, 3, 64, 64)
        x = self.cnn(x)                                       # (N, 256, 4, 4)
        x = x.reshape(x.shape[0], self.position_dim, -1)      # (N, 256, 16)
        x = x.permute(0, 2, 1)                                # (N, 16, 256)
        x = x.reshape(shape[:-3] + (self.num_positions, self.position_dim))
        return x

    def forward(self, inputs):
        """Full encoder: CNN -> spatial HRVQ -> aggregate -> stoch.

        Args:
            inputs: (B, L, C, H, W) or (B, C, H, W)
        Returns dict:
            "stoch": (*, 32, 32) aggregated, reshaped for TSSM compatibility
            "hrvq_info": dict with spatial HRVQ outputs
            "pre_vq_features": (*, 1024) continuous features for contrastive
        """
        # 1. CNN spatial features: (*, 16, 256)
        spatial_features = self.forward_cnn(inputs)

        # 2. Apply shared HRVQ to all 16 positions simultaneously
        batch_shape = spatial_features.shape[:-2]
        flat = spatial_features.reshape(-1, self.position_dim)  # (N*16, 256)
        hrvq_out = self.hrvq(flat)

        # 3. Reshape HRVQ outputs back to spatial
        z_q_spatial = hrvq_out["z_q"].reshape(batch_shape + (self.num_positions, self.position_dim))
        z_q_levels_spatial = [
            zq.reshape(batch_shape + (self.num_positions, self.position_dim))
            for zq in hrvq_out["z_q_levels"]
        ]
        indices_spatial = [
            idx.reshape(batch_shape + (self.num_positions,))
            for idx in hrvq_out["indices"]
        ]

        # 4. Aggregate: (*, 16, 256) -> flatten (*, 4096) -> Linear -> (*, 1024)
        z_q_flat = z_q_spatial.reshape(batch_shape + (self.num_positions * self.position_dim,))
        aggregated = self.spatial_aggregate(z_q_flat)

        # 5. Reshape to TSSM stoch: (*, 1024) -> (*, 32, 32)
        stoch = aggregated.reshape(batch_shape + (self.stoch_size, self.discrete))

        # 6. Pre-VQ features for contrastive (continuous, shares aggregate Linear)
        pre_vq_flat = spatial_features.reshape(batch_shape + (self.num_positions * self.position_dim,))
        pre_vq_aggregated = self.spatial_aggregate(pre_vq_flat)

        return {
            "stoch": stoch,
            "hrvq_info": {
                "z_q_spatial": z_q_spatial,
                "z_q_levels_spatial": z_q_levels_spatial,
                "indices": indices_spatial,
                "vq_loss": hrvq_out["vq_loss"],
                "perplexities": hrvq_out["perplexities"],
            },
            "pre_vq_features": pre_vq_aggregated,
        }
