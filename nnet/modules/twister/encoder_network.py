# Copyright 2025, Maxime Burchi.
# Copyright 2025, Hasaan Ahmad.  # ECHELON modifications
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

# PyTorch
import torch
import torch.nn as nn

# NeuralNets
from nnet import modules
from nnet import distributions

# ECHELON
from .fvq import HRVQ


class EncoderNetwork(nn.Module):

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
        # ECHELON: killed — uniform_mix removed, VQ has no logit-based distribution
        # ECHELON: new HRVQ params
        hrvq_num_codes: list = None,
        hrvq_commitment_costs: list = None,
        hrvq_ema_decay: float = 0.99,
        hrvq_epsilon: float = 1e-5,
    ):
        super(EncoderNetwork, self).__init__()

        # Params
        self.dim_input_cnn = dim_input_cnn
        self.dim_cnn = dim_cnn
        self.image_size = image_size
        self.dim_concat = 4*4*8*dim_cnn
        self.stoch_size = stoch_size
        self.discrete = discrete
        # ECHELON: killed — self.uniform_mix no longer used (VQ has no logit-based distribution)

        # Defaults
        if hrvq_num_codes is None:
            hrvq_num_codes = [512, 512, 512]
        if hrvq_commitment_costs is None:
            hrvq_commitment_costs = [0.25, 0.5, 1.0]

        embed_dim = stoch_size * discrete  # = 1024 for compatibility

        # 64 -> 32 -> 16 -> 8 -> 4
        self.cnn = modules.ConvNeuralNetwork(
            dim_input=self.dim_input_cnn,
            dim_layers=[dim_cnn, 2*dim_cnn, 4*dim_cnn, 8*dim_cnn],
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

        # ECHELON: renamed from representation_network, same dims, feeds HRVQ instead of reshape+softmax
        self.pre_vq_proj = modules.Linear(
            in_features=self.dim_concat,
            out_features=embed_dim,
            weight_init=dist_weight_init,
            bias_init=dist_bias_init,
        )

        # ECHELON: HRVQ module — 3-level hierarchical residual vector quantization
        self.hrvq = HRVQ(
            embed_dim=embed_dim,
            num_codes=hrvq_num_codes,
            commitment_costs=hrvq_commitment_costs,
            ema_decay=hrvq_ema_decay,
            epsilon=hrvq_epsilon,
        )

    def get_dist(self, state):
        # ECHELON: killed — HRVQ uses codebook lookup, not logit-based distributions
        raise NotImplementedError("ECHELON: removed — no logits in VQ encoder")

    def forward_cnn(self, x):
        """Unchanged from TWISTER. CNN feature extraction.

        Args:
            x: (B, L, C, H, W) or (B, C, H, W) image observations
        Returns:
            features: (*, 4096) flattened CNN features
        """
        shape = x.shape

        # (B, L, C, H, W) -> (B*L, C, H, W) / (B, C, H, W) -> (B, C, H, W)
        x = x.reshape((-1,) + shape[-3:])

        # (N, C, 64, 64) -> (N, C, 4, 4)
        x = self.cnn(x)

        # (N, C, 4, 4) -> (B, L, 4*4 * C) / (N, C, 4, 4) -> (N, 4*4 * C)
        x = x.reshape(shape[:-3] + (4*4 * 8*self.dim_cnn,))

        return x

    def forward(self, inputs):
        """ECHELON encoder: CNN -> projection -> HRVQ -> reshaped stoch.

        Args:
            inputs: (B, L, C, H, W) or (B, C, H, W) image observations
        Returns dict:
            "stoch": (*, 32, 32) — summed z_q reshaped for TSSM compatibility.
                     NOT semantically 32 features x 32 classes — just a 1024-dim
                     vector reshaped to match TWISTER's stoch shape convention.
            "hrvq_info": dict with:
                "z_q_levels": list of 3 tensors each (*, 1024) — per-level quantized
                "indices": list of 3 LongTensors each (*,) — codebook indices
                "vq_loss": scalar — total commitment+codebook loss
                "perplexities": list of 3 scalars

        Flow:
            1. forward_cnn(inputs) -> (*, 4096)
            2. self.pre_vq_proj(cnn_out) -> (*, 1024)
            3. self.hrvq(projected) -> dict with z_q, z_q_levels, indices, vq_loss, perplexities
            4. Reshape z_q from (*, 1024) to (*, 32, 32) -> "stoch"
            5. Package hrvq outputs into "hrvq_info" sub-dict
        """
        raise NotImplementedError("ECHELON: encoder forward — CNN -> pre_vq_proj -> HRVQ -> reshape to (32,32)")
