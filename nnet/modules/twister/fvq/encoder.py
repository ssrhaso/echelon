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

"""Step 1: EncoderNetwork with single-level VQ.

- hrvq_num_codes=[512] (single level instead of [512, 512, 512])
- hrvq_commitment_costs=[0.25] (single level)
- forward() implemented: CNN -> pre_vq_proj -> HRVQ -> reshape to (32, 32)
"""

import torch.nn as nn

from nnet import modules
from .vq import HRVQ


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
        hrvq_num_codes: list = None,
        hrvq_commitment_costs: list = None,
        hrvq_ema_decay: float = 0.99,
        hrvq_epsilon: float = 1e-5,
    ):
        super(EncoderNetwork, self).__init__()

        self.dim_input_cnn = dim_input_cnn
        self.dim_cnn = dim_cnn
        self.image_size = image_size
        self.dim_concat = 4 * 4 * 8 * dim_cnn
        self.stoch_size = stoch_size
        self.discrete = discrete

        if hrvq_num_codes is None:
            hrvq_num_codes = [512]
        if hrvq_commitment_costs is None:
            hrvq_commitment_costs = [0.25]

        embed_dim = stoch_size * discrete  # = 1024

        # CNN: 64 -> 32 -> 16 -> 8 -> 4
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

        # 4096 -> 1024
        self.pre_vq_proj = modules.Linear(
            in_features=self.dim_concat,
            out_features=embed_dim,
            weight_init=dist_weight_init,
            bias_init=dist_bias_init,
        )

        # Single-level HRVQ (512 codes, 1024-dim)
        self.hrvq = HRVQ(
            embed_dim=embed_dim,
            num_codes=hrvq_num_codes,
            commitment_costs=hrvq_commitment_costs,
            ema_decay=hrvq_ema_decay,
            epsilon=hrvq_epsilon,
        )

    def forward_cnn(self, x):
        """CNN feature extraction. Unchanged from TWISTER """
        shape = x.shape
        x = x.reshape((-1,) + shape[-3:])
        x = self.cnn(x)
        x = x.reshape(shape[:-3] + (4 * 4 * 8 * self.dim_cnn,))
        return x

    def forward(self, inputs):
        """CNN -> pre_vq_proj -> single VQ -> reshaped stoch """
        cnn_out  = self.forward_cnn(inputs)
        z_e      = self.pre_vq_proj(cnn_out)
        hrvq_out = self.hrvq(z_e)
        stoch    = hrvq_out["z_q"].reshape(z_e.shape[:-1] + (self.stoch_size, self.discrete))

        return {
            "stoch":     stoch,
            "hrvq_info": hrvq_out,
        }
