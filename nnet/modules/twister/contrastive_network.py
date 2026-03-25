# Copyright 2025, Maxime Burchi.
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

class ContrastiveNetwork(nn.Module):

    def __init__(
        self, 
        hidden_size=512, 
        out_size=512,
        feat_size=32*32+512, 
        embed_size=4*4*8*32,
        act_fun="ELU",
        num_layers=2
    ):
        super(ContrastiveNetwork, self).__init__()

        self.mlp_feats = modules.MultiLayerPerceptron(
            dim_input=feat_size, 
            dim_layers=[hidden_size for _ in range(num_layers-1)] + [out_size], 
            act_fun=act_fun
        )
        self.mlp_embed = modules.MultiLayerPerceptron(
            dim_input=embed_size, 
            dim_layers=[hidden_size for _ in range(num_layers-1)] + [out_size], 
            act_fun=act_fun
        )

    def forward(self, feats, embed):

        # print(feats.shape, embed.shape)

        # MLP Layers
        x = self.mlp_feats(feats)
        y = self.mlp_embed(embed)

        return x, y
    