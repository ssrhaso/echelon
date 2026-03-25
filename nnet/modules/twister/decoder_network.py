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
from nnet import distributions

class DecoderNetwork(nn.Module):

    def __init__(
        self, 
        feat_size=32*32+512, 
        dim_cnn=32, 
        dim_output_cnn=3, 
        act_fun=nn.SiLU, 
        weight_init="dreamerv3_normal", 
        bias_init="zeros", 
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
        dist_weight_init="xavier_uniform", 
        dist_bias_init="zeros",
        image_size=(64, 64)
    ):
        super(DecoderNetwork, self).__init__()

        self.feat_size = feat_size
        self.dim_output_cnn = dim_output_cnn
        self.dim_cnn = dim_cnn
        self.image_size = image_size

            
        # CNN proj
        self.proj = modules.Linear(feat_size, 8 * dim_cnn * 4 * 4, weight_init="xavier_uniform", bias_init="zeros")
        self.dim_proj_out = (8 * dim_cnn, 4, 4)

        # CNN
        # 4 -> 8 -> 16 -> 32 -> 64
        dim_layers = [4*dim_cnn, 2*dim_cnn, 1*dim_cnn, dim_output_cnn]
        self.cnn = modules.ConvTransposeNeuralNetwork(  
            dim_input=8*dim_cnn,
            dim_layers=dim_layers,
            kernel_size=4,
            strides=2,
            act_fun=[act_fun for _ in range(len(dim_layers)-1)] + [None],
            weight_init=[weight_init for _ in range(len(dim_layers)-1)] + [dist_weight_init],
            bias_init=[bias_init for _ in range(len(dim_layers)-1)] + [dist_bias_init],
            norm=[cnn_norm for _ in range(len(dim_layers)-1)] + [None],
            bias=[cnn_norm is None for _ in range(len(dim_layers)-1)] + [True],
            padding=1,
            output_padding=0,
            channels_last=False
        )

    def forward_cnn(self, x):

        # (B, N, D) -> (B, N, C * 4 * 4)
        x = self.proj(x)

        # (B, N, C * 4 * 4) -> (B * N, C, 4 * 4)
        shape = x.shape
        x = x.reshape((-1,) + self.dim_proj_out)

        # (B * N, C, 4, 4) -> (B * N, 3, 64, 64)
        x = self.cnn(x)

        # (B * N, 3, 64, 64) -> (B, N, 3, 64, 64)
        x = x.reshape(shape[:-1] + x.shape[1:])

        # Normal Distribution
        obs_dist = distributions.MSEDist(x, reinterpreted_batch_ndims=3)

        return obs_dist
    
    def forward(self, inputs):

        # Outputs
        outputs = self.forward_cnn(inputs)

        return outputs