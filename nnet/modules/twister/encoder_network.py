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
        uniform_mix=0.01,
    ):
        super(EncoderNetwork, self).__init__()

        # Params
        self.dim_input_cnn = dim_input_cnn
        self.dim_cnn = dim_cnn
        self.image_size = image_size
        self.dim_concat = 4*4*8*dim_cnn
        self.stoch_size = stoch_size
        self.discrete = discrete
        self.uniform_mix = uniform_mix

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

        # Representation Network: concat features -> logits
        self.representation_network = modules.Linear(
            in_features=self.dim_concat,
            out_features=discrete * stoch_size if discrete else 2 * stoch_size,
            weight_init=dist_weight_init, 
            bias_init=dist_bias_init, 
        )

    def get_dist(self, state):

        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def forward_cnn(self, x):

        shape = x.shape

        # (B, L, C, H, W) -> (B*L, C, H, W) / (B, C, H, W) -> (B, C, H, W)
        x = x.reshape((-1,) + shape[-3:])

        # (N, C, 64, 64) -> (N, C, 4, 4)
        x = self.cnn(x)

        # (N, C, 4, 4) -> (B, L, 4*4 * C) / (N, C, 4, 4) -> (N, 4*4 * C)
        x = x.reshape(shape[:-3] + (4*4 * 8*self.dim_cnn,))

        return x
    
    def forward(self, inputs):

        outputs = self.forward_cnn(inputs)

        # Dist params
        dist_params = {'logits': self.representation_network(outputs).reshape(outputs.shape[:-1] + (self.stoch_size, self.discrete))}

        # Sample
        stoch = self.get_dist(dist_params).rsample()

        # Return State
        return {"stoch": stoch, **dist_params}