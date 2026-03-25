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
from nnet.utils import get_module_and_params

class FeedForwardModule(nn.Module):

    """Transformer Feed Forward Module, can be used as prenorm FFN and postnorm FFN

    Args:
        dim_model: model feature dimension
        dim_ffn: expanded feature dimension
        drop_rate: dropout probability
        act_fun: inner activation function
        inner_dropout: whether to apply dropout after the inner activation function
        pre_norm: pre norm vs post norm
        residual: apply rsidual connection, required for post norm

    Input: (batch size, length, dim_model)
    Output: (batch size, length, dim_model)
    
    """

    def __init__(
        self, 
        dim_model, 
        dim_ffn, 
        drop_rate, 
        act_fun, 
        inner_dropout, 
        pre_norm=True, 
        residual=False,
        weight_init="default", 
        bias_init="default"
    ):
        super(FeedForwardModule, self).__init__()

        assert not (not residual and not pre_norm), "post norm need residual connection: residual={}, pre_norm={}".format(residual, pre_norm)
        self.residual = residual

        # Get act_fun and norm
        act_fun, act_fun_params = get_module_and_params(act_fun, modules.act_dict)

        # Pre Norm
        self.pre_norm = modules.LayerNorm(dim_model, eps=1e-6) if pre_norm else nn.Identity()

        # Layers
        self.layers = nn.Sequential(
            # Expand
            modules.Linear(dim_model, dim_ffn, weight_init=weight_init, bias_init=bias_init),
            act_fun(**act_fun_params),
            modules.Dropout(p=drop_rate) if inner_dropout else nn.Identity(),

            # Proj
            modules.Linear(dim_ffn, dim_model, weight_init=weight_init, bias_init=bias_init),
            modules.Dropout(p=drop_rate)
        )

        # Post Norm
        self.post_norm = nn.Identity() if pre_norm else modules.LayerNorm(dim_model, eps=1e-6)

    def forward(self, x):

        # Residual
        if self.residual:
            x_res = x

        # Pre Norm
        x = self.pre_norm(x)

        # Layers
        x = self.layers(x)

        # Residual
        if self.residual:
            x = x + x_res

        # Post Norm
        x = self.post_norm(x)

        return x