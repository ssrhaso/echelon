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
import torch.nn as nn

# Neural Nets
from nnet import modules

class TransformerBlock(nn.Module):

    def __init__(
        self, 
        dim_model, 
        att_params, 
        ff_ratio=4, 
        drop_rate=0.1, 
        inner_dropout=False, 
        act_fun="GELU", 
        weight_init="normal_02", 
        bias_init="zeros", 
        module_pre_norm=True,
        post_norm=False
    ):
        super(TransformerBlock, self).__init__()

        # Muti-Head Self-Attention Module
        self.self_att_module = modules.AttentionModule(
            dim_model=dim_model,
            attention=att_params,
            drop_rate=drop_rate,
            residual=True,
            pre_norm=module_pre_norm
        )

        # Feed Forward Module
        self.ff_module = modules.FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio, 
            drop_rate=drop_rate, 
            act_fun=act_fun,
            inner_dropout=inner_dropout,
            weight_init=weight_init,
            bias_init=bias_init,
            residual=True,
            pre_norm=module_pre_norm
        )

        # Post Norm
        self.post_norm = nn.LayerNorm(normalized_shape=dim_model) if post_norm and module_pre_norm else nn.Identity()

    def forward(self, x, mask=None, hidden=None, return_hidden=False, return_att_w=False):


        # Muti-Head Self-Attention Module
        outputs = self.self_att_module(x, mask=mask, hidden=hidden, return_hidden=return_hidden, return_att_w=return_att_w)

        # Feed Forward Module
        outputs.x = self.ff_module(outputs.x)

        # Post Norm
        outputs.x = self.post_norm(outputs.x)

        return outputs