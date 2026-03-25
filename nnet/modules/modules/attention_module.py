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

# Other
import inspect

class AttentionModule(nn.Module):

    """ Attention Module

    Args:
        dim_model: model feature dimension
        att_params: attention params
        drop_rate: residual dropout probability
        norm: module norm params
        pre_norm: pre norm vs post norm
        residual: if residual module or not
        channels_last: if channels_last input or not

    """

    def __init__(
            self, 
            dim_model, 
            attention={"class": "MultiHeadAttention", "params": {"num_heads": 4, "attn_drop_rate": 0.0}}, 
            drop_rate=0.0, 
            norm={"class": "LayerNorm", "params": {"eps": 1e-6}}, 
            pre_norm=True, 
            residual=True, 
            channels_last=True
        ):
        super(AttentionModule, self).__init__()

        # Params
        assert not (not residual and not pre_norm), "post norm need residual connection: residual={}, pre_norm={}".format(residual, pre_norm)
        self.residual = residual

        # Norm
        if isinstance(norm, nn.Module):
            self.pre_norm = norm if pre_norm else nn.Identity()
            self.post_norm = nn.Identity() if pre_norm else norm
        elif isinstance(norm, dict):
            self.pre_norm = modules.norm_dict[norm["class"]](dim_model, **norm["params"], channels_last=channels_last) if pre_norm else nn.Identity()
            self.post_norm = nn.Identity() if pre_norm else modules.norm_dict[norm["class"]](dim_model, **norm["params"], channels_last=channels_last)
        else:
            self.pre_norm = modules.norm_dict[norm](dim_model, channels_last=channels_last) if pre_norm else nn.Identity()
            self.post_norm = nn.Identity() if pre_norm else modules.norm_dict[norm](dim_model, channels_last=channels_last)

        # Attention
        if isinstance(attention, nn.Module):
            self.attention = attention
        else:
            self.attention = modules.att_dict[attention["class"]](dim_model=dim_model, **attention["params"], channels_last=channels_last)
            
        # Dropout
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, x_cross=None, mask=None, hidden=None, return_hidden=False, return_att_w=False):

        # Residual
        if self.residual:
            x_res = x

        # Pre Norm
        x = self.pre_norm(x)

        # Prepare kargs
        kargs = {}
        kargs["Q"] = x
        kargs["K"] = x_cross if x_cross is not None else x
        kargs["V"] = x_cross if x_cross is not None else x
        inspect_args = inspect.getfullargspec(self.attention.forwardQKV).args
        if "mask" in inspect_args:
            kargs["mask"] = mask
        if "hidden" in inspect_args:
            kargs["hidden"] = hidden
        if "return_hidden" in inspect_args:
            kargs["return_hidden"] = return_hidden
        if "return_att_w" in inspect_args:
            kargs["return_att_w"] = return_att_w
            
        # Attention
        outputs = self.attention.forwardQKV(**kargs)

        # Dropout
        outputs.x = self.dropout(outputs.x)

        # Residual
        if self.residual:
            outputs.x = outputs.x + x_res

        # Post Norm
        outputs.x = self.post_norm(outputs.x)

        return outputs
