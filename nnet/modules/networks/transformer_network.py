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
from torch import nn

# NeuralNets
from nnet import modules
from nnet import structs

class TransformerNetwork(nn.Module):

    """ TransformerNetwork: Positional Embedding + Input Dropout + Transformer Blocks + Final Layernorm """

    def __init__(
        self, 
        dim_model, 
        num_blocks, 
        att_params={"class": "MultiHeadAttention", "params":{"num_heads": 4, "attn_drop_rate": 0.1, "weight_init": "normal_02", "bias_init": "zeros"}}, 
        ff_ratio=4, 
        emb_drop_rate=0.1, 
        drop_rate=0.1, 
        act_fun="GELU", 
        pos_embedding=None, 
        mask=None, 
        inner_dropout=False, 
        weight_init="normal_02", 
        bias_init="zeros", 
        module_pre_norm=True,
        block_post_norm=False
    ):
        super(TransformerNetwork, self).__init__()

        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=emb_drop_rate)

        # Mask
        self.mask = mask

        # Transformer Blocks
        self.blocks = nn.ModuleList([modules.TransformerBlock(
            dim_model=dim_model,
            ff_ratio=ff_ratio,
            att_params=att_params,
            drop_rate=drop_rate,
            inner_dropout=inner_dropout,
            act_fun=act_fun,
            weight_init=weight_init,
            bias_init=bias_init,
            module_pre_norm=module_pre_norm,
            post_norm=block_post_norm
        ) for block_id in range(num_blocks)])

        # LayerNorm
        self.layernorm = nn.LayerNorm(normalized_shape=dim_model) if (not block_post_norm) and module_pre_norm else nn.Identity()

    def forward(self, x, lengths=None, hidden=None, return_hidden=False, return_att_w=False, mask=None, start_position=0, return_blocks_x=False):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x, start_position=start_position)

        # Input Dropout
        x = self.dropout(x)

        # Override Mask param with default mask
        if mask is None:
            mask = self.mask

        # Mask (1 or B, 1, N, N)
        if isinstance(mask, nn.Module):
            mask = mask(x, lengths)

        # Transformer Blocks
        outputs = structs.AttrDict(x=x)
        if return_hidden:
            outputs.hidden = []
        if return_att_w:
            outputs.att_w = []
        if return_blocks_x:
            outputs.blocks_x = []
        for block_id, block in enumerate(self.blocks):

            # Forward Block
            block_outputs = block(outputs.x, mask=mask, hidden=hidden[block_id] if hidden is not None else None, return_hidden=return_hidden, return_att_w=return_att_w)

            # Parse block output
            outputs.x = block_outputs.x
            if return_hidden:
                outputs.hidden.append(block_outputs.hidden)
            if return_att_w:
                outputs.att_w.append(block_outputs.att_w)
            if return_blocks_x:
                outputs.blocks_x.append(block_outputs.x)

        # LayerNorm
        outputs.x = self.layernorm(outputs.x)

        return outputs