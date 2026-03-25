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

# Neural Nets
from nnet.modules import layers
from nnet import structs

class MultiHeadAttention(nn.Module):

    """Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        attn_drop_rate: attn map dropout rate
        weight_init: linear layers weight_init
        bias_init: linear layers bias_init
        output_proj: output projection after attention
        dim_kv: input dimension of key/value for projection

    References: 
        Attention Is All You Need, Vaswani et al.
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, dim_model, num_heads, attn_drop_rate, weight_init="scaled_uniform", bias_init="zeros", output_proj=True, dim_kv=None, channels_last=True):
        super(MultiHeadAttention, self).__init__()

        # Dim Key Value
        if dim_kv == None:
            dim_kv = dim_model

        # Attention Params
        self.num_heads = num_heads # H
        self.dim_model = dim_model # D
        self.dim_head = dim_model // num_heads # d
        self.output_proj = output_proj
        self.dim_kv = dim_kv
        self.channels_last = channels_last

        # Attention Dropout
        self.dropout = layers.Dropout(attn_drop_rate) if attn_drop_rate > 0 else nn.Identity()

        # Init Layers
        self.init_layers(weight_init, bias_init)

    def init_layers(self, weight_init, bias_init):

        # Linear Layers
        self.query_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.key_layer = layers.Linear(self.dim_kv, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.value_layer = layers.Linear(self.dim_kv, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.output_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init) if self.output_proj else nn.Identity()

    def forward_inputs(self, Q, K, V):

        # (B, D, T) -> (B, T, D)
        if not self.channels_last:
            Q = Q.transpose(1, -1)
            K = K.transpose(1, -1)
            V = V.transpose(1, -1)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        return Q, K, V

    def forward_outputs(self, O):

        # Linear Layers
        O = self.output_layer(O)

        # (B, T, D) -> (B, D, T)
        if not self.channels_last:
            O = O.transpose(1, -1)

        return O

    def forward(self, x, mask=None, return_att_w=False):
        return self.forwardQKV(x, x, x, mask, return_att_w)

    def forwardQKV(self, Q, K, V, mask=None, hidden=None, return_hidden=False, return_att_w=False, detach_hidden=True):

        """Scaled Dot-Product Multi-Head Attention

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
            hidden: Optional Key and Value hidden states for decoding
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, Th + T)
            hidden: Key and value hidden states

        """

        # Input Linear Layers
        Q, K, V = self.forward_inputs(Q, K, V)

        # Batch size B
        batch_size = Q.size(0)

        # Hidden State Provided
        if hidden is not None:
            K = torch.cat([hidden[0], K], dim=1)
            V = torch.cat([hidden[1], V], dim=1)

        # Update Hidden State
        if return_hidden:
            hidden = (K.detach(), V.detach()) if detach_hidden else (K, V)

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Att scores (B, H, T, T)
        att_scores = Q.matmul(K.transpose(2, 3)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, H, T, T)
        att_w = att_scores.softmax(dim=-1)

        # Att Dropout
        att_w = self.dropout(att_w)

        # Att output (B, H, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.forward_outputs(O)

        # Format Outputs
        outputs = structs.AttrDict(x=O)
        if return_hidden:
            outputs.hidden = hidden
        if return_att_w:
            outputs.att_w = att_w

        return outputs