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
import torch.nn.functional as F

# Neural Nets
from nnet.modules import layers
from nnet.modules import embeddings
from nnet.modules.attentions import MultiHeadAttention
from nnet import structs

class RelPosMultiHeadSelfAttention(MultiHeadAttention):

    """Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements

    References: 
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Dai et al.
        https://arxiv.org/abs/1901.02860

    """

    def __init__(
        self, 
        dim_model, 
        num_heads, 
        attn_drop_rate, 
        max_pos_encoding, 
        causal, 
        weight_init="scaled_uniform", 
        bias_init="zeros", 
        output_proj=True,
        apply_bias=True,
        channels_last=True
    ):
        super(RelPosMultiHeadSelfAttention, self).__init__(
            dim_model=dim_model, 
            num_heads=num_heads, 
            attn_drop_rate=attn_drop_rate, 
            weight_init=weight_init, 
            bias_init=bias_init, 
            output_proj=output_proj,
            channels_last=channels_last
        )

        # Position Embedding Layer
        self.pos_layer = layers.Linear(self.dim_model, self.dim_model)
        self.causal = causal

        # Global content and positional bias
        self.apply_bias = apply_bias
        if self.apply_bias:
            self.u = nn.Parameter(torch.Tensor(self.dim_model)) # Content bias
            nn.init.zeros_(self.u)
            self.v = nn.Parameter(torch.Tensor(self.dim_model)) # Pos bias
            nn.init.zeros_(self.v)

        # Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = embeddings.RelSinPosEncoding(max_pos_encoding, self.dim_model, self.causal)

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, H, T, Th + 2*T-1) for full context and (B, H, T, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T, Th + T)

        References: 
            causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281
            
            full context:
            Attention Augmented Convolutional Networks, Bello et al.
            https://arxiv.org/abs/1904.09925

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Flatten (B, H, T + TTh + TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = F.pad(att_scores, pad=(seq_length2 - seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1), value=0)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1-1:]

        return att_scores
    
    def forward(self, x, mask=None, hidden=None, return_hidden=False, return_att_w=False):
        return self.forwardQKV(Q=x, K=x, V=x, mask=mask, hidden=hidden, return_hidden=return_hidden, return_att_w=return_att_w)

    def forwardQKV(self, Q, K, V, mask=None, hidden=None, return_hidden=False, return_att_w=False, detach_hidden=True):

        """Scaled Dot-Product Self-Attention with relative sinusoidal position encodings

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

        # Add Bias
        if self.apply_bias:
            Qu = Q + self.u
            Qv = Q + self.v
        else:
            Qu = Q
            Qv = Q

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1), K.size(1) - Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T, Th + T)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            # (B, H, L, Context)
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, H, T, Th + T)
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