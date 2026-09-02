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

"""Recursive dynamics core: a TRM-style weight-shared alternative to the block stack.

One causal-attention context block gathers temporal history; a single shared
reasoning MLP then refines two per-timestep states — z_H (the prediction) and
z_L (the scratchpad) — over nested cycles with the context injected each step,
following the Tiny Recursive Model recursion (arXiv 2510.04871):

    for h in H_cycles:
        for l in L_cycles:
            z_L = reason(z_L, z_H + context)
        z_H = reason(z_H, z_L)

Every cycle reuses the same reasoning weights, so depth is bought with compute
rather than parameters. The first H_cycles-1 outer cycles run without gradient
(TRM's one-step gradient approximation); only the final cycle backpropagates.

Drop-in for modules.TransformerNetwork as the TSSM uses it:
``forward(x, hidden=..., mask=..., return_hidden=True) -> AttrDict(x, hidden)``.
The hidden state is the context block's attention cache, so the streaming
imagination path (slice_hidden / get_hidden_len) works unchanged.
"""

import torch
import torch.nn as nn

from nnet import modules
from nnet import structs


class ReasoningCore(nn.Module):
    """Shared per-timestep reasoning block: additive injection + residual MLP sublayers."""

    def __init__(self, dim_model, num_layers=2, ff_ratio=2, act_fun=nn.SiLU):
        super(ReasoningCore, self).__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim_model) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(dim_model, dim_model * ff_ratio),
            act_fun(),
            nn.Linear(dim_model * ff_ratio, dim_model),
        ) for _ in range(num_layers)])

    def forward(self, z, injection):
        z = z + injection
        for norm, mlp in zip(self.norms, self.mlps):
            z = z + mlp(norm(z))
        return z


class RecursiveDynamics(nn.Module):
    """Context attention + two-state recursive reasoning, TransformerNetwork-compatible."""

    def __init__(
            self,
            dim_model,
            num_heads,
            drop_rate,
            max_pos_encoding,
            ff_ratio,
            module_pre_norm,
            context_blocks=1,
            H_cycles=3,
            L_cycles=4,
            reason_layers=2,
            reason_ff_ratio=2,
        ):
        super(RecursiveDynamics, self).__init__()

        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Temporal context: the baseline core's attention block, just fewer of them
        self.context = modules.TransformerNetwork(
            dim_model=dim_model,
            num_blocks=context_blocks,
            att_params={
                "class": "RelPosMultiHeadSelfAttention",
                "params": {
                    "num_heads": num_heads,
                    "weight_init": "default",
                    "bias_init": "default",
                    "attn_drop_rate": drop_rate,
                    "max_pos_encoding": max_pos_encoding,
                    "causal": True
                }
            },
            emb_drop_rate=0.0,
            drop_rate=drop_rate,
            pos_embedding=None,
            mask=None,
            ff_ratio=ff_ratio,
            weight_init="default",
            bias_init="default",
            act_fun="ReLU",
            module_pre_norm=module_pre_norm
        )

        # Single reasoning module shared by the z_L and z_H updates
        self.reasoning = ReasoningCore(dim_model, num_layers=reason_layers, ff_ratio=reason_ff_ratio)

        # Fixed initial states, not trainable (following the paper)
        self.register_buffer("H_init", torch.randn(dim_model))
        self.register_buffer("L_init", torch.randn(dim_model))

        self.out_norm = nn.LayerNorm(dim_model)

    def forward(self, x, lengths=None, hidden=None, return_hidden=False, return_att_w=False, mask=None, start_position=0, return_blocks_x=False):

        # Temporal context over the (cached) history
        ctx = self.context(x, lengths=lengths, hidden=hidden, return_hidden=return_hidden, return_att_w=return_att_w, mask=mask, start_position=start_position)
        c = ctx.x

        # Two recursion states, broadcast from the fixed buffers
        z_H = self.H_init.to(c.dtype).expand_as(c)
        z_L = self.L_init.to(c.dtype).expand_as(c)

        # All but the last outer cycle run without gradient
        if self.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.H_cycles - 1):
                    for _ in range(self.L_cycles):
                        z_L = self.reasoning(z_L, z_H + c)
                    z_H = self.reasoning(z_H, z_L)

        # Final cycle with gradient; context grads flow through the injection
        for _ in range(self.L_cycles):
            z_L = self.reasoning(z_L, z_H + c)
        z_H = self.reasoning(z_H, z_L)

        outputs = structs.AttrDict(x=self.out_norm(z_H))
        if return_hidden:
            outputs.hidden = ctx.hidden
        if return_att_w:
            outputs.att_w = ctx.att_w
        if return_blocks_x:
            outputs.blocks_x = [outputs.x]

        return outputs
