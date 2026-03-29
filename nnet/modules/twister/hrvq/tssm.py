# Copyright 2025, Hasaan Ahmad.
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

"""Spatial HRVQ TSSM: transformer-based sequence model with spatial dynamics.

The TSSM transformer processes one aggregated token per timestep (unchanged).
Spatial structure exists only in the dynamics head: deter(512) is projected to
16 spatial positions, each predicting 3-level codebook logits via a shared MLP.
Codebook lookup + residual sum + aggregation produces stoch(32, 32).

Changes from fvq/tssm.py:
- dynamics_predictor replaced by spatial_proj + position_embeddings + spatial_dynamics_mlp
- State dict keys: logits_l0, logits_l1, logits_l2 instead of logits
- spatial_aggregate Linear(4096 -> 1024) for codebook outputs -> stoch
"""

import torch
import torch.nn as nn

from nnet import modules
from nnet import distributions
from nnet import structs


class SpatialHRVQTSSM(nn.Module):

    def __init__(
            self,
            num_actions,
            stoch_size=32,
            act_fun=nn.SiLU,
            discrete=32,
            learn_initial=True,
            weight_init="dreamerv3_normal",
            bias_init="zeros",
            norm={"class": "LayerNorm", "params": {"eps": 1e-3}},
            uniform_mix=0.01,
            action_clip=1.0,
            dist_weight_init="xavier_uniform",
            dist_bias_init="zeros",

            # Transformer
            hidden_size=1024,
            num_blocks=4,
            ff_ratio=4,
            num_heads=16,
            drop_rate=0.1,
            att_context_left=64,
            module_pre_norm=False,

            # Spatial HRVQ dynamics
            num_positions=16,
            position_dim=256,
            num_codes: list = None,
            hrvq=None,
            spatial_proj_dim=128,
        ):
        super(SpatialHRVQTSSM, self).__init__()

        # Params
        self.num_actions = num_actions
        self.stoch_size = stoch_size
        self.act_fun = act_fun
        self.discrete = discrete
        self.learn_initial = learn_initial
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.norm = norm
        self.uniform_mix = uniform_mix
        self.action_clip = action_clip
        self.dist_weight_init = dist_weight_init
        self.dist_bias_init = dist_bias_init

        # Transformer
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.ff_ratio = ff_ratio
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.att_context_left = att_context_left
        self.max_pos_encoding = 2048

        # Spatial HRVQ params
        if num_codes is None:
            num_codes = [512, 512, 512]
        self.num_positions = num_positions
        self.position_dim = position_dim
        self.num_levels = len(num_codes)
        self.num_codes = num_codes
        self.spatial_proj_dim = spatial_proj_dim

        # Reference to encoder's HRVQ (not a child module — no extra parameters)
        self.hrvq = hrvq

        # zt + at -> Linear -> Norm -> Act -> Linear -> Norm -> et (unchanged)
        self.action_mixer = modules.MultiLayerPerceptron(
            dim_input=self.stoch_size * self.discrete + self.num_actions if self.discrete else self.stoch_size + self.num_actions,
            dim_layers=[self.hidden_size, self.hidden_size],
            act_fun=[self.act_fun, None],
            weight_init=weight_init,
            bias_init=self.bias_init,
            norm=[self.norm, None] if module_pre_norm else self.norm,
            bias=self.norm is None
        )

        # Transformer et -> dt, ht (unchanged)
        self.transformer = modules.TransformerNetwork(
            dim_model=self.hidden_size,
            num_blocks=self.num_blocks,
            att_params={
                "class": "RelPosMultiHeadSelfAttention",
                "params": {
                    "num_heads": self.num_heads,
                    "weight_init": "default",
                    "bias_init": "default",
                    "attn_drop_rate": self.drop_rate,
                    "max_pos_encoding": self.max_pos_encoding,
                    "causal": True
                }
            },
            emb_drop_rate=0.0,
            drop_rate=self.drop_rate,
            pos_embedding=None,
            mask=None,
            ff_ratio=self.ff_ratio,
            weight_init="default",
            bias_init="default",
            act_fun="ReLU",
            module_pre_norm=module_pre_norm
        )

        # Spatial dynamics: deter(512) -> per-position features (16, spatial_proj_dim)
        self.spatial_proj = modules.Linear(
            in_features=self.hidden_size,
            out_features=num_positions * spatial_proj_dim,
            weight_init=self.dist_weight_init,
            bias_init=self.dist_bias_init
        )

        # Learned position embeddings for the 4x4 grid
        self.position_embeddings = nn.Parameter(torch.randn(num_positions, 32))

        # Shared MLP: per-position features -> per-level logits
        # Input: spatial_proj_dim + 32 = 160, Output: sum(num_codes)
        self.spatial_dynamics_mlp = nn.Sequential(
            nn.Linear(spatial_proj_dim + 32, 512),
            nn.SiLU(),
            nn.Linear(512, sum(num_codes)),
        )

        # Spatial aggregation: (16*256=4096 -> 1024)
        self.spatial_aggregate = modules.Linear(
            in_features=num_positions * position_dim,
            out_features=stoch_size * discrete,
            weight_init=self.dist_weight_init,
            bias_init=self.dist_bias_init
        )

        if self.learn_initial:
            self.weight_init = nn.Parameter(torch.zeros(self.hidden_size))

    def _predict_spatial(self, deter, sample=True):
        """Core spatial dynamics prediction.

        Args:
            deter: (*, hidden_size) deterministic state from transformer
            sample: if True, sample from Categorical; if False, argmax
        Returns:
            stoch: (*, 32, 32) aggregated quantized spatial tokens, reshaped
            all_logits: list of num_levels tensors, each (*, 16, num_codes[level])
            all_indices: list of num_levels LongTensors, each (*, 16)
        """
        batch_shape = deter.shape[:-1]

        # 1. Project deter to per-position features: (*, 512) -> (*, 16, 128)
        pos_features = self.spatial_proj(deter)
        pos_features = pos_features.reshape(batch_shape + (self.num_positions, self.spatial_proj_dim))

        # 2. Add position embeddings: (*, 16, 128) cat (16, 32) -> (*, 16, 160)
        pos_emb = self.position_embeddings.expand(batch_shape + (self.num_positions, 32))
        pos_input = torch.cat([pos_features, pos_emb], dim=-1)

        # 3. Shared MLP: (*, 16, 160) -> (*, 16, sum(num_codes))
        logits_all = self.spatial_dynamics_mlp(pos_input)

        # 4. Split logits per level
        all_logits = list(torch.split(logits_all, self.num_codes, dim=-1))

        # 5. Sample or argmax per level per position
        all_indices = []
        for level_logits in all_logits:
            if sample:
                indices = torch.distributions.Categorical(logits=level_logits).sample()
            else:
                indices = level_logits.argmax(dim=-1)
            all_indices.append(indices)

        # 6. Codebook lookup + residual sum per position
        z_q_positions = torch.zeros(
            batch_shape + (self.num_positions, self.position_dim),
            device=deter.device, dtype=deter.dtype
        )
        for level, indices in enumerate(all_indices):
            idx_flat = indices.reshape(-1)
            z_q_level = self.hrvq.quantizers[level].embedding[idx_flat]
            z_q_level = z_q_level.reshape(batch_shape + (self.num_positions, self.position_dim))
            z_q_positions = z_q_positions + z_q_level

        # 7. Aggregate: (*, 16, 256) -> (*, 4096) -> (*, 1024) -> (*, 32, 32)
        z_q_flat = z_q_positions.reshape(batch_shape + (self.num_positions * self.position_dim,))
        aggregated = self.spatial_aggregate(z_q_flat)
        stoch = aggregated.reshape(batch_shape + (self.stoch_size, self.discrete))

        return stoch, all_logits, all_indices

    def get_stoch(self, deter):
        """Predict stochastic state from deterministic state (argmax).

        Args:
            deter: (*, hidden_size) deterministic state
        Returns:
            stoch: (*, stoch_size, discrete) = (*, 32, 32)
        """
        stoch, _, _ = self._predict_spatial(deter, sample=False)
        return stoch

    def initial(self, batch_size=1, seq_length=1, dtype=torch.float32, device="cpu", detach_learned=False):
        """Create initial state.

        Returns:
            AttrDict with stoch, deter, hidden, logits_l0, logits_l1, logits_l2
        """
        initial_state = structs.AttrDict(
            stoch=torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device),
            deter=torch.zeros(batch_size, seq_length, self.hidden_size, dtype=dtype, device=device),
            hidden=None
        )

        # Per-level spatial logits
        for level in range(self.num_levels):
            initial_state[f"logits_l{level}"] = torch.zeros(
                batch_size, seq_length, self.num_positions, self.num_codes[level],
                dtype=dtype, device=device
            )

        # Learned Initial
        if self.learn_initial:
            initial_state.deter = self.weight_init.repeat(batch_size, seq_length, 1)
            initial_state.stoch = self.get_stoch(initial_state.deter)

            if detach_learned:
                initial_state.deter = initial_state.deter.detach()
                initial_state.stoch = initial_state.stoch.detach()

        return initial_state

    def observe(self, states, prev_actions, is_firsts, prev_state=None, is_firsts_hidden=None, return_blocks_deter=False):
        """Unchanged from TWISTER. Runs forward model on observation sequence."""

        # Create prev_states (B, L-1, ...)
        prev_states = {key: value[:, :-1] for key, value in states.items()}

        # Initial State
        if prev_state is None:
            prev_actions[:, 0] = 0.0
            prev_state = self.initial(batch_size=prev_actions.shape[0], seq_length=1, dtype=prev_actions.dtype, device=prev_actions.device)
            is_firsts_hidden = None

        # Concat prev_state (B, L, ...)
        prev_states = {key: torch.cat([prev_state[key], value], dim=1) for key, value in prev_states.items()}
        prev_states["hidden"] = prev_state["hidden"]

        # Forward Model (B, L, D)
        posts, priors = self(states, prev_states, prev_actions, is_firsts, is_firsts_hidden, return_blocks_deter=return_blocks_deter)

        return posts, priors

    def imagine(self, p_net, prev_state, img_steps=1, is_firsts=None, is_firsts_hidden=None, actions=None):
        """Imagination loop with spatial dynamics."""

        # Policy
        policy = lambda s: p_net(self.get_feat(s).detach()).rsample()

        # Current state action
        if actions is None:
            prev_state["action"] = policy(prev_state)
        else:
            assert actions.shape[1] == img_steps
            prev_state["action"] = actions[:, :1]

        # Model Recurrent loop with St, At
        img_states = {
            "stoch": [prev_state["stoch"]],
            "deter": [prev_state["deter"]],
            "action": [prev_state["action"]],
        }
        for level in range(self.num_levels):
            img_states[f"logits_l{level}"] = [prev_state[f"logits_l{level}"]]

        for h in range(img_steps):

            # Compute mask
            mask = modules.return_mask(
                seq_len=1,
                hidden_len=self.get_hidden_len(prev_state["hidden"]),
                left_context=self.att_context_left,
                right_context=0,
                dtype=prev_state["action"].dtype,
                device=prev_state["action"].device
            )
            if is_firsts_hidden is not None:

                # Append is_first mask
                is_firts_mask = modules.return_is_firsts_mask(is_firsts=is_firsts, is_firsts_hidden=is_firsts_hidden)
                mask = mask.minimum(is_firts_mask)

                # Concat is_firsts to hidden is_firsts (B, C)
                is_firsts_hidden = torch.cat([is_firsts_hidden[:, 1:], is_firsts], dim=1)

                # Set is_firsts to zero (B, 1)
                is_firsts = torch.zeros_like(is_firsts)

            # Forward Model
            img_state = self.forward_img(
                prev_states=prev_state,
                prev_actions=prev_state["action"],
                mask=mask,
            )

            # Current state action
            if actions is None or h == img_steps - 1:
                img_state["action"] = policy(img_state)
            else:
                img_state["action"] = actions[:, h + 1:h + 2]

            # Slice hidden
            img_state["hidden"] = self.slice_hidden(img_state["hidden"])

            # Update previous state
            prev_state = img_state

            # Append to Lists
            for key, value in img_state.items():
                if key != "hidden":
                    img_states[key].append(value)

        # Stack Lists
        img_states = {k: torch.concat(v, dim=1) for k, v in img_states.items()}

        return img_states

    def get_feat(self, state, blocks_deter_id=None):
        """Unchanged from TWISTER. Concatenate stoch (flat 1024) + deter (512) = 1536."""
        return torch.cat([state["stoch"].flatten(start_dim=-2, end_dim=-1), state["deter"] if blocks_deter_id is None else state["blocks_deter"][blocks_deter_id]], dim=-1)

    def slice_hidden(self, hidden):
        """Unchanged from TWISTER."""
        hidden = [(hidden_blk[0][:, -self.att_context_left:], hidden_blk[1][:, -self.att_context_left:]) for hidden_blk in hidden]
        return hidden

    def get_hidden_len(self, hidden):
        """Unchanged from TWISTER."""
        if hidden is not None:
            return hidden[0][0].shape[1]
        else:
            return 0

    def forward_img(self, prev_states, prev_actions, mask, return_att_w=False, return_blocks_deter=False):
        """Prior prediction with spatial dynamics.

        Args:
            prev_states: dict with "stoch" (B, L, 32, 32), "deter" (B, L, 512), "hidden"
            prev_actions: (B, L, A)
            mask: attention mask
        Returns dict:
            "stoch": (B, L, 32, 32)
            "deter": (B, L, 512)
            "hidden": transformer hidden states
            "logits_l0", "logits_l1", "logits_l2": each (B, L, 16, 512)
        """
        # Clip Action -c:+c
        if self.action_clip > 0.0:
            prev_actions = prev_actions * (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()

        # Flatten stoch size and discrete size
        if self.discrete:
            stoch = prev_states["stoch"].flatten(start_dim=-2, end_dim=-1)
        else:
            stoch = prev_states["stoch"]

        # MLP action mixer (unchanged)
        x = self.action_mixer(torch.concat([stoch, prev_actions], dim=-1))

        # Transformer (unchanged)
        assert self.get_hidden_len(prev_states["hidden"]) <= self.att_context_left, \
            "warning: att context left is {} and hidden has length {}".format(self.att_context_left, self.get_hidden_len(prev_states["hidden"]))
        outputs = self.transformer(x, hidden=prev_states["hidden"], mask=mask, return_hidden=True, return_att_w=return_att_w, return_blocks_x=return_blocks_deter)
        deter, hidden = outputs.x, outputs.hidden

        # Additional Outputs
        add_out_dict = {}
        if return_att_w:
            add_out_dict["att_w"] = outputs.att_w
        if return_blocks_deter:
            add_out_dict["blocks_deter"] = outputs.blocks_x

        # Spatial dynamics prediction
        stoch, all_logits, all_indices = self._predict_spatial(deter, sample=True)

        # Build return dict
        result = {"stoch": stoch, "deter": deter, "hidden": hidden}
        for level in range(self.num_levels):
            result[f"logits_l{level}"] = all_logits[level]
        result.update(add_out_dict)

        return result

    def forward_obs(self, deter, hidden, states):
        """Posterior: use encoder's quantized output as the true state."""
        return {"deter": deter, "hidden": hidden, **states}

    def forward(self, states, prev_states, prev_actions, is_firsts, is_firsts_hidden=None, return_att_w=False, return_blocks_deter=False):
        """Full forward pass: compute prior and posterior."""
        # (B, 1 or L, A)
        assert prev_actions.dim() == 3
        # (B, 1 or L)
        assert is_firsts.dim() == 2

        # Clip Action (B, L, D) -c:+c
        if self.action_clip > 0.0:
            prev_actions *= (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()

        # Create right context mask (B, 1, L, Th+L)
        mask = modules.return_mask(seq_len=prev_actions.shape[1], hidden_len=self.get_hidden_len(prev_states["hidden"]), left_context=self.att_context_left, right_context=0, dtype=prev_actions.dtype, device=prev_actions.device)

        # Reset first states and actions at episode boundaries
        if is_firsts.any():

            # Unsqueeze is_firsts (B, L, 1)
            is_firsts = is_firsts.unsqueeze(dim=-1)

            # Reset first Actions
            prev_actions *= (1.0 - is_firsts)

            # Reset first States (B, L, ...)
            init_state = self.initial(batch_size=prev_actions.shape[0], seq_length=prev_actions.shape[1], dtype=prev_actions.dtype, device=prev_actions.device)
            for key, value in prev_states.items():

                # Hidden does not need reset, auto masked
                if key == "hidden":
                    prev_states[key] = value
                # Reset first States
                else:
                    is_firsts_r = torch.reshape(is_firsts, is_firsts.shape + (1,) * (len(value.shape) - len(is_firsts.shape)))
                    prev_states[key] = value * (1.0 - is_firsts_r) + init_state[key] * is_firsts_r

            # Mask positions of past trajectories
            is_firts_mask = modules.return_is_firsts_mask(is_firsts.squeeze(dim=-1), is_firsts_hidden=is_firsts_hidden)
            mask = mask.minimum(is_firts_mask)

        # Forward Img (prior)
        prior = self.forward_img(prev_states, prev_actions, mask, return_att_w=return_att_w, return_blocks_deter=return_blocks_deter)

        # Forward Obs (posterior — uses encoder's VQ stoch directly)
        post = self.forward_obs(prior["deter"], prior["hidden"], states)
        if return_att_w:
            post["att_w"] = prior["att_w"]
        if return_blocks_deter:
            post["blocks_deter"] = prior["blocks_deter"]

        return post, prior
