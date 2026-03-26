# Copyright 2025, Maxime Burchi.
# Copyright 2025, Hasaan Ahmad.  # ECHELON modifications
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
from nnet import structs


class TSSM(nn.Module):

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

            # ECHELON: new params for hierarchical dynamics
            num_codes: list = None,
            hrvq: nn.Module = None,
            cond_proj_dim: int = 128,
        ):
        super(TSSM, self).__init__()

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

        # ECHELON: HRVQ config
        if num_codes is None:
            num_codes = [512, 512, 512]
        self.num_codes = num_codes
        self.cond_proj_dim = cond_proj_dim

        # ECHELON: stored reference to encoder's HRVQ — NOT a child module (already owned by encoder)
        # Used for codebook lookups during imagination (indices -> embeddings)
        self._hrvq_ref = hrvq  # type: HRVQ | None

        # zt + at -> Linear -> Norm -> Act -> Linear -> Norm -> et
        # UNCHANGED: input dim = stoch_size*discrete + num_actions = 1024 + A
        self.action_mixer = modules.MultiLayerPerceptron(
            dim_input=self.stoch_size * self.discrete + self.num_actions if self.discrete else self.stoch_size + self.num_actions,
            dim_layers=[self.hidden_size, self.hidden_size],
            act_fun=[self.act_fun, None],
            weight_init=weight_init,
            bias_init=self.bias_init,
            norm=[self.norm, None] if module_pre_norm else self.norm,
            bias=self.norm is None
        )

        # Transformer et -> dt, ht
        # UNCHANGED
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

        # ECHELON: killed — self.dynamics_predictor (single Linear(hidden_size -> discrete*stoch_size))
        # Replaced by per-level dynamics heads with hierarchical conditioning

        # ECHELON: L0 head — predicts codebook indices for level 0 from transformer deter
        self.dynamics_head_l0 = None  # ECHELON: nn.Linear(hidden_size, num_codes[0]), (512 -> 512)

        # ECHELON: L0 conditioning projection — projects z_q0 (1024-dim) for conditioning L1
        self.l0_cond_proj = None  # ECHELON: nn.Linear(stoch_size * discrete, cond_proj_dim), (1024 -> 128)

        # ECHELON: L1 head — conditioned on deter + projected z_q0
        self.dynamics_head_l1 = None  # ECHELON: nn.Linear(hidden_size + cond_proj_dim, num_codes[1]), (640 -> 512)

        # ECHELON: L1 conditioning projection — projects z_q1 (1024-dim) for conditioning L2
        self.l1_cond_proj = None  # ECHELON: nn.Linear(stoch_size * discrete, cond_proj_dim), (1024 -> 128)

        # ECHELON: L2 head — conditioned on deter + projected z_q0 + projected z_q1
        self.dynamics_head_l2 = None  # ECHELON: nn.Linear(hidden_size + 2 * cond_proj_dim, num_codes[2]), (768 -> 512)

        if self.learn_initial:
            self.weight_init = nn.Parameter(torch.zeros(self.hidden_size))

    def _predict_hierarchical(self, deter, sample=True):
        """Core hierarchical prediction: L0 from deter, L1|L0, L2|L0+L1.

        Cascade: predict L0 indices from deter, look up z_q0 from HRVQ codebook,
        project and concatenate to predict L1, then L2. Sum z_q levels and reshape.

        Args:
            deter: (*, 512) transformer deterministic hidden state
            sample: if True, sample from categorical; if False, take argmax (mode)
        Returns:
            stoch: (*, 32, 32) — reshaped sum of quantized vectors (1024 -> 32x32)
            logits_l0: (*, num_codes[0]) — raw logits for level 0
            logits_l1: (*, num_codes[1]) — raw logits for level 1
            logits_l2: (*, num_codes[2]) — raw logits for level 2
        """
        raise NotImplementedError("ECHELON: hierarchical cascade prediction L0->L1->L2")

    def get_stoch(self, deter):
        """Predict hierarchical codebook indices from deter, look up, sum, reshape.

        Replaces the original dynamics_predictor + OneHotDist.mode() path.

        Args:
            deter: (*, hidden_size=512) transformer deterministic state
        Returns:
            stoch: (*, 32, 32) — summed and reshaped quantized vector (mode, not sampled)
        """
        raise NotImplementedError("ECHELON: get_stoch via _predict_hierarchical(deter, sample=False)")

    def initial(self, batch_size=1, seq_length=1, dtype=torch.float32, device="cpu", detach_learned=False):
        """Create initial state dict for TSSM.

        ECHELON changes:
        - stoch: still zeros (B, seq_len, 32, 32)
        - REMOVED: "logits" key
        - ADDED: "logits_l0" zeros (B, seq_len, num_codes[0])
        - ADDED: "logits_l1" zeros (B, seq_len, num_codes[1])
        - ADDED: "logits_l2" zeros (B, seq_len, num_codes[2])
        - learn_initial: deter -> get_stoch(deter) still works (get_stoch now uses hierarchical prediction)

        Args:
            batch_size: int
            seq_length: int
            dtype: torch.dtype
            device: str or torch.device
            detach_learned: bool — if True, detach learned initial state
        Returns:
            AttrDict with keys: stoch, logits_l0, logits_l1, logits_l2, deter, hidden
        """
        raise NotImplementedError("ECHELON: initial state with per-level logits instead of flat logits")

    def observe(self, states, prev_actions, is_firsts, prev_state=None, is_firsts_hidden=None, return_blocks_deter=False):
        """UNCHANGED from TWISTER. Operates on filtered {"stoch"} dict from WorldModel.

        The encoder output is split by WorldModel.forward BEFORE calling this method:
            tssm_states = {"stoch": encoder_out["stoch"]}  # only what TSSM expects
        So observe() never sees hrvq_info, indices, or vq_loss.

        Args:
            states: dict with "stoch" (B, L, 32, 32)
            prev_actions: (B, L, A) one-hot or continuous actions
            is_firsts: (B, L) episode boundary flags
            prev_state: optional dict from previous call
            is_firsts_hidden: optional (B, C) hidden is_firsts mask
            return_blocks_deter: bool
        Returns:
            posts: dict — posterior states (with encoder stoch + transformer deter)
            priors: dict — prior states (with predicted stoch from dynamics heads)
        """

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
        """Run imagination rollout using policy network.

        ECHELON changes:
        - img_states keys: replace "logits" with "logits_l0", "logits_l1", "logits_l2"
        - forward_img now returns per-level logits

        Args:
            p_net: policy network — called as p_net(feat) -> distribution
            prev_state: dict with stoch, deter, logits_l0, logits_l1, logits_l2, hidden, action
            img_steps: int — number of imagination steps
            is_firsts: (B, 1) or None
            is_firsts_hidden: (B, C) or None
            actions: (B, img_steps, A) or None — if provided, override policy
        Returns:
            img_states: dict with keys stoch, deter, logits_l0, logits_l1, logits_l2, action
                        each (B, 1+img_steps, ...)
        """
        raise NotImplementedError("ECHELON: imagine with per-level logits in state dict")

    def get_feat(self, state, blocks_deter_id=None):
        """UNCHANGED from TWISTER. Concatenates flattened stoch + deter.

        stoch.flatten(-2, -1) still produces 1024-dim vector regardless of whether
        stoch is TWISTER's one-hot categorical or ECHELON's reshaped VQ embedding.

        Args:
            state: dict with "stoch" (*, 32, 32) and "deter" (*, 512)
            blocks_deter_id: optional int for block-specific deter
        Returns:
            feat: (*, 1536) concatenated feature vector
        """
        return torch.cat([state["stoch"].flatten(start_dim=-2, end_dim=-1), state["deter"] if blocks_deter_id is None else state["blocks_deter"][blocks_deter_id]], dim=-1)

    def get_dist(self, state):
        # ECHELON: killed — replaced by _predict_hierarchical with per-level categoricals
        # Prior is categorical over codebook indices, not OneHotDist over (stoch_size, discrete)
        raise NotImplementedError("ECHELON: removed — no OneHotDist in VQ world model")

    def slice_hidden(self, hidden):
        """UNCHANGED from TWISTER."""
        hidden = [(hidden_blk[0][:, -self.att_context_left:], hidden_blk[1][:, -self.att_context_left:]) for hidden_blk in hidden]
        return hidden

    def get_hidden_len(self, hidden):
        """UNCHANGED from TWISTER."""
        if hidden != None:
            return hidden[0][0].shape[1]
        else:
            return 0

    def forward_img(self, prev_states, prev_actions, mask, return_att_w=False, return_blocks_deter=False):
        """Single-step forward in imagination (or training).

        ECHELON changes:
        - Action mixer input: UNCHANGED (stoch.flatten(-2,-1) still = 1024)
        - Transformer forward: UNCHANGED
        - REPLACED: dynamics_predictor call with _predict_hierarchical(deter)
        - Return dict ADDS: logits_l0, logits_l1, logits_l2
        - Return dict REMOVES: "logits" key (replaced by per-level logits)

        Args:
            prev_states: dict with "stoch" (B, 1, 32, 32), "hidden" list, etc.
            prev_actions: (B, 1, A) actions
            mask: (B, 1, 1, C+1) attention mask
            return_att_w: bool
            return_blocks_deter: bool
        Returns:
            dict with keys:
                "stoch": (B, 1, 32, 32) — predicted next stoch
                "deter": (B, 1, hidden_size) — transformer output
                "hidden": updated hidden state list
                "logits_l0": (B, 1, num_codes[0])
                "logits_l1": (B, 1, num_codes[1])
                "logits_l2": (B, 1, num_codes[2])
                optionally: "att_w", "blocks_deter"
        """
        raise NotImplementedError("ECHELON: forward_img with hierarchical dynamics heads")

    def forward_obs(self, deter, hidden, states):
        """UNCHANGED from TWISTER. Merges encoder states with transformer output.

        States no longer has "logits" key — just "stoch".
        Returns: {"deter": deter, "hidden": hidden, **states}
        Post dict = {"stoch": ..., "deter": ..., "hidden": ...}. No logits.

        Args:
            deter: (B, L, hidden_size)
            hidden: list of (K, V) tuples
            states: dict with "stoch" (B, L, 32, 32)
        Returns:
            dict: {"deter": ..., "hidden": ..., "stoch": ...}
        """
        # Return Post
        return {"deter": deter, "hidden": hidden, **states}

    def forward(self, states, prev_states, prev_actions, is_firsts, is_firsts_hidden=None, return_att_w=False, return_blocks_deter=False):
        """Full training forward: compute prior and posterior for a sequence.

        ECHELON changes:
        - is_firsts reset logic iterates over prev_states keys. Since observe() now
          passes only {"stoch"}, the reset loop handles "stoch" and "hidden" only.
          Per-level logits (logits_l0, l1, l2) are in prev_states from initial() and
          forward_img(), so the reset loop handles them too.
        - forward_img returns per-level logits instead of flat "logits"
        - forward_obs unchanged (passes through "stoch" from encoder)

        Args:
            states: dict with "stoch" (B, L, 32, 32) — from encoder
            prev_states: dict with stoch, logits_l0, logits_l1, logits_l2, deter, hidden
            prev_actions: (B, L, A)
            is_firsts: (B, L)
            is_firsts_hidden: (B, C) or None
            return_att_w: bool
            return_blocks_deter: bool
        Returns:
            post: dict — posterior states
            prior: dict — prior states with per-level logits
        """
        raise NotImplementedError("ECHELON: forward with is_firsts reset on per-level logits")
