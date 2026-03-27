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

        # zt + at -> Linear -> Norm -> Act -> Linear -> Norm -> et
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

        # Dynamics Predictor dt -> zt
        self.dynamics_predictor = modules.Linear(
            in_features=self.hidden_size, 
            out_features=self.discrete * self.stoch_size if self.discrete else 2 * self.stoch_size,
            weight_init=self.dist_weight_init,
            bias_init=self.dist_bias_init
        )

        if self.learn_initial:
            self.weight_init = nn.Parameter(torch.zeros(self.hidden_size))

    def get_stoch(self, deter):
        
        # Linear Logits
        logits = self.dynamics_predictor(deter).reshape(deter.shape[:-1] + (self.stoch_size, self.discrete))
        dist_params = {'logits': logits}
    
        # Get Mode
        stoch = self.get_dist(dist_params).mode()

        return stoch

    def initial(self, batch_size=1, seq_length=1, dtype=torch.float32, device="cpu", detach_learned=False):

        initial_state = structs.AttrDict(
            logits=torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device),
            stoch=torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device),
            deter=torch.zeros(batch_size, seq_length, self.hidden_size, dtype=dtype, device=device),
            hidden=None
        )

        # Learned Initial
        if self.learn_initial:
            initial_state.deter = self.weight_init.repeat(batch_size, seq_length, 1)
            initial_state.stoch = self.get_stoch(initial_state.deter) 

            # Detach Learned
            if detach_learned:
                initial_state.deter = initial_state.deter.detach()
                initial_state.stoch = initial_state.stoch.detach()

        return initial_state

    def observe(self, states, prev_actions, is_firsts, prev_state=None, is_firsts_hidden=None, return_blocks_deter=False):


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

        # Policy
        policy = lambda s: p_net(self.get_feat(s).detach()).rsample()
        
        # Current state action
        if actions is None:
            prev_state["action"] = policy(prev_state)
        else:
            assert actions.shape[1] == img_steps
            prev_state["action"] = actions[:, :1]

        # Model Recurrent loop with St, At
        img_states = {"stoch": [prev_state["stoch"]], "deter": [prev_state["deter"]], "logits": [prev_state["logits"]], "action": [prev_state["action"]]}
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
            if actions is None or h==img_steps-1:
                img_state["action"] = policy(img_state)
            else:
                img_state["action"] = actions[:, h+1:h+2]

            # Slice hidden
            img_state["hidden"] = self.slice_hidden(img_state["hidden"])

            # Update previous state
            prev_state = img_state

            # Append to Lists
            for key, value in img_state.items():
                if key != "hidden":
                    img_states[key].append(value)

        # Stack Lists
        img_states = {k: torch.concat(v, dim=1) for k, v in img_states.items()} # (B, 1+img_steps, D)

        return img_states

    def get_feat(self, state, blocks_deter_id=None):

        return torch.cat([state["stoch"].flatten(start_dim=-2, end_dim=-1), state["deter"] if blocks_deter_id is None else state["blocks_deter"][blocks_deter_id]], dim=-1)
    
    def get_dist(self, state):

        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def slice_hidden(self, hidden):

        hidden = [(hidden_blk[0][:, -self.att_context_left:], hidden_blk[1][:, -self.att_context_left:]) for hidden_blk in hidden]

        return hidden
    
    def get_hidden_len(self, hidden):

        if hidden != None:
            return hidden[0][0].shape[1]
        else:
            return 0

    def forward_img(self, prev_states, prev_actions, mask, return_att_w=False, return_blocks_deter=False):

        # Clip Action -c:+c
        if self.action_clip > 0.0:
            prev_actions = prev_actions * (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()

        # Flatten stoch size and discrete size
        if self.discrete:
            stoch = prev_states["stoch"].flatten(start_dim=-2, end_dim=-1)
        else:
            stoch = prev_states["stoch"]

        # MLP Img 1
        x = self.action_mixer(torch.concat([stoch, prev_actions], dim=-1))

        # Recurrent
        assert self.get_hidden_len(prev_states["hidden"]) <= self.att_context_left, "warning: att context left is {} and hidden has length {}".format(self.att_context_left, self.get_hidden_len(prev_states["hidden"]))
        outputs = self.transformer(x, hidden=prev_states["hidden"], mask=mask, return_hidden=True, return_att_w=return_att_w, return_blocks_x=return_blocks_deter)
        deter, hidden = outputs.x, outputs.hidden

        # Additional Outputs
        add_out_dict = {}
        if return_att_w:
            add_out_dict["att_w"] = outputs.att_w
        if return_blocks_deter:
            add_out_dict["blocks_deter"] = outputs.blocks_x

        # Linear Logits
        logits = self.dynamics_predictor(deter).reshape(deter.shape[:-1] + (self.stoch_size, self.discrete))
        dist_params = {'logits': logits}
    
        # Sample
        stoch = self.get_dist(dist_params).rsample()

        # Return Prior
        return {"stoch": stoch, "deter": deter, "hidden": hidden, **dist_params, **add_out_dict}
    
    def forward_obs(self, deter, hidden, states):

        # Return Post
        return {"deter": deter, "hidden": hidden, **states}

    def forward(self, states, prev_states, prev_actions, is_firsts, is_firsts_hidden=None, return_att_w=False, return_blocks_deter=False):

        # (B, 1 or L, A)
        assert prev_actions.dim() == 3 
        # (B, 1 or L)
        assert is_firsts.dim() == 2

        # Clip Action (B, L, D) -c:+c
        if self.action_clip > 0.0:
            prev_actions *= (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()

        # Create right context mask (B, 1, L, Th+L)
        mask = modules.return_mask(seq_len=prev_actions.shape[1], hidden_len=self.get_hidden_len(prev_states["hidden"]), left_context=self.att_context_left, right_context=0, dtype=prev_actions.dtype, device=prev_actions.device)

        # 1: Reset First States and Actions, necessary for traj buffer since some states will be reset mid-sequence
        # 2: Also Update mask to mask pre is_first positions
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

            # Mask positions of past trajectories # (B, 1, L, Th+L)
            is_firts_mask = modules.return_is_firsts_mask(is_firsts.squeeze(dim=-1), is_firsts_hidden=is_firsts_hidden)
            # print(mask.shape, is_firts_mask.shape, is_firsts.shape)
            mask = mask.minimum(is_firts_mask)

        # Forward Img
        prior = self.forward_img(prev_states, prev_actions, mask, return_att_w=return_att_w, return_blocks_deter=return_blocks_deter)

        # Forward Obs
        post = self.forward_obs(prior["deter"], prior["hidden"], states)
        if return_att_w:
            post["att_w"] = prior["att_w"]
        if return_blocks_deter:
            post["blocks_deter"] = prior["blocks_deter"]

        # Return post and prior
        return post, prior