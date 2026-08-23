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

# Neural Nets
from nnet import structs

class BatchEnv:

    def __init__(self, envs):

        self.envs = envs
        self.num_actions = envs[0].num_actions
        self.action_repeat = envs[0].action_repeat
        self.num_envs = len(self.envs)
        self.num_states = len(self.envs[0].obs_space())
        if hasattr(self.envs[0], "clip_low"):
            self.clip_low = self.envs[0].clip_low
        if hasattr(self.envs[0], "clip_high"):
            self.clip_high = self.envs[0].clip_high
        if hasattr(self.envs[0], "fps"):
            self.fps = self.envs[0].fps

    def sample(self):

        actions = []
        for i, env in enumerate(self.envs):
            actions.append(env.sample())

        actions = torch.stack(actions, dim=0)

        return actions
    
    def step(self, actions):

        # Env step loop
        batch_obs = structs.AttrDict()
        for i, env in enumerate(self.envs):

            # Env Step
            obs = env.step(actions[i])

            for key, value in obs.items():

                # Init
                if key not in batch_obs:
                    batch_obs[key] = []

                # Append
                batch_obs[key].append(value)

        # Stack Batch
        for key, value in batch_obs.items():
            batch_obs[key] = torch.stack(batch_obs[key], dim=0)

        return batch_obs
    
    def reset(self, env_i=None):

        # Reset Single env
        if env_i is not None:
            return self.envs[env_i].reset()

        # Env step loop
        batch_obs = structs.AttrDict()
        for i, env in enumerate(self.envs):

            # Reset
            obs = env.reset()

            for key, value in obs.items():

                # Init
                if key not in batch_obs:
                    batch_obs[key] = []

                # Append
                batch_obs[key].append(value)

        # Stack Batch
        for key, value in batch_obs.items():
            batch_obs[key] = torch.stack(batch_obs[key], dim=0)

        return batch_obs

