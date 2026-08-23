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

# NeuralNets
from nnet.envs import dm_control

class Hopper(dm_control.DeepMindControlEnv):

    """
    
    Hopper (dim(S)=14, dim(A)=4, dim(O)=15): The planar one-legged hopper introduced in (Lil- licrap et al., 2015), initialised in a random configuration. 
    In the stand task it is rewarded for bringing its torso to a minimal height. 
    In the hop task it is rewarded for torso height and forward velocity.
    
    """

    def __init__(
            self, 
            img_size=(64, 64), 
            history_frames=1, 
            episode_saving_path=None, 
            task="hop", 
            action_repeat=1
        ):

        assert task in ["hop", "stand"]
        super(Hopper, self).__init__(
            domain="hopper", 
            task=task, 
            img_size=img_size, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat
        )

        self.num_actions = 4