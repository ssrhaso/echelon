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

class Reacher(dm_control.DeepMindControlEnv):

    """
    
    Reacher (dim(S)=4, dim(A)=2, dim(O)=7): 
    The simple two-link planar reacher with a ran- domised target location. 
    The reward is one when the end effector pen- etrates the target sphere. 
    In the easy task the target sphere is bigger than on the hard task (shown on the left).

    Reacher     easy
    Reacher     hard
    
    """

    def __init__(
            self, 
            img_size=(64, 64), 
            history_frames=1, 
            episode_saving_path=None, 
            task="easy", 
            action_repeat=1
        ):

        assert task in ["easy", "hard"]
        super(Reacher, self).__init__(
            domain="reacher", 
            task=task, 
            img_size=img_size, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat
        )

        self.num_actions = 2