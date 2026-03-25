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

class Cheetah(dm_control.DeepMindControlEnv):

    """
    
    Cheetah (dim(S)=18, dim(A)=6, dim(O)=17): A running planar biped based on (Wawrzyński, 2009). 
    The reward r is linearly proportional to the forward velocity v up to a maximum of 10m/s i.e. r(v) = max(0, min(v/10, 1)).

    
    """

    def __init__(
            self, 
            task="run", 
            img_size=(64, 64), 
            history_frames=4, 
            episode_saving_path=None, 
            action_repeat=1
        ):

        assert task in ["run"]
        super(Cheetah, self).__init__(
            domain="cheetah", 
            task=task, 
            img_size=img_size, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat
        )

        self.num_actions = 6