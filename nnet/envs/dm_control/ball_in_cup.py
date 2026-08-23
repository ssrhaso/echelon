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

class BallInCup(dm_control.DeepMindControlEnv):

    """
    
    Ball in cup (dim(S)=8, dim(A)=2, dim(O)=8): 
    A planar ball-in-cup task. An actuated planar receptacle can translate 
    in the vertical plane in order to swing and catch a ball attached to its bottom. 
    The catch task has a sparse reward: 1 when the ball is in the cup, 0 otherwise.

    Tasks:
    ball_in_cup catch

    Reference:
    DeepMind Control Suite, Tassa et al.
    https://arxiv.org/abs/1801.00690
    https://www.youtube.com/watch?v=rAai4QzcYbs
    
    """

    def __init__(
            self, 
            task="catch", 
            img_size=(64, 64),
            history_frames=1, 
            episode_saving_path=None, 
            action_repeat=1
        ):

        assert task in ["catch"]
        super(BallInCup, self).__init__(
            domain="ball_in_cup", 
            task=task, 
            img_size=img_size, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat
        )

        self.num_actions = 2