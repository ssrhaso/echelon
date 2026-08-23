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

# DeepMind Control Envs
from .deep_mind_control_env import DeepMindControlEnv
from .acrobot import Acrobot
from .ball_in_cup import BallInCup
from .cartpole import Cartpole
from .cheetah import Cheetah
from .finger import Finger
from .hopper import Hopper
from .pendulum import Pendulum
from .quadruped import Quadruped
from .reacher import Reacher
from .walker import Walker

# DeepMind Control Envs Dictionary
dm_control_dict = {
    "Cheetah": Cheetah,
    "Walker": Walker,
    "Hopper": Hopper,
    "Pendulum": Pendulum,
    "Cartpole": Cartpole,
    "Reacher": Reacher,
    "Quadruped": Quadruped,
    "Acrobot": Acrobot,
    "Finger": Finger,
    "BallInCup": BallInCup,
}