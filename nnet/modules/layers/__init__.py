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

from .permute_channels import PermuteChannels
from .linear import Linear
from .dropout import Dropout
from .conv2d import Conv2d
from .conv_transpose_2d import ConvTranspose2d

# Layers Dictionary
layer_dict = {
    "PermuteChannels": PermuteChannels,
    "Linear": Linear,
    "Dropout": Dropout,
    "Conv2d": Conv2d,
    "ConvTranspose2d": ConvTranspose2d,
}