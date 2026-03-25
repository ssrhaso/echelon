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

# Other
import copy

def frozen_network(net):

    net_frozen = copy.deepcopy(net)
    net_frozen.requires_grad_(False)

    return net_frozen

def get_module_and_params(module, module_dict):

    # List of Modules
    if isinstance(module, list):

        module_params = []
        for i in range(len(module)):

            # Type
            if isinstance(module[i], type):
                module_params.append({})

            # Dict
            elif isinstance(module[i], dict):
                module_params.append(module[i]["params"])
                module[i] = module_dict[module[i]["class"]]
                    
            # Str
            elif isinstance(module[i], str):
                module_params.append({})
                module[i] = module_dict[module[i]]

            # None
            elif module[i] is None:
                module_params.append({})
                module[i] = nn.Identity
    
    # Single Module
    else:

        # Type
        if isinstance(module, type):
            module_params = {}

        # Dict
        elif isinstance(module, dict):
            module_params = module["params"]
            module = module_dict[module["class"]]

        # Str
        elif isinstance(module, str):
            module_params = {}
            module = module_dict[module]

        # None
        elif module is None:
            module_params = {}
            module = nn.Identity

    return module, module_params