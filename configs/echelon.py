# Copyright 2025, Maxime Burchi.
# Modifications Copyright 2025-2026, Hasaan Ahmad, adapted for ECHELON.
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

import nnet
import os
import json

# Extract params from filename
env_name = os.environ["env_name"]
print("ECHELON selected env_name: {}".format(env_name))

# Override Config
override_config = os.environ.get("override_config", {})
if isinstance(override_config, str):
    override_config = json.loads(override_config)
print("override_config:", override_config)

# Model
model = nnet.models.TWISTER(env_name=env_name, override_config=override_config)
model.compile()

# Training
precision = model.config.precision
grad_init_scale = model.config.grad_init_scale
epochs = model.config.epochs
epoch_length = model.config.epoch_length

# Callback Path
if os.environ.get("run_name", False):
    callback_path = "callbacks/{}/{}".format(os.environ["run_name"], env_name)
else:
    callback_path = "callbacks/{}".format(env_name)

# Replay Buffer
training_dataset = nnet.datasets.ReplayBuffer(
    batch_size=model.config.batch_size,
    root=callback_path,
    buffer_capacity=model.config.buffer_capacity,
    epoch_length=epoch_length,
    sample_length=model.config.L,
    save_trajectories=False
)
model.set_replay_buffer(training_dataset)

# Evaluation Dataset
evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_episodes)
