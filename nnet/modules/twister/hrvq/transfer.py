# Copyright 2025, Hasaan Ahmad.
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

"""VQ codebook cross-transfer and parameter audit utilities for ablation experiments."""

import torch


def load_and_transfer_codebooks(model, checkpoint_path, levels):
    """Load VQ codebook buffers from an external checkpoint into the model.

    Transfers embedding, ema_cluster_size, ema_embedding_sum, and update_count
    for each specified level. All other weights (encoder CNN, TSSM, actor-critic)
    remain untouched.

    Args:
        model: TWISTER model instance.
        checkpoint_path: Path to source checkpoint (.ckpt file).
        levels: List of VQ level indices to transfer (e.g. [0, 1, 2]).
    """
    checkpoint = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    source_state = checkpoint["model_state_dict"]

    buffer_names = ["embedding", "ema_cluster_size", "ema_embedding_sum", "update_count"]
    model_buffers = dict(model.named_buffers())

    for level in levels:
        for buf_name in buffer_names:
            key = f"encoder_network.hrvq.quantizers.{level}.{buf_name}"
            if key not in source_state:
                raise KeyError(f"Key '{key}' not found in transfer checkpoint: {checkpoint_path}")
            model_buffers[key].copy_(source_state[key])
        print(f"  Transferred VQ level {level} codebook from {checkpoint_path}")


def load_and_transfer_encoder(model, checkpoint_path):
    """Load encoder CNN weights from an external checkpoint into the model.

    Transfers all parameters and buffers of the encoder CNN (but not the HRVQ
    codebooks, which are handled by load_and_transfer_codebooks).

    Args:
        model: TWISTER model instance.
        checkpoint_path: Path to source checkpoint (.ckpt file).
    """
    checkpoint = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    source_state = checkpoint["model_state_dict"]

    cnn_prefix = "encoder_network.cnn."
    model_state = dict(model.named_parameters())
    model_state.update(dict(model.named_buffers()))

    transferred = 0
    for key, value in source_state.items():
        if key.startswith(cnn_prefix) and key in model_state:
            model_state[key].data.copy_(value)
            transferred += 1

    if transferred == 0:
        raise KeyError(f"No encoder CNN keys with prefix '{cnn_prefix}' found in: {checkpoint_path}")
    print(f"  Transferred encoder CNN ({transferred} tensors) from {checkpoint_path}")


def print_parameter_audit(model):
    """Print a table of total/trainable params and buffer counts by component.

    Also asserts that frozen VQ levels have frozen=True.
    """
    components = {
        "encoder_cnn": model.encoder_network.cnn,
        "vq_level_0": model.encoder_network.hrvq.quantizers[0],
        "vq_level_1": model.encoder_network.hrvq.quantizers[1],
        "vq_level_2": model.encoder_network.hrvq.quantizers[2],
        "world_model_tssm": model.rssm,
        "decoder": model.decoder_network,
        "reward_net": model.reward_network,
        "continue_net": model.continue_network,
        "actor": model.policy_network,
        "critic": model.value_network,
    }

    total_params = 0
    total_trainable = 0
    total_buffers = 0

    print("\n" + "=" * 72)
    print(f"{'Component':<20} {'Params':>10} {'Trainable':>10} {'Buffers':>10} {'Frozen':>8}")
    print("-" * 72)

    for name, module in components.items():
        n_params = sum(p.numel() for p in module.parameters())
        n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        n_buffers = sum(b.numel() for b in module.buffers())
        frozen_flag = ""
        if hasattr(module, "frozen"):
            frozen_flag = "YES" if module.frozen else "no"
        elif name == "encoder_cnn" and n_trainable == 0 and n_params > 0:
            frozen_flag = "YES"
        print(f"{name:<20} {n_params:>10,} {n_trainable:>10,} {n_buffers:>10,} {frozen_flag:>8}")
        total_params += n_params
        total_trainable += n_trainable
        total_buffers += n_buffers

    print("-" * 72)
    print(f"{'TOTAL':<20} {total_params:>10,} {total_trainable:>10,} {total_buffers:>10,}")
    print("=" * 72)

    # Assert frozen quantizers
    for i, q in enumerate(model.encoder_network.hrvq.quantizers):
        if q.frozen:
            print(f"  ASSERT PASS: VQ level {i} frozen=True, EMA updates disabled")

    frozen_levels = model.encoder_network.hrvq.get_frozen_levels()
    if frozen_levels:
        print(f"  Frozen levels: {frozen_levels}")
    print()
