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

"""Cross-game transfer of HRVQ codebooks and encoder weights, plus the launch-time
parameter audit."""

from collections import OrderedDict

import torch


def _load_source_state(model, checkpoint_path):
    """Return the model state dict of a source checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    return checkpoint["model_state_dict"]


def load_and_transfer_codebooks(model, checkpoint_path, levels, source_state=None):
    """Load VQ codebook buffers from an external checkpoint into the model.

    Copies embedding, ema_cluster_size, ema_embedding_sum and update_count for
    each level named; nothing else is touched.

    Args:
        model: TWISTER model instance.
        checkpoint_path: Source checkpoint (.ckpt).
        levels: HRVQ level indices to transfer, e.g. [0, 1, 2].
        source_state: Pre-loaded state dict, to avoid a second disk read.
    """
    if source_state is None:
        source_state = _load_source_state(model, checkpoint_path)

    buffer_names = ["embedding", "ema_cluster_size", "ema_embedding_sum", "update_count"]
    model_buffers = dict(model.named_buffers())

    for level in levels:
        for buf_name in buffer_names:
            key = f"encoder_network.hrvq.quantizers.{level}.{buf_name}"
            if key not in source_state:
                raise KeyError(f"Key '{key}' not found in transfer checkpoint: {checkpoint_path}")
            model_buffers[key].copy_(source_state[key])
        print(f"  Transferred VQ level {level} codebook from {checkpoint_path}")


def load_and_transfer_encoder(model, checkpoint_path, source_state=None):
    """Load encoder CNN weights from an external checkpoint into the model.

    Copies every parameter and buffer of the encoder CNN; the codebooks are
    handled by load_and_transfer_codebooks.

    Args:
        model: TWISTER model instance.
        checkpoint_path: Source checkpoint (.ckpt).
        source_state: Pre-loaded state dict, to avoid a second disk read.
    """
    if source_state is None:
        source_state = _load_source_state(model, checkpoint_path)

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


# Modules whose parameter shapes depend on the action-space size. Re-initialised in
# every target, even one whose action count matches the source, so the condition is
# uniform across games.
ACTION_CONDITIONED_PREFIXES = (
    "policy_network.",       # output dim = num_actions
    "contrastive_network.",  # input dim = feat_size + t * num_actions
    "rssm.action_mixer.",    # input dim = stoch features concatenated with the action
)


def load_and_transfer_all(model, checkpoint_path, source_state=None):
    """Warm-start every task-agnostic weight from a source checkpoint.

    The whole-model-transfer baseline: copies the encoder, HRVQ, world model,
    decoder and the reward, value and continue heads, leaving only
    ACTION_CONDITIONED_PREFIXES at random init. Nothing is frozen, so the result
    is a pure warm start unless --freeze_* is also passed.

    A shape mismatch outside the action-conditioned modules, or a missing target
    parameter, aborts: the checkpoint is then a different architecture and a
    partial transfer would misreport the baseline. A missing buffer is tolerated.

    Args:
        model: TWISTER model instance, already built for the target game.
        checkpoint_path: Source checkpoint (.ckpt).
        source_state: Pre-loaded state dict, to avoid a second disk read.

    Returns:
        Dict of tensor-name lists under 'transferred', 'reinit',
        'missing_buffers' and 'cast', for provenance printing and W&B logging.
    """
    if source_state is None:
        source_state = _load_source_state(model, checkpoint_path)

    params = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())
    target = OrderedDict(params)
    target.update(buffers)

    transferred, reinit, missing_params, missing_buffers, cast, mismatched = [], [], [], [], [], []

    for key, tensor in target.items():
        if key.startswith(ACTION_CONDITIONED_PREFIXES):
            reinit.append(key)
            continue
        if key not in source_state:
            (missing_params if key in params else missing_buffers).append(key)
            continue
        src = source_state[key]
        if tuple(src.shape) != tuple(tensor.shape):
            mismatched.append((key, tuple(src.shape), tuple(tensor.shape)))
            continue
        if src.dtype != tensor.dtype:
            cast.append(key)
        tensor.data.copy_(src)
        transferred.append(key)

    if mismatched:
        detail = "\n".join(
            "    {}: source {} vs target {}".format(k, s, t) for k, s, t in mismatched
        )
        raise RuntimeError(
            "full weight transfer aborted: {} tensor(s) differ in shape outside the "
            "action-conditioned modules, so {} is not the same architecture as this "
            "run:\n{}".format(len(mismatched), checkpoint_path, detail)
        )
    if missing_params:
        raise KeyError(
            "full weight transfer aborted: {} target parameter(s) absent from {} "
            "(first few: {}). Transferring a subset silently would misreport the "
            "baseline.".format(len(missing_params), checkpoint_path, missing_params[:5])
        )
    if not transferred:
        raise KeyError("no transferable tensors found in: {}".format(checkpoint_path))

    # A renamed module would drop out of the exclusion list silently and let a
    # matching-action target inherit the source policy.
    unmatched = [p for p in ACTION_CONDITIONED_PREFIXES
                 if not any(k.startswith(p) for k in reinit)]
    if unmatched:
        raise RuntimeError(
            "full weight transfer aborted: action-conditioned prefix(es) {} matched no "
            "tensor in this model; a module has been renamed and the exclusion list "
            "in transfer.py is stale.".format(unmatched)
        )

    return {
        "transferred": transferred,
        "reinit": reinit,
        "missing_buffers": missing_buffers,
        "cast": cast,
    }


def print_transfer_provenance(summary, checkpoint_path):
    """Print which tensors the whole-model-transfer baseline carried over."""
    print("\nWeight-transfer provenance (source: {}):".format(checkpoint_path))
    print("  transferred     : {} tensors".format(len(summary["transferred"])))
    print("  re-initialised  : {} tensors (action-conditioned)".format(len(summary["reinit"])))
    for prefix in ACTION_CONDITIONED_PREFIXES:
        n = sum(1 for k in summary["reinit"] if k.startswith(prefix))
        print("      {:<24} {} tensors".format(prefix + "*", n))
    if summary["missing_buffers"]:
        print("  WARNING: {} target buffer(s) absent from source and left at init: {}".format(
            len(summary["missing_buffers"]), summary["missing_buffers"][:5]))
    if summary["cast"]:
        print("  NOTE: {} tensor(s) dtype-cast on copy (source precision differs from "
              "this run's)".format(len(summary["cast"])))


def print_parameter_audit(model):
    """Print per-component parameter, trainable and buffer counts, and the frozen
    state of each HRVQ level."""
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

    for i, q in enumerate(model.encoder_network.hrvq.quantizers):
        if q.frozen:
            print(f"  ASSERT PASS: VQ level {i} frozen=True, EMA updates disabled")

    frozen_levels = model.encoder_network.hrvq.get_frozen_levels()
    if frozen_levels:
        print(f"  Frozen levels: {frozen_levels}")
    print()
