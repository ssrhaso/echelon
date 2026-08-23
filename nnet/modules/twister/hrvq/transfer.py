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

from collections import OrderedDict

import torch


def _load_source_state(model, checkpoint_path):
    """Load and cache source checkpoint state dict (avoids redundant disk reads)."""
    checkpoint = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    return checkpoint["model_state_dict"]


def load_and_transfer_codebooks(model, checkpoint_path, levels, source_state=None):
    """Load VQ codebook buffers from an external checkpoint into the model.

    Transfers embedding, ema_cluster_size, ema_embedding_sum, and update_count
    for each specified level. All other weights (encoder CNN, TSSM, actor-critic)
    remain untouched.

    Args:
        model: TWISTER model instance.
        checkpoint_path: Path to source checkpoint (.ckpt file).
        levels: List of VQ level indices to transfer (e.g. [0, 1, 2]).
        source_state: Pre-loaded state dict (optional, avoids redundant disk read).
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

    Transfers all parameters and buffers of the encoder CNN (but not the HRVQ
    codebooks, which are handled by load_and_transfer_codebooks).

    Args:
        model: TWISTER model instance.
        checkpoint_path: Path to source checkpoint (.ckpt file).
        source_state: Pre-loaded state dict (optional, avoids redundant disk read).
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


# Modules whose parameter shapes depend on the size of the action space. The
# transfer targets do not share an action count (Pong, Demon Attack and Up'n'Down
# have 6 actions, Ms Pac-Man and Asterix 9), so these weights cannot be carried
# over from a Pong source in general. They are re-initialised in EVERY target --
# including the games whose action count happens to match Pong's -- because a
# baseline that silently transferred the policy head for two of four games would
# not be one condition but two.
ACTION_CONDITIONED_PREFIXES = (
    "policy_network.",       # output dim = num_actions
    "contrastive_network.",  # input dim = feat_size + t * num_actions
    "rssm.action_mixer.",    # input dim = stoch features concatenated with the action
)


def load_and_transfer_all(model, checkpoint_path, source_state=None):
    """Warm-start every task-agnostic weight from a source checkpoint.

    This is the weight-transfer baseline: unlike load_and_transfer_codebooks
    (codebooks only) and load_and_transfer_encoder (encoder CNN only), it copies
    the encoder, HRVQ, world model, decoder and the reward/value/continue heads.
    Only ACTION_CONDITIONED_PREFIXES are left at their random init. Nothing is
    frozen here -- freezing remains the caller's responsibility, so the baseline
    is a pure warm start unless --freeze_* is also passed.

    A shape mismatch outside the action-conditioned modules is an error, not a
    skip: it means the source and target are not the same architecture, and a
    baseline that quietly transferred half of what it claimed would be worse
    than no baseline at all. A source that is missing a target *parameter* is an
    error for the same reason; a missing buffer is reported but tolerated.

    Args:
        model: TWISTER model instance (already built for the TARGET game).
        checkpoint_path: Path to source checkpoint (.ckpt file).
        source_state: Pre-loaded state dict (optional, avoids redundant disk read).

    Returns:
        dict with keys 'transferred', 'reinit', 'missing_buffers', 'cast' --
        each a list of tensor names -- for provenance printing and W&B logging.
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

    # Every exclusion prefix must match something. If a module is ever renamed the
    # prefix silently stops excluding it, and the failure is invisible precisely
    # where it matters most: a target whose action count happens to equal the
    # source's (Demon Attack, Up'n'Down vs Pong) would quietly inherit the source
    # policy while the 9-action targets raised on shape. That asymmetry would look
    # like a result.
    unmatched = [p for p in ACTION_CONDITIONED_PREFIXES
                 if not any(k.startswith(p) for k in reinit)]
    if unmatched:
        raise RuntimeError(
            "full weight transfer aborted: action-conditioned prefix(es) {} matched no "
            "tensor in this model -- a module has been renamed and the exclusion list "
            "in transfer.py is stale.".format(unmatched)
        )

    return {
        "transferred": transferred,
        "reinit": reinit,
        "missing_buffers": missing_buffers,
        "cast": cast,
    }


def print_transfer_provenance(summary, checkpoint_path):
    """Print what the weight-transfer baseline actually carried over.

    The counts belong in the launch log for the same reason the stitching recipe
    does: the difference between this baseline and the codebook-only conditions
    is exactly which tensors moved, and that should be checkable from stdout
    without re-deriving it from the flags.
    """
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


def parse_codebook_map(map_str, sources_str, num_levels=3):
    """Parse the per-level codebook stitching specification.

    The stitching matrix transfers each RVQ level's codebook from a (possibly
    different) donor checkpoint, e.g. level 0 from Pong, levels 1-2 from Demon
    Attack. This decouples the donor *per level* rather than per model.

    Args:
        map_str:     "0:pong,1:da,2:da" -- donor name per RVQ level.
        sources_str: "pong=path/to/pong.ckpt,da=path/to/da.ckpt" -- donor paths.
        num_levels:  number of RVQ levels that must be covered exactly once.

    Returns:
        OrderedDict {level: (donor_name, checkpoint_path)} ordered by level.

    Raises:
        ValueError on malformed input, an unknown donor name, a duplicate or
        out-of-range level, or incomplete coverage of [0, num_levels).
    """
    if not sources_str:
        raise ValueError("--codebook_sources is required with --codebook_map")

    sources = {}
    for pair in sources_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, sep, path = pair.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(
                f"--codebook_sources entry must be 'name=path' (got {pair!r})"
            )
        sources[name.strip()] = path.strip()

    level_map = {}
    for item in map_str.split(","):
        item = item.strip()
        if not item:
            continue
        lvl_s, sep, donor = item.partition(":")
        if not sep:
            raise ValueError(
                f"--codebook_map entry must be 'level:donor' (got {item!r})"
            )
        try:
            lvl = int(lvl_s.strip())
        except ValueError:
            raise ValueError(f"--codebook_map level must be an integer (got {lvl_s!r})")
        donor = donor.strip()
        if lvl in level_map:
            raise ValueError(f"--codebook_map level {lvl} specified more than once")
        if not 0 <= lvl < num_levels:
            raise ValueError(
                f"--codebook_map level {lvl} out of range [0, {num_levels})"
            )
        if donor not in sources:
            raise ValueError(
                f"--codebook_map donor {donor!r} for level {lvl} is not defined in "
                f"--codebook_sources (known: {sorted(sources)})"
            )
        level_map[lvl] = (donor, sources[donor])

    missing = [lvl for lvl in range(num_levels) if lvl not in level_map]
    if missing:
        raise ValueError(
            f"--codebook_map must cover every RVQ level [0, {num_levels}); "
            f"missing {missing}"
        )

    return OrderedDict(sorted(level_map.items()))


def load_and_transfer_codebook_map(model, level_map):
    """Stitch codebooks: load each level from its mapped donor checkpoint.

    Levels mapped to the same donor share a single disk read. Only VQ buffers
    are touched (encoder CNN / world model / actor-critic are untouched);
    freezing is the caller's responsibility (stitched levels must be frozen so
    the donor codebook is preserved rather than overwritten by EMA).

    Args:
        model: TWISTER model instance.
        level_map: OrderedDict {level: (donor_name, path)} from parse_codebook_map.
    """
    by_path = OrderedDict()
    for lvl, (donor, path) in level_map.items():
        by_path.setdefault(path, []).append(lvl)

    state_cache = {}
    for path, levels in by_path.items():
        if path not in state_cache:
            state_cache[path] = _load_source_state(model, path)
        load_and_transfer_codebooks(model, path, sorted(levels), state_cache[path])

    return state_cache


def verify_codebook_map(model, level_map, state_cache=None):
    """Assert each frozen level's embedding bit-matches its donor checkpoint.

    A safety net for the stitching matrix: across a 24-run grid, one
    mis-specified cell (wrong donor, partial load) would silently invalidate a
    whole arm. We re-read each donor's level-l embedding and require exact
    equality with what now sits in the model. Returns {level: (donor, sha1)}
    fingerprints for the launch log so two runs of the same cell are visibly
    identical.

    Raises RuntimeError on any mismatch.
    """
    import hashlib

    print("\nCodebook stitch verification:")
    fingerprints = {}
    for lvl, (donor, path) in sorted(level_map.items()):
        src = state_cache[path] if state_cache and path in state_cache else _load_source_state(model, path)
        want = src[f"encoder_network.hrvq.quantizers.{lvl}.embedding"].detach().cpu()
        got = model.encoder_network.hrvq.quantizers[lvl].embedding.detach().cpu()
        if not torch.equal(got, want):
            raise RuntimeError(
                f"codebook stitch verification FAILED at L{lvl} "
                f"(donor={donor}, {path}): loaded embedding does not match donor"
            )
        sha = hashlib.sha1(want.numpy().tobytes()).hexdigest()[:10]
        fingerprints[lvl] = (donor, sha)
        print(f"  L{lvl} <- {donor:<10} match=YES  sha1={sha}")
    return fingerprints


def print_codebook_provenance(level_map):
    """Print which donor supplies each RVQ level (the stitching-matrix cell).

    Surfaces the exact recipe in the launch log so the per-run sanity check can
    confirm the intended cell before committing hours of compute.
    """
    print("\nCodebook provenance (stitching cell):")
    for lvl, (donor, path) in sorted(level_map.items()):
        print(f"  L{lvl} <- {donor:<10} {path}")
    recipe = "[" + ",".join(level_map[lvl][0] for lvl in sorted(level_map)) + "]"
    print(f"  recipe: {recipe}")


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
