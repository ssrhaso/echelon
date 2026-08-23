"""Basic mechanistic analysis on existing checkpoint(s) - CPU friendly.

Takes any number of HRVQ checkpoints, runs each one on a SHARED set of frames
(deterministic from --frame_seed), and reports per-checkpoint:
 - Codebook usage per level (active codes, perplexity)
 - Residual quant error per level (from the forward pass)
 - Per-level cascade reconstruction MSE (L0, L0+L1, L0+L1+L2)
For 2+ checkpoints, also produces pairwise Jaccard of used code IDs and
per-code nearest-neighbour cosine of the codebook embeddings.

Examples (run from repo root):

  # Single ckpt
  python scripts/mech_analysis_basic.py \
      --ckpts checkpoints_local/breakout_freeze-L012_seed5/best.ckpt

  # Compare conditions on shared Breakout frames
  python scripts/mech_analysis_basic.py \
      --ckpts \
          checkpoints_local/breakout_freeze-L012_seed5/best.ckpt \
          checkpoints_local/breakout_freeze-all_seed5/best.ckpt \
      --n_frames 256 --frame_seed 0
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime

import torch
import torch.nn.functional as F


def _tensor_sha1(t: torch.Tensor) -> str:
    """Stable hash of a tensor's bytes - confirms two runs got identical frames."""
    return hashlib.sha1(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

# Repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nnet  # noqa: E402
from nnet.modules.twister.hrvq import spatial_cascade_decode  # noqa: E402


GAME_FROM_DIRNAME = {
    "breakout": "breakout",
    "mspacman": "ms_pacman",
    "ms_pacman": "ms_pacman",
    "pong": "pong",
}

CKPT_DIR_RE = re.compile(
    r"^(?P<game>[a-z_]+?)_(?P<condition>scratch|warmstart|freeze-(?:enc|L0|L01|L012|all))"
    r"(?:_seed(?P<seed>\d+))?$"
)


def parse_ckpt_dirname(ckpt_path: str) -> dict:
    """Parse 'breakout_freeze-L012_seed5' -> {game, condition, seed}."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    m = CKPT_DIR_RE.match(parent)
    if m:
        return {
            "dirname": parent,
            "game": m.group("game"),
            "condition": m.group("condition"),
            "seed": int(m.group("seed")) if m.group("seed") else None,
        }
    return {"dirname": parent, "game": None, "condition": None, "seed": None}


def infer_env_name(ckpt_path: str) -> str:
    """Infer 'atari100k-<game>' from a path like checkpoints_local/<dir>/best.ckpt."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    token = re.split(r"[_-]", parent)[0].lower()
    if token not in GAME_FROM_DIRNAME:
        raise SystemExit(
            f"Cannot infer game from '{parent}'. Pass --frame_env explicitly."
        )
    return f"atari100k-{GAME_FROM_DIRNAME[token]}"


def build_model(env_name: str, device: str = "cpu"):
    # Avoid creating eval env / prefilling buffer - we drive the env manually.
    model = nnet.models.TWISTER(
        env_name=env_name,
        override_config={"eval_episodes": 0, "pre_fill_steps": 0},
    )
    model.to(device).eval()
    return model


def load_weights(model, ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if missing:
        print(f"  [warn] {len(missing)} missing key(s), first 3: {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected key(s), first 3: {unexpected[:3]}")
    print(f"  loaded {ckpt_path}")


@torch.no_grad()
def _reset_rssm_state(model, device):
    """Initial (prev_latent, prev_action) for an episode."""
    prev_latent = model.rssm.initial(
        batch_size=1, seq_length=1, dtype=torch.float32, detach_learned=True
    )
    prev_action = torch.zeros(1, model.env.num_actions, dtype=torch.float32, device=device)
    return prev_latent, prev_action


@torch.no_grad()
def collect_frames(
    model,
    n_frames: int,
    frame_seed: int,
    policy: str = "random",
    device: str = "cpu",
) -> torch.Tensor:
    """Returns (N, 3, 64, 64) uint8 frames, fully reproducible from `frame_seed`.

    policy="random": one-hot uniform actions via env.sample(). Cheap, but the
        frame distribution is biased (early-death, idle screens) so codebook
        usage gets under-counted.
    policy="agent":  rollout under `model`'s trained policy (encoder -> RSSM
        -> policy network). In-distribution for the encoder; matches what the
        model saw during training. Slower per frame but the right thing for a
        paper figure.

    Determinism sources covered:
 - ALE reset seed (initial RAM / sticky-action draws)
 - torch RNG for env.sample() and policy.sample() draws
    """
    assert policy in ("random", "agent")
    env = model.env  # BatchEnv with num_envs=1 for atari100k
    env.envs[0].seed(frame_seed)
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(frame_seed)
    try:
        obs = env.reset()
        frames = [obs.state[0]]
        if policy == "agent":
            prev_latent, prev_action = _reset_rssm_state(model, device)
        while len(frames) < n_frames:
            if policy == "random":
                action = env.sample()
            else:
                state = obs.state.to(device)
                x = model.preprocess_inputs(state, time_stacked=False)
                enc_out = model.encoder_network(x)
                latent_in = {"stoch": enc_out["stoch"].unsqueeze(dim=1)}
                latent, _ = model.rssm(
                    states=latent_in,
                    prev_states=prev_latent,
                    prev_actions=prev_action.unsqueeze(dim=1),
                    is_firsts=torch.zeros(1, 1, device=device),
                    is_firsts_hidden=None,
                )
                feat = model.rssm.get_feat(latent).squeeze(dim=1)
                action = model.policy_network(feat).sample()
                latent["hidden"] = model.rssm.slice_hidden(latent["hidden"])
                prev_latent, prev_action = latent, action
            obs = env.step(action.argmax(dim=-1))
            frames.append(obs.state[0])
            if obs.is_last[0].item() > 0:
                env.envs[0].seed(frame_seed + len(frames))
                obs = env.reset()
                frames.append(obs.state[0])
                if policy == "agent":
                    prev_latent, prev_action = _reset_rssm_state(model, device)
        frames = torch.stack(frames[:n_frames], dim=0).to(device)
    finally:
        torch.random.set_rng_state(gen_state)
    return frames


@torch.no_grad()
def encode(model, frames: torch.Tensor) -> dict:
    x = model.preprocess_inputs(frames, time_stacked=False)  # (N, 3, 64, 64) float32
    out = model.encoder_network(x)
    info = out["hrvq_info"]
    return {
        "x_target": x,
        "indices": info["indices"],                       # 3x (N, 16) Long
        "z_q_levels_spatial": info["z_q_levels_spatial"], # 3x (N, 16, 256)
        "residual_errors": [e.item() for e in info["residual_errors"]],
        "pre_vq_features": out["pre_vq_features"],         # (N, 16*256) continuous z
    }


def codebook_usage(indices, num_codes: int):
    stats = []
    for idx in indices:
        flat = idx.reshape(-1)
        active = flat.unique().numel()
        counts = torch.bincount(flat, minlength=num_codes).float()
        probs = counts / counts.sum().clamp_min(1)
        perp = torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item()
        stats.append({"active": active, "usage": active / num_codes, "perplexity": perp})
    return stats


def index_overlap_pair(idx_a, idx_b):
    out = []
    for a, b in zip(idx_a, idx_b):
        ua = set(a.reshape(-1).tolist())
        ub = set(b.reshape(-1).tolist())
        jacc = len(ua & ub) / max(1, len(ua | ub))
        out.append({"used_A": len(ua), "used_B": len(ub), "jaccard": jacc})
    return out


@torch.no_grad()
def embedding_overlap_pair(hrvq_a, hrvq_b):
    out = []
    for lvl in range(hrvq_a.num_levels):
        ea = F.normalize(hrvq_a.quantizers[lvl].embedding.float(), dim=-1)
        eb = F.normalize(hrvq_b.quantizers[lvl].embedding.float(), dim=-1)
        sim = ea @ eb.T
        nn_cos = sim.max(dim=1).values
        out.append({
            "mean_nn_cos": nn_cos.mean().item(),
            "median_nn_cos": nn_cos.median().item(),
        })
    return out


@torch.no_grad()
def codebook_drift_from_source(hrvq_target, hrvq_source, identity_tol: float = 1e-7):
    """Per-level drift of target codebook vs source codebook (matched by index).

    For freeze conditions the source-derived levels should be byte-identical:
    L2 dist = 0, matched cosine = 1, identity_rate = 1.0. Anything else flags
    a transfer/freezing bug. For warmstart/scratch these are the actual paper
    numbers: how far did training move each level?
    """
    out = []
    for lvl in range(hrvq_target.num_levels):
        ct = hrvq_target.quantizers[lvl].embedding.float()
        cs = hrvq_source.quantizers[lvl].embedding.float()
        if ct.shape != cs.shape:
            raise ValueError(
                f"Level {lvl}: codebook shape mismatch target={ct.shape} source={cs.shape}"
            )
        delta = ct - cs                                                # (K, D)
        l2 = delta.pow(2).sum(dim=-1).sqrt()                            # (K,)
        rel_l2 = l2 / cs.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)
        matched_cos = F.cosine_similarity(ct, cs, dim=-1)               # (K,)
        identity_rate = (l2 < identity_tol).float().mean().item()
        out.append({
            "mean_l2": l2.mean().item(),
            "median_l2": l2.median().item(),
            "mean_rel_l2": rel_l2.mean().item(),
            "mean_matched_cos": matched_cos.mean().item(),
            "median_matched_cos": matched_cos.median().item(),
            "identity_rate": identity_rate,
        })
    return out


@torch.no_grad()
def encoder_feature_drift(target_pre_vq, source_pre_vq, position_dim):
    """Diagnostic D3: rho_enc = mean cosine between source- and target-encoder
    continuous features z (pre-VQ) on the SAME held-out frames.

    Both inputs are (N, num_positions*position_dim); we reshape to per-position
    (N*num_positions, position_dim) and average the cosine over all positions.
    rho_enc == 1.0 by construction when the encoder is frozen from source
    (freeze-enc / freeze-all); values < 1 quantify STE-induced encoder drift.
    """
    t = target_pre_vq.reshape(-1, position_dim).float()
    s = source_pre_vq.reshape(-1, position_dim).float()
    cos = F.cosine_similarity(t, s, dim=-1)
    return {"rho_enc_mean": cos.mean().item(), "rho_enc_median": cos.median().item()}


@torch.no_grad()
def cascade_reconstruction(
    model, x_target, z_q_levels_spatial, return_samples: int = 0,
):
    """MSE of cascade decode (levels 0..k) vs preprocessed target.

    If return_samples > 0, also returns the first `return_samples` decoded
    frames per cascade depth (shape (return_samples, 3, 64, 64), in [-0.5, 0.5]).
    """
    out = []
    samples = [] if return_samples > 0 else None
    for up_to in range(len(z_q_levels_spatial)):
        obs_dist = spatial_cascade_decode(
            model.decoder_network,
            z_q_levels_spatial,
            up_to_level=up_to,
            dim_cnn=model.config.dim_cnn,
        )
        x_rec = obs_dist.mode()
        mse = F.mse_loss(x_rec, x_target).item()
        out.append({"levels_used": list(range(up_to + 1)), "mse": mse})
        if samples is not None:
            samples.append(x_rec[:return_samples].detach().clone())
    if samples is not None:
        return out, samples
    return out


def _label(meta: dict, ckpt_path: str) -> str:
    if meta.get("condition") is not None:
        seed = meta["seed"]
        return f"{meta['game']}/{meta['condition']}" + (f"/s{seed}" if seed is not None else "")
    return os.path.basename(os.path.dirname(ckpt_path))


def _print_per_ckpt_block(label: str, n_tokens: int, num_codes: int,
                          usage, residual_errors, recon, drift=None,
                          enc_drift=None):
    print(f"\n## {label}")
    print(f"  usage (N={n_tokens} tokens/level):")
    for lvl, s in enumerate(usage):
        print(f"    L{lvl}: active {s['active']:>4}/{num_codes}"
              f"  ({s['usage']*100:5.1f}%)  perp={s['perplexity']:7.2f}")
    print(f"  residual quant error (||r-zq||^2 / ||r||^2):")
    for lvl, e in enumerate(residual_errors):
        print(f"    L{lvl}: {e:.4f}")
    print(f"  cascade reconstruction MSE:")
    for r in recon:
        levels = "+".join(f"L{l}" for l in r["levels_used"])
        print(f"    {levels:<10} {r['mse']:.6f}")
    if drift is not None:
        print(f"  codebook drift from source (matched-index, K={num_codes}):")
        for lvl, d in enumerate(drift):
            print(f"    L{lvl}: mean L2={d['mean_l2']:7.4f}  rel={d['mean_rel_l2']:.4f}"
                  f"  matched cos={d['mean_matched_cos']:6.3f}"
                  f"  identity={d['identity_rate']*100:5.1f}%")
    if enc_drift is not None:
        print(f"  encoder feature drift vs source (rho_enc, D3): "
              f"mean={enc_drift['rho_enc_mean']:.4f}  median={enc_drift['rho_enc_median']:.4f}")


def _print_summary_table(rows, num_codes, with_drift: bool):
    """One row per ckpt, key metrics in columns."""
    print("\n## Summary (one row per checkpoint)")
    headers = [
        "label",
        "L0 act%", "L1 act%", "L2 act%",
        "L0 perp", "L1 perp", "L2 perp",
        "MSE L0", "MSE L01", "MSE L012",
    ]
    if with_drift:
        headers += ["L0 drift", "L1 drift", "L2 drift",
                    "L0 idnt%", "L1 idnt%", "L2 idnt%"]
    widths = [max(len(h), 22) if i == 0 else max(len(h), 8) for i, h in enumerate(headers)]
    line = "  " + " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        cells = [
            r["label"],
            f"{r['usage'][0]['usage']*100:5.1f}",
            f"{r['usage'][1]['usage']*100:5.1f}",
            f"{r['usage'][2]['usage']*100:5.1f}",
            f"{r['usage'][0]['perplexity']:6.2f}",
            f"{r['usage'][1]['perplexity']:6.2f}",
            f"{r['usage'][2]['perplexity']:6.2f}",
            f"{r['recon'][0]['mse']:.6f}",
            f"{r['recon'][1]['mse']:.6f}",
            f"{r['recon'][2]['mse']:.6f}",
        ]
        if with_drift:
            d = r.get("drift") or [None] * 3
            for lvl in range(3):
                cells.append(f"{d[lvl]['mean_l2']:.4f}" if d[lvl] else "  -- ")
            for lvl in range(3):
                cells.append(f"{d[lvl]['identity_rate']*100:5.1f}" if d[lvl] else "  -- ")
        print("  " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)))


def _compute_pairwise(rows, models):
    """For every (i<j) ckpt pair and each HRVQ level, compute Jaccard + NN cos."""
    n = len(rows)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            per_level = []
            jacc_list = index_overlap_pair(rows[i]["indices"], rows[j]["indices"])
            cos_list = embedding_overlap_pair(
                models[i].encoder_network.hrvq, models[j].encoder_network.hrvq
            )
            for lvl in range(len(jacc_list)):
                per_level.append({
                    "level": lvl,
                    "jaccard": jacc_list[lvl]["jaccard"],
                    "used_a": jacc_list[lvl]["used_A"],
                    "used_b": jacc_list[lvl]["used_B"],
                    "mean_nn_cos": cos_list[lvl]["mean_nn_cos"],
                    "median_nn_cos": cos_list[lvl]["median_nn_cos"],
                })
            out.append({
                "a": rows[i]["label"],
                "b": rows[j]["label"],
                "levels": per_level,
            })
    return out


def _print_pairwise(rows, pairwise):
    """Per-level NxN matrix of Jaccard / mean NN cosine of codebook embeddings."""
    if not pairwise:
        return
    n = len(rows)
    # Materialise NxN lookup from upper-triangular pairwise list.
    lookup = {(p["a"], p["b"]): p for p in pairwise}
    print("\n## Pairwise codebook overlap (rows used the SAME frames)")
    for lvl in range(3):
        print(f"\n  -- Level L{lvl} : Jaccard of used IDs / mean NN cosine of codebook --")
        head = "    " + "".ljust(24) + "".join(
            f"{rows[j]['label'][:18]:>20}" for j in range(n)
        )
        print(head)
        for i in range(n):
            row_cells = []
            for j in range(n):
                if i == j:
                    row_cells.append("   .  /   .  ")
                    continue
                key = (rows[i]["label"], rows[j]["label"])
                if key not in lookup:
                    key = (rows[j]["label"], rows[i]["label"])
                lvl_data = lookup[key]["levels"][lvl]
                row_cells.append(f"{lvl_data['jaccard']:5.3f} / {lvl_data['mean_nn_cos']:5.3f}")
            print(f"    {rows[i]['label'][:22]:<22}" + "".join(f"{c:>20}" for c in row_cells))


def _write_outputs(output_dir, *, tag, frame_env, frame_seed, frame_policy,
                   frame_policy_ckpt, n_frames, ckpts, source_ckpt,
                   frame_sha1, num_codes, rows, pairwise):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # results.json - everything we computed, plus provenance.
    results = {
        "schema_version": 1,
        "tag": tag,
        "timestamp_utc": ts,
        "frame_env": frame_env,
        "frame_seed": frame_seed,
        "frame_policy": frame_policy,
        "frame_policy_ckpt": frame_policy_ckpt or ckpts[0],
        "frame_sha1": frame_sha1,
        "n_frames": n_frames,
        "num_codes": num_codes,
        "ckpts": ckpts,
        "source_ckpt": source_ckpt,
        "rows": [
            {
                "label": r["label"],
                "meta": r["meta"],
                "usage": r["usage"],
                "residual_errors": r["residual_errors"],
                "recon": r["recon"],
                "drift": r["drift"],
                "enc_drift": r.get("enc_drift"),
            }
            for r in rows
        ],
        "pairwise": pairwise,
    }
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[write] {json_path}")

    # summary.csv - long format, one row per (ckpt, level). Easy for pandas.
    csv_path = os.path.join(output_dir, "summary.csv")
    fields = [
        "tag", "label", "game", "condition", "seed", "level",
        "num_codes", "active_codes", "usage_pct", "perplexity",
        "residual_err", "recon_mse_cumulative",
        "drift_mean_l2", "drift_mean_rel_l2", "drift_matched_cos",
        "drift_identity_rate", "rho_enc",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            for lvl in range(len(r["usage"])):
                d = (r["drift"][lvl] if r["drift"] else None)
                w.writerow({
                    "tag": tag,
                    "label": r["label"],
                    "game": r["meta"].get("game"),
                    "condition": r["meta"].get("condition"),
                    "seed": r["meta"].get("seed"),
                    "level": lvl,
                    "num_codes": num_codes,
                    "active_codes": r["usage"][lvl]["active"],
                    "usage_pct": r["usage"][lvl]["usage"] * 100,
                    "perplexity": r["usage"][lvl]["perplexity"],
                    "residual_err": r["residual_errors"][lvl],
                    "recon_mse_cumulative": r["recon"][lvl]["mse"],
                    "drift_mean_l2": d["mean_l2"] if d else "",
                    "drift_mean_rel_l2": d["mean_rel_l2"] if d else "",
                    "drift_matched_cos": d["mean_matched_cos"] if d else "",
                    "drift_identity_rate": d["identity_rate"] if d else "",
                    # rho_enc is one scalar per ckpt; repeat on each level row.
                    "rho_enc": (r["enc_drift"]["rho_enc_mean"]
                                if r.get("enc_drift") else ""),
                })
    print(f"[write] {csv_path}")


def _agg_by_condition(rows, metric_fn):
    """Group rows by (game, condition) and apply metric_fn per group.
    metric_fn(rows_in_group) -> dict[level] -> list of per-seed values.
    Returns: ordered list of (label, dict[level] -> list).
    """
    from collections import OrderedDict
    groups = OrderedDict()
    for r in rows:
        key = (r["meta"].get("game"), r["meta"].get("condition"))
        if any(k is None for k in key):
            key = (None, r["label"])  # fall back to label if dirname unparseable
        groups.setdefault(key, []).append(r)
    out = []
    for (game, cond), grp in groups.items():
        out.append((f"{game}/{cond}" if game else cond, metric_fn(grp)))
    return out


def _write_plots(output_dir, *, rows, num_codes, frames, with_drift, with_pairwise):
    """Two paper-ready PNGs:
      mech_summary.png        : 2x2 metrics figure across conditions
      mech_reconstructions.png: per-ckpt cascade image grid
    Aggregation: within each (game, condition), shows mean as the bar and
    individual seeds as scatter points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    LEVELS = ["L0", "L1", "L2"]
    CASCADE = ["L0", "L0+L1", "L0+L1+L2"]

    # Aggregations: dict per condition, per level, list of seed values
    def usage_pct(group):
        return {lvl: [r["usage"][lvl]["usage"] * 100 for r in group] for lvl in range(3)}

    def recon_mse(group):
        return {lvl: [r["recon"][lvl]["mse"] for r in group] for lvl in range(3)}

    def drift_l2(group):
        return {lvl: [r["drift"][lvl]["mean_l2"] for r in group if r["drift"]] for lvl in range(3)}

    def perplexity(group):
        return {lvl: [r["usage"][lvl]["perplexity"] for r in group] for lvl in range(3)}

    usage_by_cond = _agg_by_condition(rows, usage_pct)
    recon_by_cond = _agg_by_condition(rows, recon_mse)
    persistence_panel_data = (
        _agg_by_condition(rows, drift_l2) if with_drift
        else _agg_by_condition(rows, perplexity)
    )
    persistence_label = "Codebook L2 drift from source" if with_drift else "Codebook perplexity"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("HRVQ mechanistic analysis", fontsize=14, y=0.995)

    def _grouped_bars(ax, data, ylabel, log=False):
        conditions = [c for c, _ in data]
        x = np.arange(len(conditions))
        width = 0.25
        for li, lvl in enumerate(range(3)):
            means = [np.mean(d[lvl]) if d[lvl] else np.nan for _, d in data]
            ax.bar(x + (li - 1) * width, means, width, label=LEVELS[li], alpha=0.85)
            # Scatter individual seeds
            for ci, (_, d) in enumerate(data):
                if d[lvl]:
                    ax.scatter(
                        [ci + (li - 1) * width] * len(d[lvl]),
                        d[lvl], color="black", s=8, zorder=3,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    _grouped_bars(axes[0, 0], usage_by_cond, "Active codes (%)")
    axes[0, 0].set_title("Codebook usage per level")

    # Reconstruction: x is cascade depth, one line per condition
    ax = axes[0, 1]
    for cond, d in recon_by_cond:
        means = [np.mean(d[lvl]) if d[lvl] else np.nan for lvl in range(3)]
        ax.plot(CASCADE, means, marker="o", label=cond)
        for li, lvl in enumerate(range(3)):
            if d[lvl]:
                ax.scatter([li] * len(d[lvl]), d[lvl], color="black", s=8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("MSE (log)")
    ax.set_title("Cascade reconstruction (lower is better)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    _grouped_bars(axes[1, 0], persistence_panel_data, persistence_label,
                  log=with_drift is False)
    axes[1, 0].set_title(persistence_label + " per level")

    # Bottom-right panel: identity_rate (% byte-identical to source) when we
    # have a source ckpt - this is the cleanest "what got frozen" picture.
    # Otherwise: residual quant error per level.
    ax = axes[1, 1]
    if with_drift:
        def identity_rate(group):
            return {lvl: [r["drift"][lvl]["identity_rate"] * 100
                          for r in group if r["drift"]] for lvl in range(3)}
        data = _agg_by_condition(rows, identity_rate)
        _grouped_bars(ax, data, "Identity rate vs source (%)")
        ax.set_title("Codebook identity rate vs source")
        ax.set_ylim(-2, 102)
    else:
        def residual(group):
            return {lvl: [r["residual_errors"][lvl] for r in group] for lvl in range(3)}
        data = _agg_by_condition(rows, residual)
        _grouped_bars(ax, data, "Relative residual error")
        ax.set_title("Residual quant error per level")

    fig.tight_layout()
    summary_path = os.path.join(output_dir, "mech_summary.png")
    fig.savefig(summary_path, dpi=140)
    plt.close(fig)
    print(f"[plot ] {summary_path}")

    # Reconstruction image grid: one row per ckpt = [orig, L0, L01, L012] x K frames
    K = frames.shape[0]
    rows_with_samples = [r for r in rows if r["recon_samples"] is not None]
    if not rows_with_samples:
        return
    n = len(rows_with_samples)
    fig, axes = plt.subplots(
        n * 4, K, figsize=(K * 1.3, n * 4 * 1.3), squeeze=False,
    )

    def _to_img(t):
        # t in [-0.5, 0.5] -> uint8 [0, 255], (3, 64, 64) -> (64, 64, 3)
        arr = (t.clamp(-0.5, 0.5) + 0.5).cpu().numpy()
        arr = (arr * 255).astype(np.uint8)
        return np.transpose(arr, (1, 2, 0))

    orig_imgs = [_to_img(f.float() / 255.0 - 0.5) for f in frames]
    for ri, r in enumerate(rows_with_samples):
        per_level = r["recon_samples"]  # list of 3 tensors (K, 3, 64, 64)
        labels_row = ["orig"] + CASCADE
        for col in range(K):
            for rr, label in enumerate(labels_row):
                ax = axes[ri * 4 + rr, col]
                if label == "orig":
                    img = orig_imgs[col]
                else:
                    img = _to_img(per_level[rr - 1][col])
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    side = f"{r['label']}\n{label}" if rr == 0 else label
                    ax.set_ylabel(side, fontsize=8)

    fig.tight_layout()
    rec_path = os.path.join(output_dir, "mech_reconstructions.png")
    fig.savefig(rec_path, dpi=140)
    plt.close(fig)
    print(f"[plot ] {rec_path}")


def run_analysis(
    *,
    ckpts,
    source_ckpt=None,
    frame_env=None,
    frame_policy="random",
    frame_policy_ckpt=None,
    n_frames=32,
    seed=0,
    frame_seed=None,
    output_dir=None,
    tag=None,
    plot=False,
    plot_sample_frames=4,
    device="cpu",
    verbose=True,
):
    """Run the full mechanistic analysis programmatically.

    This is the importable entry point behind the CLI: a manifest-driven runner
    (scripts/run_diagnostics.py) calls this directly instead of shelling out, so
    every CLI flag has a keyword argument here with the same default.

    Returns a dict: {rows, pairwise, num_codes, frame_env, frame_seed,
    frame_sha1}. `rows` is one dict per checkpoint (label, meta, usage,
    residual_errors, recon, drift, indices); `pairwise` holds the per-level
    Jaccard / codebook-cosine matrices for every checkpoint pair.
    """
    if plot and output_dir is None:
        raise ValueError("plot=True requires output_dir to be set")
    if not ckpts:
        raise ValueError("ckpts must be a non-empty list of checkpoint paths")

    def _say(*a):
        if verbose:
            print(*a)

    torch.manual_seed(seed)
    frame_seed = frame_seed if frame_seed is not None else seed

    # 1. Build the "frame env" model and collect shared frames.
    frame_env = frame_env or infer_env_name(ckpts[0])
    frame_policy_ckpt = frame_policy_ckpt or ckpts[0]
    _say(f"== Build frame-collection env ({frame_env}) ==")
    frame_model = build_model(frame_env, device)
    load_weights(frame_model, frame_policy_ckpt, device)

    _say(f"\n== Collect {n_frames} frames (policy={frame_policy}, "
         f"policy_ckpt={os.path.basename(os.path.dirname(frame_policy_ckpt))}, "
         f"frame_seed={frame_seed}) ==")
    frames = collect_frames(
        frame_model, n_frames,
        frame_seed=frame_seed,
        policy=frame_policy,
        device=device,
    )
    frame_sha1 = _tensor_sha1(frames)
    _say(f"  frames: shape={tuple(frames.shape)}, dtype={frames.dtype}, sha1={frame_sha1[:12]}")
    num_codes = frame_model.encoder_network.hrvq.quantizers[0].num_codes

    # 1b. Optional source reference (e.g. Pong pretraining ckpt). We keep its
    # HRVQ submodule (codebook drift) and its pre-VQ encoder features on the
    # shared frames (rho_enc / D3) - then drop the rest of the model.
    source_hrvq = None
    source_pre_vq = None
    source_pos_dim = None
    if source_ckpt is not None:
        src_env = infer_env_name(source_ckpt)
        _say(f"\n== Build source reference ({src_env}) ==")
        src_model = build_model(src_env, device)
        load_weights(src_model, source_ckpt, device)
        source_hrvq = src_model.encoder_network.hrvq
        source_pre_vq = encode(src_model, frames)["pre_vq_features"].detach()
        source_pos_dim = src_model.encoder_network.position_dim
        del src_model

    # 2. For each ckpt: build a fresh model with the SAME env, load weights, encode + analyse.
    rows = []
    models = []
    for ckpt_path in ckpts:
        meta = parse_ckpt_dirname(ckpt_path)
        label = _label(meta, ckpt_path)

        # Reuse frame_model only when it's already the right weights;
        # otherwise build a new model (pairwise overlap below needs distinct
        # HRVQ submodules per ckpt, so we cannot just hot-swap weights into
        # frame_model).
        if ckpt_path == frame_policy_ckpt:
            model = frame_model
        else:
            model = build_model(frame_env, device)
            load_weights(model, ckpt_path, device)

        enc = encode(model, frames)
        usage = codebook_usage(enc["indices"], num_codes)
        if plot:
            recon, recon_samples = cascade_reconstruction(
                model, enc["x_target"], enc["z_q_levels_spatial"],
                return_samples=plot_sample_frames,
            )
        else:
            recon = cascade_reconstruction(model, enc["x_target"], enc["z_q_levels_spatial"])
            recon_samples = None
        drift = (
            codebook_drift_from_source(model.encoder_network.hrvq, source_hrvq)
            if source_hrvq is not None else None
        )
        enc_drift = (
            encoder_feature_drift(enc["pre_vq_features"], source_pre_vq, source_pos_dim)
            if source_pre_vq is not None else None
        )

        if verbose:
            _print_per_ckpt_block(
                label, n_frames * 16, num_codes,
                usage, enc["residual_errors"], recon, drift=drift,
                enc_drift=enc_drift,
            )
        rows.append({
            "label": label,
            "meta": meta,
            "indices": enc["indices"],
            "usage": usage,
            "residual_errors": enc["residual_errors"],
            "recon": recon,
            "drift": drift,
            "enc_drift": enc_drift,
            "recon_samples": recon_samples,
        })
        models.append(model)

    # 3. Cross-checkpoint summary table.
    if verbose:
        _print_summary_table(rows, num_codes, with_drift=source_hrvq is not None)

    # 4. Pairwise overlap matrices (only meaningful for 2+ ckpts).
    pairwise = _compute_pairwise(rows, models) if len(rows) >= 2 else []
    if verbose:
        _print_pairwise(rows, pairwise)

    # 5. Optional structured export for plotting / aggregation.
    if output_dir is not None:
        _write_outputs(
            output_dir,
            tag=tag,
            frame_env=frame_env,
            frame_seed=frame_seed,
            frame_policy=frame_policy,
            frame_policy_ckpt=frame_policy_ckpt,
            n_frames=n_frames,
            ckpts=list(ckpts),
            source_ckpt=source_ckpt,
            frame_sha1=frame_sha1,
            num_codes=num_codes,
            rows=rows,
            pairwise=pairwise,
        )

    # 6. Optional plot generation.
    if plot:
        _write_plots(
            output_dir,
            rows=rows,
            num_codes=num_codes,
            frames=frames[:plot_sample_frames],
            with_drift=source_hrvq is not None,
            with_pairwise=len(rows) >= 2,
        )

    return {
        "rows": rows,
        "pairwise": pairwise,
        "num_codes": num_codes,
        "frame_env": frame_env,
        "frame_seed": frame_seed,
        "frame_sha1": frame_sha1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="One or more checkpoint paths. The FIRST ckpt's env (or "
                         "--frame_env) is used to collect the shared frames.")
    ap.add_argument("--source_ckpt", default=None,
                    help="Optional reference ckpt (e.g. Pong source) for codebook "
                         "drift metric. Each --ckpts entry is compared level-by-level "
                         "to this source by matched index.")
    ap.add_argument("--frame_env", default=None,
                    help="e.g. atari100k-breakout. Auto-inferred from --ckpts[0] if omitted.")
    ap.add_argument("--frame_policy", choices=["random", "agent"], default="random",
                    help="'random' = uniform actions; 'agent' = rollout under "
                         "--frame_policy_ckpt's policy (in-distribution frames).")
    ap.add_argument("--frame_policy_ckpt", default=None,
                    help="Checkpoint whose policy drives frame collection when "
                         "--frame_policy=agent. Defaults to --ckpts[0].")
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="Torch RNG seed (init / non-frame ops)")
    ap.add_argument("--frame_seed", type=int, default=None,
                    help="Seed for frame collection (env + actions). Defaults to --seed.")
    ap.add_argument("--output_dir", default=None,
                    help="If set, writes results.json + summary.csv (long format, "
                         "one row per ckpt x level) for downstream aggregation.")
    ap.add_argument("--tag", default=None,
                    help="Optional label written into results.json (e.g. 'breakout_run1').")
    ap.add_argument("--plot", action="store_true",
                    help="Write mech_summary.png + mech_reconstructions.png to "
                         "--output_dir (requires --output_dir). Aggregates seeds "
                         "within each condition as mean +/- spread.")
    ap.add_argument("--plot_sample_frames", type=int, default=4,
                    help="How many frames to show in the reconstruction-grid plot.")
    args = ap.parse_args()
    if args.plot and args.output_dir is None:
        ap.error("--plot requires --output_dir")

    run_analysis(
        ckpts=args.ckpts,
        source_ckpt=args.source_ckpt,
        frame_env=args.frame_env,
        frame_policy=args.frame_policy,
        frame_policy_ckpt=args.frame_policy_ckpt,
        n_frames=args.n_frames,
        seed=args.seed,
        frame_seed=args.frame_seed,
        output_dir=args.output_dir,
        tag=args.tag,
        plot=args.plot,
        plot_sample_frames=args.plot_sample_frames,
    )


if __name__ == "__main__":
    main()
