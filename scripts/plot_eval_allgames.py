"""
Multi-game eval plots for the ECHELON transfer study.

Reads:  wandb_export/eval_epoch_long.csv  (+ runs_metadata.csv for dedup)
Writes (under wandb_export/plots/):
  learning_curves_per_game.png   2x3 grid, one panel per game, raw score,
                                  mean +/- 95% bootstrap CI band + faint seed lines.
  aggregate_iqm.png              human-normalised IQM / Median / Mean / Optimality-Gap
                                  forest plot, pooled across the 4 full-matrix games.
  aggregate_learning_curve.png   human-normalised IQM curve over env steps (pooled).
  per_game_final_stats.csv       per (game, condition): n, median, IQR, mean, 95% CI.
  aggregate_iqm_summary.csv      per condition: pooled IQM/median/mean/opt-gap + 95% CIs.

Stats notes
-----------
* Per-game curves use the MEAN across seeds with a stratified (resample-seeds)
  95% bootstrap CI band. Transfer conditions have only n=3 seeds in the three
  newer games, so faint individual-seed lines are drawn to keep the n honest;
  IQM is reserved for the pooled aggregate where n is large enough to trim.
* The cross-game aggregate human-normalises each game, then for each condition
  pools final scores across games and bootstraps by resampling seeds WITHIN
  each game (stratified bootstrap, Agarwal et al. NeurIPS 2021).
* Human/random reference scores: Atari100k convention (Kaiser et al., SimPLe
  2020 Table), as used by DreamerV2/V3, EfficientZero, TWISTER.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import trim_mean

ROOT       = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "wandb_export"
PLOT_DIR   = EXPORT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Atari100k random / human reference scores (Kaiser et al. 2020, SimPLe Table).
REF = {
    "pong":         (-20.7, 14.6),
    "breakout":     (1.7,   30.5),
    "demon_attack": (152.1, 1971.0),
    "freeway":      (0.0,   29.6),
    "ms_pacman":    (307.3, 6951.6),
    "up_n_down":    (533.4, 11693.2),
}

# Display order / labels for the per-game grid (all six games).
GAME_GRID = ["breakout", "demon_attack", "ms_pacman",
             "up_n_down", "pong", "freeway"]
GAME_LABEL = {
    "breakout": "Breakout", "demon_attack": "Demon Attack",
    "ms_pacman": "Ms. Pac-Man", "up_n_down": "Up'n'Down",
    "pong": "Pong (source)", "freeway": "Freeway (boundary)",
}

# Games that have the full 7-condition matrix -> used for the pooled aggregate.
AGG_GAMES = ["breakout", "demon_attack", "ms_pacman", "up_n_down"]

CONDITION_ORDER = [
    "scratch", "warmstart", "freeze-enc",
    "freeze-L0", "freeze-L01", "freeze-L012", "freeze-all",
]
COLORS = {
    "scratch":     "#888888",
    "warmstart":   "#1f77b4",
    "freeze-enc":  "#9467bd",
    "freeze-L0":   "#2ca02c",
    "freeze-L01":  "#17becf",
    "freeze-L012": "#bcbd22",
    "freeze-all":  "#d62728",
}

STEP_GRID = list(range(10000, 100001, 10000))   # 10k .. 100k env steps


def iqm(x):
    """Interquartile mean = 25%-trimmed mean."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    if len(x) < 4:           # trimming removes everything; fall back to mean
        return float(np.mean(x))
    return float(trim_mean(x, 0.25))


def human_normalise(raw, game):
    r, h = REF[game]
    return (raw - r) / (h - r)


# ---------------------------------------------------------------------------
# Load + dedup + bucket to a clean 10k step grid
# ---------------------------------------------------------------------------
long_df  = pd.read_csv(EXPORT_DIR / "eval_epoch_long.csv")
meta     = pd.read_csv(EXPORT_DIR / "runs_metadata.csv")
score_df = long_df[long_df["metric"] == "score"].copy()

# keep one run per (game, condition, seed): the one with the most eval points
keep_ids = (meta[meta["n_eval_points"] > 0]
            .sort_values("n_eval_points", ascending=False)
            .drop_duplicates(["game", "condition", "seed"])["run_id"].tolist())
score_df = score_df[score_df["run_id"].isin(keep_ids)].copy()

score_df["step_bucket"] = (score_df["step"] / 10000).round().astype(int) * 10000
score_df = score_df[score_df["step_bucket"].between(10000, 100000)]
score_df = (score_df
            .groupby(["game", "condition", "seed", "step_bucket"])["value"]
            .mean().reset_index())


def seed_step_array(game, cond, normalise=False):
    """-> ndarray (n_seeds, n_steps) on STEP_GRID, NaN where missing."""
    sub = score_df[(score_df["game"] == game) & (score_df["condition"] == cond)]
    if sub.empty:
        return None, []
    seeds = sorted(sub["seed"].unique())
    arr = np.full((len(seeds), len(STEP_GRID)), np.nan)
    for i, sd in enumerate(seeds):
        for j, st in enumerate(STEP_GRID):
            row = sub[(sub["seed"] == sd) & (sub["step_bucket"] == st)]
            if len(row):
                arr[i, j] = row["value"].iloc[0]
    if normalise:
        arr = human_normalise(arr, game)
    return arr, seeds


def boot_band(arr, reps=2000, seed=0):
    """Per-step mean + 95% CI by resampling seeds (rows) with replacement."""
    rng = np.random.RandomState(seed)
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)
    boots = np.empty((reps, arr.shape[1]))
    for b in range(reps):
        boots[b] = np.nanmean(arr[rng.randint(0, n, size=n)], axis=0)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return mean, lo, hi


# ---------------------------------------------------------------------------
# 1) Per-game learning-curve grid
# ---------------------------------------------------------------------------
print("[1/3] per-game learning curves ...", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
final_rows = []

for ax, game in zip(axes.flat, GAME_GRID):
    present = [c for c in CONDITION_ORDER
               if not score_df[(score_df.game == game) &
                               (score_df.condition == c)].empty]
    for cond in present:
        arr, seeds = seed_step_array(game, cond)
        if arr is None:
            continue
        mean, lo, hi = boot_band(arr)
        x = np.array(STEP_GRID)
        c = COLORS[cond]
        # faint individual seeds
        for row in arr:
            ax.plot(x, row, color=c, lw=0.5, alpha=0.18)
        ax.plot(x, mean, color=c, lw=2.0, label=f"{cond} (n={arr.shape[0]})")
        if arr.shape[0] >= 3:
            ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0)

        # final (last-5 / available) per-seed score for the stats table
        finals = np.nanmean(arr[:, -5:], axis=1)
        finals = finals[~np.isnan(finals)]
        final_rows.append({
            "game": game, "condition": cond, "n_seeds": int(len(finals)),
            "final_mean": float(np.mean(finals)),
            "final_median": float(np.median(finals)),
            "final_q25": float(np.percentile(finals, 25)),
            "final_q75": float(np.percentile(finals, 75)),
            "final_min": float(np.min(finals)), "final_max": float(np.max(finals)),
        })

    r, h = REF[game]
    ax.axhline(h, ls="--", lw=0.8, color="black", alpha=0.45)
    ax.text(101000, h, " human", fontsize=7, color="black", alpha=0.6, va="center")
    ax.set_title(GAME_LABEL[game], fontsize=11)
    ax.set_xlabel("Atari100k env steps")
    ax.set_ylabel("Eval score (raw)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=1)

fig.suptitle("Pong-pretrained HRVQ transfer: per-game learning curves\n"
             "bold = mean over seeds, band = 95% bootstrap CI, faint = individual seeds",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(PLOT_DIR / "learning_curves_per_game.png", dpi=150, bbox_inches="tight")
print(f"  wrote {PLOT_DIR / 'learning_curves_per_game.png'}")

pd.DataFrame(final_rows).to_csv(PLOT_DIR / "per_game_final_stats.csv", index=False)
print(f"  wrote {PLOT_DIR / 'per_game_final_stats.csv'}")


# ---------------------------------------------------------------------------
# Build pooled normalised final-score dict for the aggregate
#   agg_final[cond] = list of (game, ndarray of per-seed normalised finals)
#   agg_curve[cond] = list of (game, ndarray (n_seeds, n_steps) normalised)
# ---------------------------------------------------------------------------
agg_final, agg_curve = {}, {}
for cond in CONDITION_ORDER:
    fins, curves = [], []
    for game in AGG_GAMES:
        arr, _ = seed_step_array(game, cond, normalise=True)
        if arr is None:
            continue
        finals = np.nanmean(arr[:, -5:], axis=1)
        finals = finals[~np.isnan(finals)]
        if len(finals):
            fins.append((game, finals))
        curves.append((game, arr))
    if fins:
        agg_final[cond] = fins
        agg_curve[cond] = curves


def stratified_boot(per_game_arrays, agg_func, reps=5000, seed=0):
    """per_game_arrays: list of (game, ndarray) where ndarray rows are seeds.
    Resample seeds within each game, pool rows, apply agg_func. Returns
    (point, lo, hi). agg_func maps a pooled ndarray -> scalar or vector."""
    rng = np.random.RandomState(seed)
    pooled = np.concatenate([a for _, a in per_game_arrays], axis=0)
    point = agg_func(pooled)
    boots = []
    for _ in range(reps):
        parts = []
        for _, a in per_game_arrays:
            n = a.shape[0]
            parts.append(a[rng.randint(0, n, size=n)])
        boots.append(agg_func(np.concatenate(parts, axis=0)))
    boots = np.stack(boots)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return point, lo, hi


# ---------------------------------------------------------------------------
# 2) Aggregate forest plot: Median / IQM / Mean / Optimality-Gap
# ---------------------------------------------------------------------------
print("[2/3] cross-game aggregate forest ...", flush=True)


def quad_metrics(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.array([
        np.median(x),
        iqm(x),
        np.mean(x),
        np.mean(np.maximum(0.0, 1.0 - x)),   # optimality gap vs human=1.0
    ])


metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]
conds_present = [c for c in CONDITION_ORDER if c in agg_final]
pt, lo, hi = {}, {}, {}
for cond in conds_present:
    arrs = [(g, v.reshape(-1, 1)) for g, v in agg_final[cond]]
    p, l, h = stratified_boot(arrs, lambda a: quad_metrics(a[:, 0]))
    pt[cond], lo[cond], hi[cond] = p, l, h

fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True)
for ax, m_idx, m_name in zip(axes, range(4), metric_names):
    ys = list(range(len(conds_present)))
    for y, cond in zip(ys, conds_present):
        c = COLORS[cond]
        ax.plot([lo[cond][m_idx], hi[cond][m_idx]], [y, y], color=c, lw=4, alpha=0.45)
        ax.scatter(pt[cond][m_idx], y, s=80, color=c, edgecolor="black", lw=0.8, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels(conds_present)
    ax.invert_yaxis(); ax.set_title(m_name, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    if m_name != "Optimality Gap":
        ax.axvline(1.0, ls="--", lw=0.7, color="black", alpha=0.4)

n_seeds_total = {c: sum(len(v) for _, v in agg_final[c]) for c in conds_present}
fig.suptitle("Cross-game transfer aggregate (Breakout, Demon Attack, Ms. Pac-Man, Up'n'Down)\n"
             "human-normalised final score, point estimate + 95% stratified bootstrap CI "
             "(1.0 = human; pong/freeway excluded - see notes)",
             fontsize=12)
fig.supxlabel("Human-normalised score", y=-0.02, fontsize=10)
fig.tight_layout()
fig.savefig(PLOT_DIR / "aggregate_iqm.png", dpi=150, bbox_inches="tight")
print(f"  wrote {PLOT_DIR / 'aggregate_iqm.png'}")

rows = []
for cond in conds_present:
    rows.append({
        "condition": cond, "n_seeds_pooled": n_seeds_total[cond],
        "n_games": len(agg_final[cond]),
        "median": pt[cond][0], "median_lo": lo[cond][0], "median_hi": hi[cond][0],
        "iqm": pt[cond][1], "iqm_lo": lo[cond][1], "iqm_hi": hi[cond][1],
        "mean": pt[cond][2], "mean_lo": lo[cond][2], "mean_hi": hi[cond][2],
        "opt_gap": pt[cond][3], "opt_gap_lo": lo[cond][3], "opt_gap_hi": hi[cond][3],
    })
agg_df = pd.DataFrame(rows).sort_values("iqm", ascending=False)
agg_df.to_csv(PLOT_DIR / "aggregate_iqm_summary.csv", index=False)
print(f"  wrote {PLOT_DIR / 'aggregate_iqm_summary.csv'}")
print("\n--- aggregate IQM table (sorted) ---")
print(agg_df.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 3) Aggregate normalised IQM learning curve
# ---------------------------------------------------------------------------
print("\n[3/3] aggregate normalised learning curve ...", flush=True)


def iqm_per_step(pooled):           # pooled: (N, n_steps)
    return np.array([iqm(pooled[:, k]) for k in range(pooled.shape[1])])


fig, ax = plt.subplots(figsize=(10, 6))
for cond in conds_present:
    p, l, h = stratified_boot(agg_curve[cond], iqm_per_step, reps=2000)
    x = np.array(STEP_GRID)
    c = COLORS[cond]
    n = sum(a.shape[0] for _, a in agg_curve[cond])
    ax.plot(x, p, color=c, lw=2, label=f"{cond} (n={n})")
    ax.fill_between(x, l, h, color=c, alpha=0.13, lw=0)

ax.axhline(1.0, ls="--", lw=0.8, color="black", alpha=0.5)
ax.text(101000, 1.0, " human", fontsize=8, alpha=0.7, va="center")
ax.set_xlabel("Atari100k env steps")
ax.set_ylabel("Human-normalised score (IQM, 95% bootstrap CI)")
ax.set_title("Cross-game transfer: aggregate normalised learning curves\n"
             "IQM over 4 games x seeds, 95% stratified bootstrap CI", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(PLOT_DIR / "aggregate_learning_curve.png", dpi=150, bbox_inches="tight")
print(f"  wrote {PLOT_DIR / 'aggregate_learning_curve.png'}")
print("\nDone.")
