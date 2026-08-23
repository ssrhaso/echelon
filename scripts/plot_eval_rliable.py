"""
Visualise eval results with TWISTER-style statistics (IQM + stratified
bootstrap CIs) using the `rliable` library from Agarwal et al. NeurIPS 2021.

Reads:  wandb_export/eval_epoch_long.csv
Writes: wandb_export/plots/learning_curves_iqm.png
        wandb_export/plots/final_iqm_bars.png
        wandb_export/plots/per_seed_strip.png
        wandb_export/plots/iqm_summary.csv

Why IQM (not mean):
  Mean is dominated by tail seeds (in our data, freeze-L0 mean=43.8 is
  driven by one seed at 154 of 10 seeds). IQM drops the top/bottom 25%
  of seeds and averages the middle, which TWISTER and modern Atari
  papers report as the headline statistic.

Human-normalised scoring uses Breakout values from the Atari100k
convention (Hafner DreamerV2 / Kaiser SimPLe):
  random = 1.7,  human = 30.5
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# rliable's bootstrap is broken against current arch on numpy 2 - but its
# point-estimate metrics (IQM, optimality gap) are pure numpy, so we reuse
# those and roll our own stratified bootstrap below.
from rliable import metrics

warnings.filterwarnings("ignore", category=FutureWarning)


def stratified_bootstrap_ci(
    score_dict, agg_func, reps=2000, ci=0.95, rng=None
):
    """Per-condition bootstrap: resample seeds with replacement, recompute the
    aggregate, take the (2.5, 97.5) percentile interval. `score_dict` maps
    condition -> ndarray of shape (n_seeds, ...) ; agg_func operates on the
    same shape and returns an ndarray of metric values.
    """
    rng = rng or np.random.RandomState(0)
    point, lo, hi = {}, {}, {}
    for cond, arr in score_dict.items():
        n = arr.shape[0]
        point[cond] = agg_func(arr)
        boots = np.stack([
            agg_func(arr[rng.randint(0, n, size=n)])
            for _ in range(reps)
        ])
        lo[cond] = np.percentile(boots, (1 - ci) / 2 * 100, axis=0)
        hi[cond] = np.percentile(boots, (1 + ci) / 2 * 100, axis=0)
    return point, lo, hi

ROOT       = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "wandb_export"
PLOT_DIR   = EXPORT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Atari100k Breakout reference scores
BREAKOUT_RANDOM = 1.7
BREAKOUT_HUMAN  = 30.5

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

def human_normalise(raw):
    return (raw - BREAKOUT_RANDOM) / (BREAKOUT_HUMAN - BREAKOUT_RANDOM)

# ---------------------------------------------------------------------------
# Load + tidy
# ---------------------------------------------------------------------------
long_df = pd.read_csv(EXPORT_DIR / "eval_epoch_long.csv")
score_df = long_df[long_df["metric"] == "score"].copy()

# Drop dead/empty runs (already filtered by long_df having values), then
# deduplicate (run_id, seed) -- keep the run with most eval points.
meta = pd.read_csv(EXPORT_DIR / "runs_metadata.csv")
keep_ids = (meta[meta["n_eval_points"] > 0]
            .sort_values("n_eval_points", ascending=False)
            .drop_duplicates(["family","condition","seed"])
            ["run_id"].tolist())
score_df = score_df[score_df["run_id"].isin(keep_ids)].copy()

# Bucket steps to 10k for clean alignment
score_df["step_bucket"] = (score_df["step"] / 10000).round().astype(int) * 10000
score_df = (score_df
            .groupby(["condition","seed","step_bucket"])["value"]
            .mean().reset_index())   # collapse any duplicates within bucket

step_grid = sorted(score_df["step_bucket"].unique())
step_grid = [s for s in step_grid if 10000 <= s <= 100000]

# Build {condition: ndarray[n_seeds, n_steps]} of human-normalised scores
score_arrays = {}
for cond in CONDITION_ORDER:
    sub = score_df[score_df["condition"] == cond]
    if sub.empty:
        continue
    seeds = sorted(sub["seed"].unique())
    arr = np.full((len(seeds), len(step_grid)), np.nan)
    for i, sd in enumerate(seeds):
        for j, st in enumerate(step_grid):
            row = sub[(sub["seed"] == sd) & (sub["step_bucket"] == st)]
            if len(row):
                arr[i, j] = row["value"].iloc[0]
    score_arrays[cond] = human_normalise(arr)
    print(f"{cond:<13} shape={arr.shape}  (n_seeds × n_step_buckets)")

# ---------------------------------------------------------------------------
# 1) Learning curves: IQM ± 95% stratified bootstrap CI per step
# ---------------------------------------------------------------------------
print("\n[1/3] Computing learning-curve IQM with bootstrap CIs ...")

RNG = np.random.RandomState(0)

def iqm_per_step(arr):
    """arr: (n_seeds, n_steps).  Returns (n_steps,) of IQM values, ignoring NaNs."""
    out = np.empty(arr.shape[1])
    for k in range(arr.shape[1]):
        col = arr[:, k]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            out[k] = np.nan
        else:
            out[k] = metrics.aggregate_iqm(col[:, None])
    return out

iqm_curve, iqm_lo, iqm_hi = stratified_bootstrap_ci(
    score_arrays, iqm_per_step, reps=2000, rng=RNG)

fig, ax = plt.subplots(figsize=(10, 6))
for cond, arr in score_arrays.items():
    y    = iqm_curve[cond]
    ylo  = iqm_lo[cond]
    yhi  = iqm_hi[cond]
    x    = np.array(step_grid)
    color = COLORS[cond]
    ax.plot(x, y, color=color, lw=2, label=f"{cond} (n={arr.shape[0]})")
    ax.fill_between(x, ylo, yhi, color=color, alpha=0.15, lw=0)

ax.axhline(1.0, ls="--", lw=0.8, color="black", alpha=0.5)
ax.text(101000, 1.02, "human", fontsize=8, color="black", alpha=0.7)
ax.set_xlabel("Atari100k env steps")
ax.set_ylabel("Human-normalised score (IQM, 95% bootstrap CI)")
ax.set_title("Breakout transfer (Pong → Breakout): learning curves\n"
             "IQM across seeds, 95% bootstrap CI",
             fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}k"))
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(PLOT_DIR / "learning_curves_iqm.png", dpi=150, bbox_inches="tight")
print(f"  wrote {PLOT_DIR / 'learning_curves_iqm.png'}")

# ---------------------------------------------------------------------------
# 2) Final-step bar chart: IQM ± 95% CI at the last fully-observed step
# ---------------------------------------------------------------------------
print("\n[2/3] Computing final-step IQM with bootstrap CIs ...")

final_scores = {c: arr[:, -1][~np.isnan(arr[:, -1])][:, None]
                for c, arr in score_arrays.items()}

def aggregate_func(x):
    """x: (n_seeds, 1).  Returns 4-vector: median, IQM, mean, optimality gap."""
    return np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x, 1.0),
    ])

agg_scores, agg_lo, agg_hi = stratified_bootstrap_ci(
    final_scores, aggregate_func, reps=5000, rng=RNG)

# Hand-rolled forest plot (one panel per metric)
metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)
conds = list(final_scores.keys())
for ax, m_idx, m_name in zip(axes, range(4), metric_names):
    ys = list(range(len(conds)))
    for y, cond in zip(ys, conds):
        c    = COLORS[cond]
        pt   = agg_scores[cond][m_idx]
        lo   = agg_lo[cond][m_idx]
        hi   = agg_hi[cond][m_idx]
        ax.plot([lo, hi], [y, y], color=c, lw=4, alpha=0.45)
        ax.scatter(pt, y, s=70, color=c, edgecolor="black", lw=0.8, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(conds)
    ax.invert_yaxis()
    ax.set_title(m_name, fontsize=10)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("Final performance @ 100k - point estimate + 95% bootstrap CI\n"
             "(human-normalised; 1.0 = human level)",
             fontsize=11)
fig.supxlabel("Human-normalised score", y=-0.02, fontsize=10)
fig.tight_layout()
fig.savefig(PLOT_DIR / "final_iqm_bars.png", dpi=150, bbox_inches="tight")
print(f"  wrote {PLOT_DIR / 'final_iqm_bars.png'}")

rows = []
for cond in final_scores:
    s = final_scores[cond].squeeze()
    rows.append({
        "condition":  cond,
        "n_seeds":    len(s),
        "median":     float(agg_scores[cond][0]),
        "median_lo":  float(agg_lo[cond][0]),
        "median_hi":  float(agg_hi[cond][0]),
        "iqm":        float(agg_scores[cond][1]),
        "iqm_lo":     float(agg_lo[cond][1]),
        "iqm_hi":     float(agg_hi[cond][1]),
        "mean":       float(agg_scores[cond][2]),
        "mean_lo":    float(agg_lo[cond][2]),
        "mean_hi":    float(agg_hi[cond][2]),
        "opt_gap":    float(agg_scores[cond][3]),
    })
agg_df = pd.DataFrame(rows).sort_values("iqm", ascending=False)
agg_df.to_csv(PLOT_DIR / "iqm_summary.csv", index=False)
print(f"  wrote {PLOT_DIR / 'iqm_summary.csv'}")
print("\n--- Final IQM table (sorted by IQM) ---")
print(agg_df.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 3) Per-seed strip plot: every seed's final score, with IQM range overlaid
# ---------------------------------------------------------------------------
print("\n[3/3] Drawing per-seed strip plot ...")

fig, ax = plt.subplots(figsize=(9, 5))
y_positions = {c: i for i, c in enumerate(score_arrays.keys())}

for cond, arr in score_arrays.items():
    finals = arr[:, -1][~np.isnan(arr[:, -1])]
    y = y_positions[cond]
    jitter = (np.random.RandomState(42).rand(len(finals)) - 0.5) * 0.25
    ax.scatter(finals, np.full_like(finals, y) + jitter,
               s=40, alpha=0.7, color=COLORS[cond], edgecolor="black", lw=0.4)
    # IQM
    iqm_v = metrics.aggregate_iqm(finals[:, None])
    ax.scatter(iqm_v, y, marker="D", s=110, color=COLORS[cond],
               edgecolor="black", lw=1.2, zorder=5,
               label=f"{cond} IQM" if cond == list(score_arrays.keys())[0] else None)

ax.axvline(1.0, ls="--", lw=0.8, color="black", alpha=0.5)
ax.text(1.01, len(y_positions) - 0.4, "human", fontsize=8, alpha=0.6)
ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(list(y_positions.keys()))
ax.invert_yaxis()
ax.set_xlabel("Human-normalised score at 100k steps")
ax.set_title("Per-seed final scores (dots) with IQM (diamond)\n"
             "shows how much one outlier seed shifts the picture",
             fontsize=11)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_DIR / "per_seed_strip.png", dpi=150, bbox_inches="tight")
print(f"  wrote {PLOT_DIR / 'per_seed_strip.png'}")

print("\nDone.")
