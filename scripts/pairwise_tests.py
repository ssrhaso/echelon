"""
Pairwise statistical tests on final-step seed scores.

For each ordered pair (A, B) of conditions, compute:
 - Probability of improvement P(A > B): the rliable / Agarwal-2021 metric.
    Equivalent to a normalised Mann-Whitney U statistic. For two seed
    samples X, Y of size n_x, n_y:
        P(X > Y) = (#{(i,j): x_i > y_j} + 0.5 #{ties}) / (n_x * n_y)
    This is rank-based, robust to outliers, doesn't assume normality.
 - 95% bootstrap CI on P(A > B) by resampling seeds with replacement.
 - Welch's t-test (handles unequal variance, unequal n).
 - Cohen's d effect size (standardised mean difference).
 - Power analysis: given observed effect size, what n per group is
    needed for 80% power at alpha=0.05?
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT       = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "wandb_export"
PLOT_DIR   = EXPORT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

CONDITION_ORDER = [
    "scratch", "warmstart", "freeze-enc",
    "freeze-L0", "freeze-L01", "freeze-L012", "freeze-all",
]

# Load + dedup like before
long_df = pd.read_csv(EXPORT_DIR / "eval_epoch_long.csv")
meta    = pd.read_csv(EXPORT_DIR / "runs_metadata.csv")

keep_ids = (meta[meta["n_eval_points"] > 0]
            .sort_values("n_eval_points", ascending=False)
            .drop_duplicates(["family","condition","seed"])
            ["run_id"].tolist())
score_df = long_df[(long_df["metric"] == "score") & long_df["run_id"].isin(keep_ids)].copy()
score_df["step_bucket"] = (score_df["step"] / 10000).round().astype(int) * 10000

# Final-step (100k) score per (condition, seed)
finals = (score_df[score_df["step_bucket"] == 100000]
          .groupby(["condition","seed"])["value"].mean().reset_index())

samples = {c: finals[finals["condition"] == c]["value"].to_numpy()
           for c in CONDITION_ORDER if len(finals[finals["condition"] == c])}

print("Final-step samples (raw Breakout scores):")
for c, s in samples.items():
    print(f"  {c:<13}  n={len(s):>2}  mean={s.mean():6.2f}  std={s.std(ddof=1):6.2f}  "
          f"values={np.round(np.sort(s), 1).tolist()}")

# ---------------------------------------------------------------------------
def prob_improvement(x, y):
    """P(X > Y) with 0.5 weight on ties (rank-based, outlier-robust)."""
    nx, ny = len(x), len(y)
    cmp = (x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()
    return cmp / (nx * ny)

def boot_ci(x, y, fn, reps=5000, ci=0.95, rng=None):
    rng = rng or np.random.RandomState(0)
    nx, ny = len(x), len(y)
    vals = np.empty(reps)
    for r in range(reps):
        ix = rng.randint(0, nx, size=nx)
        iy = rng.randint(0, ny, size=ny)
        vals[r] = fn(x[ix], y[iy])
    return (np.percentile(vals, (1-ci)/2 * 100),
            np.percentile(vals, (1+ci)/2 * 100))

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sx, sy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx-1)*sx + (ny-1)*sy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled

def n_for_power(d, power=0.80, alpha=0.05):
    """Approx n per group for two-sided two-sample t-test."""
    if abs(d) < 1e-6:
        return float("inf")
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha/2)
    z_b = norm.ppf(power)
    return ((z_a + z_b)**2 * 2) / (d**2)

# ---------------------------------------------------------------------------
RNG = np.random.RandomState(0)
conds = list(samples.keys())
rows = []
P = np.full((len(conds), len(conds)), np.nan)

for i, A in enumerate(conds):
    for j, B in enumerate(conds):
        if A == B:
            P[i, j] = 0.5
            continue
        x, y = samples[A], samples[B]
        p   = prob_improvement(x, y)
        plo, phi = boot_ci(x, y, prob_improvement, reps=2000, rng=RNG)
        d   = cohens_d(x, y)
        t, pval = stats.ttest_ind(x, y, equal_var=False)
        n_needed = n_for_power(d)
        P[i, j] = p
        if i < j:  # report each unordered pair once
            rows.append({
                "A": A, "B": B,
                "n_A": len(x), "n_B": len(y),
                "mean_A": x.mean(), "mean_B": y.mean(),
                "P(A>B)": p, "P_lo": plo, "P_hi": phi,
                "cohens_d": d,
                "welch_t": t, "welch_p": pval,
                "n_needed_per_group_80pct_power": n_needed,
                "significant_p<0.05": pval < 0.05,
                "P_CI_excludes_0.5": (plo > 0.5) or (phi < 0.5),
            })

df = pd.DataFrame(rows)
df.to_csv(PLOT_DIR / "pairwise_tests.csv", index=False)

print("\n" + "="*80)
print("PAIRWISE TESTS at step=100k (sorted by |P(A>B) - 0.5|, biggest gaps first)")
print("="*80)
df_sorted = df.assign(strength=lambda d: (d["P(A>B)"] - 0.5).abs()).sort_values("strength", ascending=False).drop(columns="strength")
cols = ["A","B","n_A","n_B","mean_A","mean_B","P(A>B)","P_lo","P_hi",
        "cohens_d","welch_p","P_CI_excludes_0.5","n_needed_per_group_80pct_power"]
print(df_sorted[cols].round(3).to_string(index=False))

print("\n" + "="*80)
print("STATISTICALLY SIGNIFICANT PAIRS (Welch p<0.05 OR P_CI excludes 0.5)")
print("="*80)
sig = df[(df["welch_p"] < 0.05) | (df["P_CI_excludes_0.5"])]
if len(sig):
    print(sig[cols].round(3).to_string(index=False))
else:
    print("  (none - no pair separates at 95% confidence)")

# ---------------------------------------------------------------------------
# Heatmap of P(row > col)
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(P, cmap="RdBu_r", vmin=0, vmax=1)
ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=45, ha="right")
ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds)
for i in range(len(conds)):
    for j in range(len(conds)):
        v = P[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=9, color="white" if abs(v-0.5)>0.25 else "black")
plt.colorbar(im, ax=ax, label="P(row > column)")
ax.set_title("Probability that row condition beats column condition\n"
             "(at step=100k, raw Breakout scores)\n"
             "0.5 = tie; > 0.5 means row wins more often",
             fontsize=10)
fig.tight_layout()
fig.savefig(PLOT_DIR / "pairwise_prob_heatmap.png", dpi=150, bbox_inches="tight")
print(f"\nwrote {PLOT_DIR / 'pairwise_prob_heatmap.png'}")
print(f"wrote {PLOT_DIR / 'pairwise_tests.csv'}")
