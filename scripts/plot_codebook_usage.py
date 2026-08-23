"""Per-level codebook-usage plots from the transfer diagnostics.

Two figures, both straight from results/diagnostics_raw.csv (no checkpoints):

  jaccard_per_level.{png,pdf}
      Grouped bars of Jaccard overlap J^(l) with the Pong source, one panel per
      game, bars grouped by RVQ level (L0/L1/L2) within each condition. Makes the
      coarse-to-fine collapse visible: under the fully-frozen conditions the bars
      fall J0 > J1 > J2, while adaptable codebooks sit at a flat ~0.15-0.23.

  active_codes_per_level.{png,pdf}
      The same layout for the active-code count |A^(l)|, the literal companion to
      the perplexity already reported in the main table.

Bars are means over the 3 retained seeds; error bars are std.

Usage:
  python scripts/plot_codebook_usage.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "diagnostics_raw.csv"
OUT_DIR = ROOT / "results" / "figures"

ORDER = ["scratch", "warmstart", "freeze-L0", "freeze-L01", "freeze-L012", "freeze-enc", "freeze-all"]
GAME_TITLE = {"demon_attack": "Demon Attack", "ms_pacman": "Ms Pac-Man", "up_n_down": "Up'n'Down"}
GAMES = ["demon_attack", "ms_pacman", "up_n_down"]
LEVEL_COLORS = ["#3b6ea5", "#e08214", "#5aae61"]  # L0, L1, L2


def _panel(ax, df, game, prefix, ylabel):
    g = df[df.game == game]
    conds = [c for c in ORDER if c in set(g.condition)]
    x = np.arange(len(conds))
    w = 0.26
    for l in range(3):
        means = [g[g.condition == c][f"{prefix}{l}"].mean() for c in conds]
        stds = [g[g.condition == c][f"{prefix}{l}"].std(ddof=0) for c in conds]
        ax.bar(x + (l - 1) * w, means, w, yerr=stds, capsize=2,
               color=LEVEL_COLORS[l], label=f"L{l}", error_kw={"elinewidth": 0.7})
    ax.set_title(GAME_TITLE[game], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("freeze-", "fz-") for c in conds], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)


def make_figure(df, prefix, ylabel, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(GAMES), figsize=(4.2 * len(GAMES), 3.2), sharey=True)
    for ax, game in zip(axes, GAMES):
        _panel(ax, df, game, prefix, ylabel)
    axes[0].legend(title="RVQ level", fontsize=8, title_fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{fname}.{ext}"
        fig.savefig(p, dpi=150)
        try:
            print(f"[ok] {p.relative_to(ROOT)}")
        except ValueError:  # output dir outside the repo (e.g. under test)
            print(f"[ok] {p}")
    plt.close(fig)


def main():
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} - run compute_diagnostics.py first")
    df = pd.read_csv(RAW)
    make_figure(df, "J", r"Jaccard $J^{(\ell)}$ with Pong", "jaccard_per_level")
    make_figure(df, "active", r"Active codes $|A^{(\ell)}|$", "active_codes_per_level")


if __name__ == "__main__":
    main()
