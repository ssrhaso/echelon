"""
Aggregate per-seed diagnostics (results/diagnostics_raw.csv) to mean +/- std per
(game, condition), ordered by the freeze ladder.

Beyond the raw D1/D2/D3 metrics this derives the two scale-free quantities the
raw ones cannot be read without:

  Size-matched Jaccard null (D1).  J^(l) falls with |A_tgt| even under pure
  chance overlap, and |A_tgt| ranges over ~50-350 across the freeze ladder, so
  raw J^(l) confounds "how much code reuse" with "how many codes are active".
  Under the exchangeability null "A_tgt is a uniformly random |A_tgt|-subset of
  the K codes", the intersection is Hypergeometric(K, |A_src|, |A_tgt|), giving

      Jnull^(l) = m / (S + T - m),      m = S*T/K
      Jratio^(l) = J^(l) / Jnull^(l)    (1 = chance, >1 = reuse, <1 = avoidance)
      Jz^(l)    = (inter - m) / sd(inter)

  This is a null on *set size only*: it does not assume uniform code usage, it
  asks whether the observed overlap exceeds what a same-sized arbitrary code set
  would achieve. Jratio/Jz are what the coarse-to-fine reuse claim needs.

  Normalised quantisation error (D2).  E^(l) is an unnormalised MSE in each
  checkpoint's own pre-VQ feature space; two encoders differing only by a scale
  on z give E^(l) differing by its square, so cross-condition comparison of raw
  E^(l) is not meaningful. Since E^(l) = E||r^(l+1)||^2, the residual energies
  are R = (zenergy, E0, E1, E2) and we report

      Erel^(l)     = E^(l) / R^(0)        (feature energy left after level l)
      contract^(l) = R^(l+1) / R^(l)      (fraction of its own input a level
                                           fails to absorb; 0 = perfect)

Derived metrics are computed per seed and then averaged, not computed from the
per-condition means.

Emits:
  - results/diagnostics_agg.csv   (tidy mean/std table, raw + derived)
  - stdout summary per game

Usage:
  python scripts/aggregate_diagnostics.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "diagnostics_raw.csv"
OUT = ROOT / "results" / "diagnostics_agg.csv"

K = 512
NLEVELS = 3
ORDER = ["scratch", "warmstart", "freeze-L0", "freeze-L01", "freeze-L012", "freeze-enc", "freeze-all"]
BASE_METRICS = ["J0", "J1", "J2", "E0", "E1", "E2", "rho_enc",
                "perp0", "perp1", "perp2", "active0", "active1", "active2"]
DERIVED = ([f"Jnull{l}" for l in range(NLEVELS)] + [f"Jratio{l}" for l in range(NLEVELS)]
           + [f"Jz{l}" for l in range(NLEVELS)] + [f"ovl{l}" for l in range(NLEVELS)]
           + [f"Erel{l}" for l in range(NLEVELS)] + [f"contract{l}" for l in range(NLEVELS)]
           + ["zenergy"])

NEEDED = [f"srcactive{l}" for l in range(NLEVELS)] + [f"inter{l}" for l in range(NLEVELS)] + ["zenergy"]


def add_derived(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Attach size-matched-null and scale-free columns. Returns (df, ok)."""
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        print(f"[warn] {RAW.name} lacks {missing}; re-run scripts/compute_diagnostics.py "
              f"to enable the size-matched null and normalised error. "
              f"Emitting raw metrics only.")
        return df, False

    for l in range(NLEVELS):
        S = df[f"srcactive{l}"].astype(float)
        T = df[f"active{l}"].astype(float)
        inter = df[f"inter{l}"].astype(float)

        # Hypergeometric(K, S, T) moments for the intersection under the
        # size-matched exchangeability null.
        m = S * T / K
        var = S * T * (K - S) * (K - T) / (K ** 2 * (K - 1))
        df[f"Jnull{l}"] = m / (S + T - m)
        df[f"Jratio{l}"] = df[f"J{l}"] / df[f"Jnull{l}"]
        df[f"Jz{l}"] = (inter - m) / np.sqrt(var)
        # Size-robust alternative to Jaccard: fraction of the smaller set covered.
        df[f"ovl{l}"] = inter / np.minimum(S, T)

    # Residual energies R^(0..3) = (zenergy, E0, E1, E2); E^(l) = E||r^(l+1)||^2.
    R = [df["zenergy"].astype(float)] + [df[f"E{l}"].astype(float) for l in range(NLEVELS)]
    for l in range(NLEVELS):
        df[f"Erel{l}"] = R[l + 1] / R[0]
        df[f"contract{l}"] = R[l + 1] / R[l]
    return df, True


def main():
    df = pd.read_csv(RAW)
    df, has_derived = add_derived(df)
    metrics = BASE_METRICS + (DERIVED if has_derived else [])

    rows = []
    for (game, cond), g in df.groupby(["game", "condition"]):
        row = {"game": game, "condition": cond, "n": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=0)
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg["cond_order"] = agg.condition.map({c: i for i, c in enumerate(ORDER)})
    agg = agg.sort_values(["game", "cond_order"]).drop(columns="cond_order")
    agg.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    for game, g in agg.groupby("game"):
        print(f"\n=== {game} ===")
        cols = ["condition", "n", "active0_mean", "active1_mean", "active2_mean",
                "J0_mean", "J1_mean", "J2_mean"]
        if has_derived:
            cols += ["Jratio0_mean", "Jratio1_mean", "Jratio2_mean",
                     "Jz0_mean", "Jz1_mean", "Jz2_mean",
                     "E0_mean", "Erel0_mean", "Erel1_mean", "Erel2_mean"]
        else:
            cols += ["E0_mean", "E1_mean", "E2_mean"]
        cols += ["rho_enc_mean"]
        show = g[cols].copy()
        for c in show.columns:
            if c.startswith(("J0", "J1", "J2", "rho", "Erel", "contract")):
                show[c] = show[c].round(3)
            elif c.startswith(("Jratio", "Jz")):
                show[c] = show[c].round(2)
            elif c.startswith(("E", "active", "perp")):
                show[c] = show[c].round(1)
        print(show.to_string(index=False))

    if has_derived:
        print("\nJratio: 1.0 = chance overlap for that active-set size; "
              ">1 = code reuse; <1 = target avoids the source's cells.")
        print("Erel:   fraction of pre-VQ feature energy still unquantised after level l.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
