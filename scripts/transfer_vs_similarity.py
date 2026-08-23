"""Does the cost of a frozen codebook track visual similarity to the source?

Sec. 4.1 sets up a selection design whose whole purpose is to dissociate visual
similarity S_vis from control similarity S_phys, on the argument that HRVQ
quantises per-frame CNN features with no action or temporal conditioning, so
S_vis is the *only* axis along which codebook reuse can mechanistically help.
That is a directional prediction fixed before any transfer run: freeze cost
should track S_vis and be independent of S_phys. This script runs it.

Freeze cost is measured in human-normalised units so it is comparable across
games with wildly different score scales:

    delta(cond, game) = HNS(cond, game) - HNS(from-scratch, game)

with HNS = (score - random)/(human - random) computed on each seed's near-final
return, then averaged. Seed-level bootstrap gives a CI on each delta.

With only four target games this is a *directional* check on a pre-registered
prediction, not an inferential test; the correlations are reported with that
caveat and the per-game ordering is printed so the reader can see the raw
pattern rather than trusting a coefficient computed on n=4.

Usage:
  python scripts/transfer_vs_similarity.py

Source of truth for returns is wandb_export/eval_epoch_long.csv, the per-epoch
eval export. It reproduces every cell of the paper's Table 8 to rounding and
carries the full seed counts. Do NOT read returns from
results/paper_aggregate.json: that file is a stale mid-flight snapshot which is
short several seeds, and the W&B history behind it has since been cleaned up, so
it cannot be refreshed.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "wandb_export" / "eval_epoch_long.csv"
SEL = ROOT / "priori" / "selection_table.csv"
OUT = ROOT / "results" / "transfer_vs_similarity.csv"

# (random, human) Atari 100k reference scores, from the IRIS Table 1 column
# (Micheli et al., ICLR 2023), the same source the appendix suite table uses.
REF = {
    "breakout":     (1.7, 30.5),
    "demon_attack": (152.1, 1971.0),
    "ms_pacman":    (307.3, 6951.6),
    "up_n_down":    (533.4, 11693.2),
}
TITLE = {"breakout": "Breakout", "demon_attack": "Demon Attack",
         "ms_pacman": "Ms Pac-Man", "up_n_down": "Up'n'Down"}
SEL_NAME = {"breakout": "Breakout", "demon_attack": "DemonAttack",
            "ms_pacman": "MsPacman", "up_n_down": "UpNDown"}
CONDS = ["warmstart", "freeze-L0", "freeze-L01", "freeze-L012", "freeze-enc", "freeze-all"]
GAMES = ["breakout", "demon_attack", "ms_pacman", "up_n_down"]
NBOOT = 10000
RNG = np.random.default_rng(0)


def load_near_final():
    """Per-seed near-final return: the mean of each seed's last 5 eval points."""
    d = pd.read_csv(EVAL)
    d = d[d.metric == "score"]
    return (d.sort_values("epoch_idx")
             .groupby(["game", "condition", "seed"])["value"]
             .apply(lambda s: s.tail(5).mean())
             .rename("last5").reset_index())


def seed_scores(ns, game, cond):
    """Near-final returns for one (game, condition) cell, or None if absent."""
    v = ns[(ns.game == game) & (ns.condition == cond)]["last5"].to_numpy(float)
    return v if len(v) else None


def hns(x, game):
    r, h = REF[game]
    return (x - r) / (h - r)


def boot_delta(a, b, game):
    """Bootstrap CI on HNS(a) - HNS(b) resampling seeds independently."""
    da = hns(RNG.choice(a, (NBOOT, len(a)), replace=True).mean(1), game)
    db = hns(RNG.choice(b, (NBOOT, len(b)), replace=True).mean(1), game)
    d = da - db
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ns = load_near_final()
    sel = pd.read_csv(SEL).set_index("game")

    rows = []
    for game in GAMES:
        scratch = seed_scores(ns, game, "scratch")
        if scratch is None:
            print(f"[skip] {game}: no from-scratch seeds")
            continue
        svis = float(sel.loc[SEL_NAME[game], "s_vis"])
        sphys = float(sel.loc[SEL_NAME[game], "s_phys"])
        for cond in CONDS:
            s = seed_scores(ns, game, cond)
            if s is None:
                continue
            h, h0 = float(hns(s.mean(), game)), float(hns(scratch.mean(), game))
            delta = h - h0
            lo, hi = boot_delta(s, scratch, game)
            rows.append({"game": game, "condition": cond, "s_vis": svis, "s_phys": sphys,
                         "n": len(s), "n_scratch": len(scratch),
                         "hns": h, "hns_scratch": h0, "delta": delta,
                         # Fractional loss. delta alone is bounded by the game's
                         # own scratch HNS, so a game with little headroom (Ms
                         # Pac-Man, HNS 0.08) cannot post a large delta however
                         # badly it does; rel makes the games comparable.
                         "rel": delta / h0 if h0 else float("nan"),
                         "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("Freeze cost relative to from-scratch, in HNS (delta) and as a fraction of the")
    print("game's own from-scratch HNS (rel). Negative = worse than scratch.")
    print("95% CI from seed-level bootstrap; * = CI excludes zero.\n")
    for cond in CONDS:
        d = df[df.condition == cond].sort_values("s_vis")
        if d.empty:
            continue
        print(f"--- {cond} ---")
        for _, r in d.iterrows():
            star = "  *" if (r.lo > 0) or (r.hi < 0) else ""
            print(f"  {TITLE[r.game]:<13} S_vis={r.s_vis:.2f} S_phys={r.s_phys:.2f}  "
                  f"delta={r.delta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]  "
                  f"rel={r.rel:+6.0%}  n={r.n:.0f}{star}")
        if len(d) >= 3:
            for target in ("delta", "rel"):
                out = []
                for axis in ("s_vis", "s_phys"):
                    pr = np.corrcoef(d[axis].values, d[target].values)[0, 1]
                    sr = pd.Series(d[axis].values).corr(pd.Series(d[target].values),
                                                        method="spearman")
                    out.append(f"{axis}: r={pr:+.2f} rho={sr:+.2f}")
                print(f"    vs {target:<5} " + "   ".join(out) + f"   (n={len(d)})")
        print()

    print("n=4 games: these correlations are a directional check on the pre-registered")
    print("S_vis prediction of Sec. 4.1, not an inferential test. Read the orderings.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
