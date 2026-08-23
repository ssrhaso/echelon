#!/usr/bin/env python
"""Primary statistics for the PTA submission.

Inputs : analysis_pta/returns_per_seed.csv (built by build_returns.py)
         judgement/runs_full.csv           (host / precision provenance)
Outputs: analysis_pta/{stack_audit,iqm_hns,prob_improvement,per_game_cells,controls}.csv

Design decisions, all stated in the paper:
  * Statistic  : near-final return = mean of a seed's last 5 evaluation points.
  * Inclusion  : complete 10-point eval curve on the standard 10k-step grid.
  * Aggregation: human-normalized score (HNS), interquartile mean (IQM) over the
                 pooled game x seed matrix, stratified bootstrap CIs (seeds
                 resampled within game), 10,000 replicates, seed 0.
  * Exclusion  : a game is dropped from the cross-game aggregate if its ladder
                 mixes GPU or numerical precision within or across cells.
"""
import csv
import os
import statistics as st
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis_pta")
REPS, RNG_SEED = 10000, 0

# Canonical Atari-100k (random, human) references.
REF = {"breakout": (1.7, 30.5), "demon_attack": (152.1, 1971.0),
       "ms_pacman": (307.3, 6951.6), "up_n_down": (533.4, 11693.2)}

LADDER = ["from-scratch", "adapt-CB012", "freeze-CB0", "freeze-CB01",
          "freeze-CB012", "adapt-CB012+encinit", "freeze-CB012+encinit"]
BASE = "from-scratch"
CONTROLS = ["freeze-CBrand012", "whole-model-transfer",
            "adapt-CB012 (DA donor)", "freeze-CB012 (DA donor)"]

# Historical label -> accurate label (must match build_returns.RENAME).
HIST = {"scratch": "from-scratch", "warmstart": "adapt-CB012", "freeze-L0": "freeze-CB0",
        "freeze-L01": "freeze-CB01", "freeze-L012": "freeze-CB012",
        "freeze-enc": "adapt-CB012+encinit", "freeze-all": "freeze-CB012+encinit"}


def hns(game, v):
    r, h = REF[game]
    return (v - r) / (h - r)


def load_returns():
    """Ladder cells come from the historical export; control cells from SLURM stdout.

    Keeping the two evidence streams apart is deliberate: the aug26 cluster batch
    ran on a different stack (A100 / float32) from the historical per-game ladders,
    so a cell must never pool runs from both.
    """
    ladder, control = defaultdict(list), defaultdict(list)
    path = os.path.join(OUT, "returns_per_seed.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["complete"] != "1" or not r["nearfinal"]:
                continue
            tgt = ladder if r["evidence"] == "wandb_export" else control
            tgt[(r["game"], r["condition"])].append(float(r["nearfinal"]))
    return ladder, control


def stack_audit():
    """Which games have a single (gpu, precision) stack across their whole ladder?"""
    cells = defaultdict(set)
    path = os.path.join(ROOT, "judgement", "runs_full.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("canonical") != "True" or r.get("complete") != "True":
                continue
            if r["game"] not in REF or r["condition"] not in HIST:
                continue
            cells[(r["game"], HIST[r["condition"]])].add((r["gpu"], r["precision"]))
    game_stacks = defaultdict(set)
    for (g, _c), s in cells.items():
        game_stacks[g] |= s
    rows, clean = [], []
    for g in sorted(game_stacks):
        mixed_cells = sorted(c for (gg, c), s in cells.items() if gg == g and len(s) > 1)
        ok = len(game_stacks[g]) == 1
        rows.append(dict(game=g, n_stacks=len(game_stacks[g]),
                         stacks=" | ".join(a + "/" + b for a, b in sorted(game_stacks[g])),
                         mixed_cells=";".join(mixed_cells) or "-",
                         verdict="single-stack" if ok else "MIXED -> excluded"))
        if ok:
            clean.append(g)
    with open(os.path.join(OUT, "stack_audit.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return clean, rows


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))


def poi(x_by_game, y_by_game):
    """Probability of improvement: mean over games of P(X>Y) + 0.5 P(X=Y)."""
    vals = []
    for g in x_by_game:
        x, y = np.asarray(x_by_game[g]), np.asarray(y_by_game[g])
        d = x[:, None] - y[None, :]
        vals.append(float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size))
    return float(np.mean(vals))


def main():
    data, control = load_returns()
    clean, audit_rows = stack_audit()
    print("stack audit:")
    for r in audit_rows:
        print("  {:<14} {:<20} stacks={}  mixed_cells={}".format(
            r["game"], r["verdict"], r["n_stacks"], r["mixed_cells"]))
    games = [g for g in clean if g in REF]
    print("\ncross-game aggregate over: {}\n".format(games))

    rng = np.random.default_rng(RNG_SEED)
    base_mat = {g: np.array([hns(g, v) for v in data[(g, BASE)]]) for g in games}

    iqm_rows, poi_rows = [], []
    for cond in LADDER:
        mat = {g: np.array([hns(g, v) for v in data[(g, cond)]])
               for g in games if data.get((g, cond))}
        if len(mat) != len(games):
            continue
        pt = iqm(np.concatenate(list(mat.values())))
        bs = np.empty(REPS)
        for b in range(REPS):
            pooled = [mat[g][rng.integers(0, len(mat[g]), len(mat[g]))] for g in mat]
            bs[b] = iqm(np.concatenate(pooled))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        n_runs = sum(len(v) for v in mat.values())
        iqm_rows.append(dict(condition=cond, n_games=len(mat), n_runs=n_runs,
                             iqm_hns=round(pt, 4), ci_lo=round(float(lo), 4),
                             ci_hi=round(float(hi), 4)))
        if cond != BASE:
            p = poi(mat, base_mat)
            bs2 = np.empty(REPS)
            for b in range(REPS):
                xa = {g: mat[g][rng.integers(0, len(mat[g]), len(mat[g]))] for g in mat}
                ya = {g: base_mat[g][rng.integers(0, len(base_mat[g]), len(base_mat[g]))]
                      for g in mat}
                bs2[b] = poi(xa, ya)
            l2, h2 = np.percentile(bs2, [2.5, 97.5])
            poi_rows.append(dict(condition=cond, poi_vs_scratch=round(p, 4),
                                 ci_lo=round(float(l2), 4), ci_hi=round(float(h2), 4)))

    for name, rows in (("iqm_hns", iqm_rows), ("prob_improvement", poi_rows)):
        with open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("IQM of HNS (stratified bootstrap, 95% CI):")
    for r in iqm_rows:
        print("  {:<24} {:.3f}  [{:.3f}, {:.3f}]   n_runs={}".format(
            r["condition"], r["iqm_hns"], r["ci_lo"], r["ci_hi"], r["n_runs"]))
    print("\nProbability of improvement vs from-scratch:")
    for r in poi_rows:
        print("  {:<24} {:.3f}  [{:.3f}, {:.3f}]".format(
            r["condition"], r["poi_vs_scratch"], r["ci_lo"], r["ci_hi"]))

    with open(os.path.join(OUT, "per_game_cells.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game", "condition", "n", "median", "iqr_lo", "iqr_hi",
                    "mean", "std", "hns_median"])
        for g in sorted(REF):
            for c in LADDER:
                v = sorted(data.get((g, c), []))
                if not v:
                    continue
                q1, q3 = np.percentile(v, [25, 75])
                w.writerow([g, c, len(v), round(st.median(v), 2), round(float(q1), 2),
                            round(float(q3), 2), round(st.mean(v), 2),
                            round(st.stdev(v), 2) if len(v) > 1 else "",
                            round(hns(g, st.median(v)), 4)])

    ctrl_games = ["demon_attack", "ms_pacman", "up_n_down"]
    with open(os.path.join(OUT, "controls.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game", "condition", "n", "mean", "std", "median", "values"])
        for g in ctrl_games:
            for c in CONTROLS:
                v = sorted(control.get((g, c), []))
                if not v:
                    continue
                w.writerow([g, c, len(v), round(st.mean(v), 1),
                            round(st.stdev(v), 1) if len(v) > 1 else "",
                            round(st.median(v), 1), ";".join("{:.1f}".format(x) for x in v)])
    print("\nwrote stack_audit / iqm_hns / prob_improvement / per_game_cells / controls .csv")


if __name__ == "__main__":
    main()
