#!/usr/bin/env python
"""Build the codebook-stitching result table from the SLURM logs (ground truth).

The live W&B eval data cross-contaminates when two stitch jobs share a node, so
the per-process stdout logs are the authoritative source (this script never reads
or writes W&B). It parses logs/stitch_*.out, extracts each run's true eval curve,
eval game, cell, and seed, and emits per eval game:

  - per-cell near-final return (mean of the last 5 eval points), with n and spread
  - per-level MAIN EFFECT across the 2^3 factorial (matched donor - foreign donor)

Cells are read as L0 L1 L2 donor letters: D = matched (eval game's own codebook),
P = foreign (Pong). PPP = all-foreign = the freeze-L012 baseline (per game).

Usage:
    python scripts/stitch_logs_to_table.py --logdir ~/Desktop/echelon_logs --game ms_pacman
    python scripts/stitch_logs_to_table.py --logdir ./logs --game all
"""
import argparse
import glob
import os
import re
import statistics as st

CELLS_ORDER = ["PPP", "DPP", "PDP", "PPD", "DDP", "PDD", "DPD", "DDD"]
DESC = {"PPP": "none (foreign)", "DPP": "coarse", "PDP": "mid", "PPD": "fine",
        "DDP": "coarse+mid", "PDD": "mid+fine", "DPD": "coarse+fine", "DDD": "all (ceiling)"}
LEVEL = {0: "COARSE (L0)", 1: "MID (L1)", 2: "FINE (L2)"}
GAME_RE = re.compile(r"env_name:\s*atari100k-([a-z0-9_]+)", re.I)
TASK = re.compile(r"cell=([A-Z]{3})\s+seed=(\d+)")
EV = re.compile(r"eval score:\s*([-\d.]+)")


def parse_logs(logdir):
    """Return {(game, cell, seed): [eval scores]}, keeping the log with the most points."""
    best = {}
    for p in glob.glob(os.path.join(logdir, "**", "stitch_*.out"), recursive=True):
        game = cell = seed = None
        ev = []
        with open(p, errors="ignore") as f:
            for line in f:
                if game is None:
                    m = GAME_RE.search(line)
                    if m:
                        game = m.group(1)
                m = TASK.search(line)
                if m and cell is None:
                    cell, seed = m.group(1), int(m.group(2))
                m = EV.search(line)
                if m:
                    ev.append(float(m.group(1)))
        if cell is None or not ev:
            continue
        game = game or "demon_attack"  # back-compat for logs without the env_name line
        k = (game, cell, seed)
        if k not in best or len(ev) > len(best[k]):
            best[k] = ev
    return best


def near_final(v):
    return st.mean(v[-5:]) if len(v) >= 5 else st.mean(v)


def baseline_ppp(game):
    """freeze-L012 (all-foreign) baseline for the game, from the manifest if present."""
    try:
        import yaml
        cfg = yaml.safe_load(open("experiments/codebook_stitching.yaml"))
        return cfg["games"].get(game, {}).get("baseline_ppp")
    except Exception:
        return None


def report(game, best, min_evals, ppp_override):
    cellvals = {}
    for cell in ["DPP", "PDP", "PPD", "DDP", "PDD", "DPD", "DDD"]:
        vs = [near_final(best[(game, cell, s)]) for s in range(3)
              if (game, cell, s) in best and len(best[(game, cell, s)]) >= min_evals]
        if vs:
            cellvals[cell] = vs
    if not cellvals:
        print(f"\n[{game}] no completed cells found")
        return
    ppp = ppp_override if ppp_override is not None else baseline_ppp(game)
    if ppp is not None:
        cellvals["PPP"] = [float(ppp)]

    print(f"\n========== {game} ==========")
    print(f"{'cell':5}{'matched':16}{'n':>3}{'mean':>8}   spread")
    print("-" * 45)
    for c in CELLS_ORDER:
        if c in cellvals:
            v = cellvals[c]
            print(f"{c:5}{DESC[c]:16}{len(v):>3}{st.mean(v):>8.0f}   [{min(v):.0f}-{max(v):.0f}]")
    cm = {c: st.mean(v) for c, v in cellvals.items()}
    print("per-level MAIN EFFECT (matched - foreign, mean over factorial cells):")
    for i in range(3):
        da = [cm[c] for c in cm if c[i] == "D"]
        pg = [cm[c] for c in cm if c[i] == "P"]
        if da and pg:
            print(f"  {LEVEL[i]:12}: matched {st.mean(da):.0f} vs foreign {st.mean(pg):.0f}"
                  f"  -> {st.mean(da) - st.mean(pg):+.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="results/stitch_logs")
    ap.add_argument("--game", default="demon_attack",
                    help="eval game token, or 'all' for every game found in the logs")
    ap.add_argument("--min-evals", type=int, default=8,
                    help="drop seeds with fewer eval points (walltime-killed / incomplete)")
    ap.add_argument("--ppp", type=float, default=None,
                    help="override all-foreign baseline (else from manifest)")
    args = ap.parse_args()

    best = parse_logs(args.logdir)
    games = sorted({g for g, _, _ in best}) if args.game == "all" else [args.game]
    for g in games:
        report(g, best, args.min_evals, args.ppp)
    print("\nrefs (DA): scratch 546 | freeze-L012 502 | freeze-all 310")


if __name__ == "__main__":
    main()
