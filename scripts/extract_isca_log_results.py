#!/usr/bin/env python
"""Extract per-run eval curves for the aug15 experiment batch (weight-transfer,
freeze-random, DA-donor) from ISCA SLURM stdout logs, and compare against the
paper's existing cells.

The W&B summaries for these runs are cross-contaminated between tasks sharing a
node (identical eval values on node-pairs, nonsense step counts), so the stdout
logs are the ONLY trusted source: each task prints its own `eval score:` lines
in-process. Header line `=== task N | <game> / <condition> / seed <s> ===`
identifies the cell; W&B is never consulted.

Inputs:
  isca_logs/aug26_*.out              (scp'd from ISCA ~/echelon/logs/)
  wandb_export/eval_epoch_long.csv   (paper's eval curves)

Protocol: near-final score = mean of the LAST 5 eval points (eval every 5
epochs, 10 points for a complete run), matching tab:results_main. A cell with
several attempts (crashed + requeued) keeps the attempt with most evals.

Usage: python scripts/extract_isca_log_results.py
"""
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

LOG_DIR = "isca_logs"
EXPORT = "wandb_export/eval_epoch_long.csv"
HEADER_RE = re.compile(r"=== task \d+ \| (?P<game>\S+) / (?P<cond>\S+) / seed (?P<seed>\d+) ===")
EVAL_RE = re.compile(r"^eval score: ([0-9.+-eE]+)")

# paper cells to compare against, per game (from eval_epoch_long.csv)
BASELINE_CONDS = ["scratch", "warmstart", "freeze-L012"]
GAMES = ["breakout", "ms_pacman", "up_n_down", "demon_attack"]
COMPLETE_N_EVALS = 10


def parse_log(path):
    game = cond = seed = None
    scores = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = HEADER_RE.search(line)
            if m:
                game, cond, seed = m.group("game"), m.group("cond"), int(m.group("seed"))
            m = EVAL_RE.match(line.strip())
            if m:
                scores.append(float(m.group(1)))
    return game, cond, seed, scores


def nearfinal(scores):
    return st.mean(scores[-5:]) if len(scores) >= 5 else None


def main():
    # ---- new runs from logs; best attempt per cell ----
    best = {}
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "aug26_*.out"))):
        game, cond, seed, scores = parse_log(path)
        if game is None:
            continue
        key = (game, cond, seed)
        if key not in best or len(scores) > len(best[key][1]):
            best[key] = (os.path.basename(path), scores)

    print("== per-cell status (from logs; {} evals = complete) ==".format(COMPLETE_N_EVALS))
    cells = defaultdict(dict)  # (game, cond) -> seed -> nearfinal
    for (game, cond, seed), (fname, scores) in sorted(best.items()):
        status = "complete" if len(scores) >= COMPLETE_N_EVALS else f"PARTIAL {len(scores)}/10"
        nf = nearfinal(scores)
        print(f"  {game:<13} {cond:<16} seed{seed}  {status:<12} "
              f"final={scores[-1] if scores else None}  nearfinal={nf and round(nf, 1)}  [{fname}]")
        if len(scores) >= COMPLETE_N_EVALS:
            cells[(game, cond)][seed] = nf

    # ---- paper cells from the eval export ----
    runs = defaultdict(list)  # (game, cond, run_id) -> [(step, score)]
    with open(EXPORT, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["metric"] != "score" or row["game"] not in GAMES:
                continue
            if row["condition"] not in BASELINE_CONDS:
                continue
            runs[(row["game"], row["condition"], row["run_id"])].append(
                (int(row["step"]), float(row["value"])))
    paper = defaultdict(list)  # (game, cond) -> [nearfinal per seed-run]
    for (game, cond, _rid), pts in runs.items():
        pts.sort()
        scores = [v for _s, v in pts]
        if len(scores) >= COMPLETE_N_EVALS:
            paper[(game, cond)].append(st.mean(scores[-5:]))

    # ---- comparison table ----
    def fmt_cell(vals):
        if not vals:
            return "---"
        if len(vals) == 1:
            return f"{vals[0]:.1f} (n=1)"
        return f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f} (n={len(vals)})"

    print("\n== near-final score, mean ± std over seeds ==")
    print(f"  {'game':<13} {'condition':<18} {'source':<7} cell")
    for game in GAMES:
        for cond in BASELINE_CONDS:
            if (game, cond) in paper:
                print(f"  {game:<13} {cond:<18} {'paper':<7} {fmt_cell(paper[(game, cond)])}")
        for (g, cond), by_seed in sorted(cells.items()):
            if g == game:
                print(f"  {game:<13} {cond:<18} {'NEW':<7} {fmt_cell(list(by_seed.values()))}")
        print()


if __name__ == "__main__":
    main()
