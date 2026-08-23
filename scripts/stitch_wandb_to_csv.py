#!/usr/bin/env python
"""Pull all codebook-stitching runs from W&B and export core metrics to CSV.

Queries entity/project for runs named transfer/stitch/<game>/<cell>/seed<seed>,
computes each run's near-final return (mean of its last 5 eval points) via
scan_history, and writes:

  - results/stitch_wandb_runs.csv     one row per run (raw, for auditing)
  - results/stitch_wandb_summary.csv  per (game, cell) aggregate + per-level
                                       main effect, paper-table-ready

Caveat (see memory/project_codebook_stitching.md): early stitch jobs sharing a
SLURM node could cross-contaminate live W&B eval curves. The per-task
TMPDIR/WANDB_DIR fix landed before this 63/63 grid completed, but if any
per-cell mean here looks inconsistent with scripts/stitch_logs_to_table.py
(which reads the ISCA .out logs instead), trust the logs.

Usage:
    python scripts/stitch_wandb_to_csv.py
    python scripts/stitch_wandb_to_csv.py --game demon_attack
"""
import argparse
import csv
import os
import re
import statistics as st

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "haso-university-of-the-west-of-england")
PROJECT = os.environ.get("WANDB_PROJECT", "nnet")
EVAL_KEYS = ["Evaluation-epoch/0/score", "Evaluation-epoch\\0\\score"]

NAME_RE = re.compile(r"^transfer/stitch/(?P<game>[a-z0-9_]+)/(?P<cell>[A-Z]{3})/seed(?P<seed>\d+)$")
CELLS_ORDER = ["PPP", "DPP", "PDP", "PPD", "DDP", "PDD", "DPD", "DDD"]
DESC = {"PPP": "none (foreign)", "DPP": "coarse", "PDP": "mid", "PPD": "fine",
        "DDP": "coarse+mid", "PDD": "mid+fine", "DPD": "coarse+fine", "DDD": "all (ceiling)"}
LEVEL = {0: "COARSE (L0)", 1: "MID (L1)", 2: "FINE (L2)"}


def fetch_evals(run):
    """Return sorted [(step, score), ...] for whichever eval key the run used.

    Bounded to the tail of the run (via the summary's logged step count) instead
    of a full scan_history -- eval points are sparse (every few thousand steps)
    but a full unbounded scan over a ~100k-step run's dense per-step training
    metrics is far slower than needed when we only want the last 5 eval points.
    """
    max_step = run.summary.get("_step") or run.summary.get("global_step")
    min_step = max(0, int(max_step) - 60000) if max_step else 0
    for key in EVAL_KEYS:
        rows = []
        for row in run.scan_history(keys=[key, "global_step"], page_size=2000, min_step=min_step):
            s, v = row.get("global_step"), row.get(key)
            if v is None or s is None:
                continue
            rows.append((int(s), float(v)))
        if rows:
            rows.sort()
            return rows
    return []


def near_final(vals):
    return st.mean(vals[-5:]) if len(vals) >= 5 else (st.mean(vals) if vals else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None, help="restrict to one eval game (default: all)")
    ap.add_argument("--min-evals", type=int, default=8,
                    help="drop runs with fewer eval points (walltime-killed / incomplete)")
    ap.add_argument("--out-runs", default="results/stitch_wandb_runs.csv")
    ap.add_argument("--out-summary", default="results/stitch_wandb_summary.csv")
    args = ap.parse_args()

    api = wandb.Api(timeout=120)
    print(f"listing runs in {ENTITY}/{PROJECT} ...", flush=True)
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    matched = []
    for r in all_runs:
        m = NAME_RE.match(r.name or "")
        if not m:
            continue
        if args.game and m.group("game") != args.game:
            continue
        matched.append((r, m))
    print(f"{len(matched)} stitch runs matched (of {len(all_runs)} total in project)", flush=True)

    rows = []
    for r, m in matched:
        game, cell, seed = m.group("game"), m.group("cell"), int(m.group("seed"))
        evals = fetch_evals(r)
        nf = near_final([v for _, v in evals])
        rows.append({
            "game": game, "cell": cell, "level_desc": DESC.get(cell, cell), "seed": seed,
            "n_evals": len(evals), "near_final_return": round(nf, 2) if nf is not None else "",
            "complete": len(evals) >= args.min_evals,
            "state": r.state, "run_id": r.id, "run_name": r.name,
            "created_at": r.created_at,
        })
        print(f"  {game}/{cell}/seed{seed}: n_evals={len(evals):2} "
              f"near_final={nf if nf is None else round(nf, 1)} state={r.state}", flush=True)

    rows.sort(key=lambda d: (d["game"], CELLS_ORDER.index(d["cell"]) if d["cell"] in CELLS_ORDER else 99, d["seed"]))

    os.makedirs(os.path.dirname(args.out_runs), exist_ok=True)
    with open(args.out_runs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out_runs} ({len(rows)} rows)")

    # --- per (game, cell) aggregate + per-level main effect ---
    games = sorted({d["game"] for d in rows})
    summary_rows = []
    for g in games:
        cellvals = {}
        for cell in CELLS_ORDER:
            vs = [d["near_final_return"] for d in rows
                  if d["game"] == g and d["cell"] == cell and d["complete"] and d["near_final_return"] != ""]
            if vs:
                cellvals[cell] = vs
        for cell in CELLS_ORDER:
            if cell in cellvals:
                v = cellvals[cell]
                summary_rows.append({
                    "game": g, "row_type": "cell", "cell_or_level": cell,
                    "label": DESC.get(cell, cell), "n": len(v),
                    "mean": round(st.mean(v), 2),
                    "min": round(min(v), 2), "max": round(max(v), 2),
                })
        cm = {c: st.mean(v) for c, v in cellvals.items()}
        for i in range(3):
            da = [cm[c] for c in cm if c[i] == "D"]
            pg = [cm[c] for c in cm if c[i] == "P"]
            if da and pg:
                summary_rows.append({
                    "game": g, "row_type": "main_effect", "cell_or_level": f"L{i}",
                    "label": LEVEL[i], "n": "",
                    "mean": round(st.mean(da) - st.mean(pg), 2),
                    "min": round(st.mean(pg), 2), "max": round(st.mean(da), 2),
                })

    with open(args.out_summary, "w", newline="") as f:
        fieldnames = ["game", "row_type", "cell_or_level", "label", "n", "mean", "min", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {args.out_summary} ({len(summary_rows)} rows)")
    print("\n(main_effect rows: min=foreign mean, max=matched mean, mean=matched-foreign delta)")
    print("Caveat: cross-check demon_attack cells against scripts/stitch_logs_to_table.py "
          "(ISCA .out logs) before citing in the paper -- see memory/project_codebook_stitching.md")


if __name__ == "__main__":
    main()
