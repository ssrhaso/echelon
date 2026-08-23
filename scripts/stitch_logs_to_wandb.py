#!/usr/bin/env python
"""Reconstruct CLEAN stitch eval curves on W&B from the SLURM log files.

Why: when two stitch jobs share a node, their live wandb runs cross-contaminate
(the wandb service socket is shared per-node, outside WANDB_DIR control), so the
online eval curves get scrambled between runs. Each job's stdout log, however, is
its own process output and is ground truth. This reads logs/stitch_*.out, parses
the true eval curve and the run's intended name, and writes one clean wandb run
per (cell, seed) under group 'stitch_clean' (names: transfer/stitchclean/...),
kept separate from the crossed originals so analysis can filter to the correct data.

Idempotent: fixed run id per (cell, seed), resume='allow' -> re-running after more
runs finish UPDATES the same clean run instead of duplicating. When a cell/seed
has several logs (re-submits), the one with the MOST eval points wins.

Run on ISCA from the repo root (where logs/ and your wandb auth live):
    python scripts/stitch_logs_to_wandb.py
"""
import glob
import os
import re
import sys

import wandb

ENT = os.environ.get("WANDB_ENTITY", "haso-university-of-the-west-of-england")
PROJ = os.environ.get("WANDB_PROJECT", "nnet")
EVAL_PERIOD = 5
LOGDIR = "logs"

# The job echoes `=== array task N | cell=DPP seed=0 | ...` to stdout (.out);
# wandb's "Syncing run" line goes to stderr (.err), so parse the cell/seed here.
TASK_RE = re.compile(r"cell=([A-Z]{3})\s+seed=(\d+)")
EVAL_RE = re.compile(r"eval score:\s*([-\d.]+)")


def parse_log(path):
    """Return (cell, seed, [eval scores]) from one SLURM stdout log."""
    cell, seed, evals = None, None, []
    with open(path, errors="ignore") as f:
        for line in f:
            if cell is None:
                m = TASK_RE.search(line)
                if m:
                    cell, seed = m.group(1), int(m.group(2))
            m = EVAL_RE.search(line)
            if m:
                evals.append(float(m.group(1)))
    return cell, seed, evals


def main():
    logs = sorted(glob.glob(os.path.join(LOGDIR, "stitch_*.out")))
    if not logs:
        sys.exit(f"no logs in {LOGDIR}/ -- run from the repo root on ISCA")

    # Collapse to one (cell, seed) -> best log (most eval points = the real run).
    best = {}
    for path in logs:
        cell, seed, evals = parse_log(path)
        if cell is None or not evals:
            print(f"skip {os.path.basename(path)} (cell={cell}, n_evals={len(evals)})")
            continue
        key = (cell, seed)
        if key in best and len(best[key][1]) >= len(evals):
            continue
        best[key] = (evals, path)

    for (cell, seed), (evals, path) in sorted(best.items()):
        run = wandb.init(
            entity=ENT, project=PROJ,
            id=f"stitchclean{cell}s{seed}", resume="allow",
            name=f"transfer/stitchclean/demon_attack/{cell}/seed{seed}",
            group="stitch_clean", job_type=cell,
            tags=["stitch", "clean-from-log"],
            config={"cell": cell, "seed": seed,
                    "source_log": os.path.basename(path), "n_evals": len(evals)},
            reinit=True,
        )
        for i, v in enumerate(evals):
            wandb.log({"Evaluation-epoch/0/score": v, "epoch": (i + 1) * EVAL_PERIOD},
                      step=(i + 1) * EVAL_PERIOD)
        nf = sum(evals[-5:]) / min(5, len(evals))
        run.summary["near_final_return"] = round(nf, 2)
        run.summary["n_evals"] = len(evals)
        run.summary["complete"] = len(evals) >= 10
        run.finish()
        flag = "done" if len(evals) >= 10 else f"partial({len(evals)}/10)"
        print(f"  {cell} seed{seed}: n={len(evals):2} near-final={nf:6.1f}  [{flag}]")

    print(f"\nReconstructed {len(best)} clean runs -> W&B group 'stitch_clean' "
          f"(filter name 'stitchclean'). Re-run after more jobs finish to update.")


if __name__ == "__main__":
    main()
