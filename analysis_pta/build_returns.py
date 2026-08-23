#!/usr/bin/env python
"""Build the canonical per-seed near-final return table for the PTA submission.

Sources, in evidence-priority order:
  1. isca_logs/aug26_*.out  -- SLURM stdout, ground truth for the aug15/26 control batch
  2. wandb_export/eval_epoch_long.csv -- historical ladder eval curves

Statistic: near-final return = mean of a seed's LAST 5 evaluation points
(eval every 5 epochs; a complete run has 10 points). Cells with <10 eval points
are marked incomplete and excluded from aggregates.

Writes analysis_pta/returns_per_seed.csv and returns_cells.csv. Read-only w.r.t.
all existing artifacts.
"""
import csv, glob, os, re, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis_pta")
EXPORT = os.path.join(ROOT, "wandb_export", "eval_epoch_long.csv")
LOGDIR = os.path.join(ROOT, "isca_logs")
COMPLETE = 10

HEADER_RE = re.compile(r"===\s*(?:isca-aug26\s+)?task \d+ \| (?P<game>\S+) / (?P<cond>\S+) / seed (?P<seed>\d+) ===")
EVAL_RE = re.compile(r"^eval score: ([0-9.+-eE]+)")
DONOR_RE = re.compile(r"--transfer_checkpoint (\S+)")

# Historical condition labels -> scientifically accurate names.
RENAME = {
    "scratch":     "from-scratch",
    "warmstart":   "adapt-CB012",
    "freeze-L0":   "freeze-CB0",
    "freeze-L01":  "freeze-CB01",
    "freeze-L012": "freeze-CB012",
    "freeze-enc":  "adapt-CB012+encinit",   # was "freeze-enc": encoder was NEVER frozen
    "freeze-all":  "freeze-CB012+encinit",  # was "freeze-all": enc & WM were NEVER frozen
}


def parse_log(path):
    game = cond = seed = donor = None
    scores = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = HEADER_RE.search(line)
            if m:
                game, cond, seed = m.group("game"), m.group("cond"), int(m.group("seed"))
            m = DONOR_RE.search(line)
            if m and donor is None:
                donor = os.path.basename(m.group(1))
            m = EVAL_RE.match(line.strip())
            if m:
                scores.append(float(m.group(1)))
    return game, cond, seed, donor, scores


def collect_logs():
    """Return {(game, cond_label, seed): (n_evals, nearfinal, source)}; best attempt per cell."""
    best = {}
    for path in sorted(glob.glob(os.path.join(LOGDIR, "aug26_*.out"))):
        game, cond, seed, donor, scores = parse_log(path)
        if game is None:
            continue
        # Distinguish donor for the second-donor ladder.
        if donor and donor.startswith("da_seed4"):
            label = {"warmstart": "adapt-CB012 (DA donor)",
                     "freeze-L012": "freeze-CB012 (DA donor)"}.get(cond, cond + " (DA donor)")
        else:
            label = {"weight-transfer": "whole-model-transfer",
                     "freeze-random": "freeze-CBrand012"}.get(cond, RENAME.get(cond, cond))
        key = (game, label, seed)
        if key not in best or len(scores) > best[key][0]:
            best[key] = (len(scores), st.mean(scores[-5:]) if len(scores) >= 5 else None,
                         os.path.basename(path))
    return best


def collect_export():
    runs = defaultdict(list)
    with open(EXPORT, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["metric"] != "score":
                continue
            runs[(row["game"], row["condition"], int(row["seed"]), row["run_id"])].append(
                (int(row["step"]), float(row["value"])))
    out = {}
    for (game, cond, seed, rid), pts in runs.items():
        pts.sort()
        sc = [v for _s, v in pts]
        label = RENAME.get(cond, cond)
        key = (game, label, seed, rid)
        out[key] = (len(sc), st.mean(sc[-5:]) if len(sc) >= 5 else None,
                    "wandb_export/eval_epoch_long.csv:" + rid)
    return out


def main():
    rows = []
    for (g, c, s, rid), (n, nf, src) in sorted(collect_export().items()):
        rows.append(dict(game=g, condition=c, seed=s, run=rid, n_evals=n,
                         nearfinal="" if nf is None else round(nf, 4),
                         complete=int(n >= COMPLETE), source=src, evidence="wandb_export"))
    for (g, c, s), (n, nf, src) in sorted(collect_logs().items()):
        rows.append(dict(game=g, condition=c, seed=s, run=src, n_evals=n,
                         nearfinal="" if nf is None else round(nf, 4),
                         complete=int(n >= COMPLETE), source="isca_logs/" + src,
                         evidence="slurm_stdout"))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "returns_per_seed.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cells = defaultdict(list)
    for r in rows:
        if r["complete"] and r["nearfinal"] != "":
            cells[(r["game"], r["condition"], r["evidence"])].append(float(r["nearfinal"]))
    with open(os.path.join(OUT, "returns_cells.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game", "condition", "evidence", "n", "mean", "std", "median", "q1", "q3", "values"])
        for (g, c, e), v in sorted(cells.items()):
            v = sorted(v)
            q1 = st.median(v[:len(v) // 2]) if len(v) > 1 else v[0]
            q3 = st.median(v[(len(v) + 1) // 2:]) if len(v) > 1 else v[0]
            w.writerow([g, c, e, len(v), round(st.mean(v), 2),
                        round(st.stdev(v), 2) if len(v) > 1 else "",
                        round(st.median(v), 2), round(q1, 2), round(q3, 2),
                        ";".join(f"{x:.2f}" for x in v)])
    print("wrote analysis_pta/returns_per_seed.csv and returns_cells.csv")
    inc = [r for r in rows if not r["complete"]]
    print(f"incomplete runs excluded from cells: {len(inc)}")
    for r in inc:
        print(f"  {r['game']:<13} {r['condition']:<28} seed{r['seed']} n_evals={r['n_evals']} [{r['source']}]")


if __name__ == "__main__":
    main()
