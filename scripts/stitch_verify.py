"""Log-vs-W&B verification for the codebook-stitching grids.

When two stitch jobs share a SLURM node their live W&B eval curves can
cross-contaminate: the wandb service socket is per-node and outside WANDB_DIR
control. Cells re-derived from raw per-process stdout can disagree with W&B by
enough to flip a main effect, so every cell is verified against its own process
output before it is used.

Ground truth is the per-process SLURM stdout: it is that process's own file and cannot
be crossed. W&B is the value under test.

Keying: each array task writes logs/stitch_<jobid>_<taskid>.out (the curve, the game,
the cell/seed) and .err (wandb's "View run at .../runs/<id>" line). We key by the W&B
RUN ID from the .err whenever it is present, and fall back to matching on
(game, cell, seed) only when it is not -- the fallback is recorded per row so a
run-id-keyed verdict is never confused with a name-matched one.

Per-cell verdict:
    match      log curve and W&B curve agree within tolerance
    mismatch   they disagree -> that cell's W&B value is not trustworthy
    no-log     no stdout for this cell -> unverifiable, cannot be cited
    no-wandb   stdout exists but no matching W&B run

Outputs (judgement/):
    stitch_verify_<game>.csv      per (cell, seed) verdict row
    stitch_verify_summary.md      per-game verdict table + main-effect comparison
    stitch_rerun_list.txt         the exact cells needing re-runs

Usage:
    python scripts/stitch_verify.py --games demon_attack
    python scripts/stitch_verify.py --games ms_pacman up_n_down --logdir /path/to/isca/logs
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "haso-university-of-the-west-of-england")
PROJECT = os.environ.get("WANDB_PROJECT", "nnet")

CELLS_ORDER = ["PPP", "DPP", "PDP", "PPD", "DDP", "PDD", "DPD", "DDD"]
DESC = {"PPP": "none (foreign)", "DPP": "coarse", "PDP": "mid", "PPD": "fine",
        "DDP": "coarse+mid", "PDD": "mid+fine", "DPD": "coarse+fine", "DDD": "all (ceiling)"}
LEVEL = {0: "COARSE (L0)", 1: "MID (L1)", 2: "FINE (L2)"}

GAME_RE = re.compile(r"env_name:\s*atari100k-([a-z0-9_]+)", re.I)
TASK_RE = re.compile(r"cell=([A-Z]{3})\s+seed=(\d+)")
EVAL_RE = re.compile(r"eval score:\s*([-\d.]+)")
RUNID_RE = re.compile(r"/runs/([A-Za-z0-9]{8})")
NAME_RE = re.compile(r"^transfer/stitch/(?P<game>[a-z0-9_]+)/(?P<cell>[A-Z]{3})/seed(?P<seed>\d+)$")
EVAL_KEYS = ["Evaluation-epoch/0/score", "Evaluation-epoch\\0\\score"]


def parse_log(out_path):
    """Parse one array task's stdout (+ its .err for the W&B run id)."""
    game = cell = seed = None
    evals = []
    with open(out_path, errors="ignore") as fh:
        for line in fh:
            if game is None:
                m = GAME_RE.search(line)
                if m:
                    game = m.group(1)
            if cell is None:
                m = TASK_RE.search(line)
                if m:
                    cell, seed = m.group(1), int(m.group(2))
            m = EVAL_RE.search(line)
            if m:
                evals.append(float(m.group(1)))

    run_id = None
    err_path = out_path[:-4] + ".err"
    if os.path.exists(err_path):
        with open(err_path, errors="ignore") as fh:
            for line in fh:
                m = RUNID_RE.search(line)
                if m:
                    run_id = m.group(1)
                    break
    return {"game": game, "cell": cell, "seed": seed, "evals": evals,
            "run_id": run_id, "log": os.path.basename(out_path)}


def collect_logs(logdir):
    """{(game, cell, seed): [all records]} -- one per array task that ran that cell."""
    out = defaultdict(list)
    pattern = os.path.join(logdir, "**", "stitch_*.out")
    for path in sorted(glob.glob(pattern, recursive=True)):
        rec = parse_log(path)
        if rec["cell"] is None or not rec["evals"]:
            continue
        # Back-compat: the earliest Demon Attack logs predate the env_name echo.
        rec["game"] = rec["game"] or "demon_attack"
        out[(rec["game"], rec["cell"], rec["seed"])].append(rec)
    return out


def select_log(recs, wb_by_id):
    """Pick the log that corresponds to the run W&B actually kept.

    A cell was often re-submitted, and each submission opened its OWN W&B run, so a
    cell can have several logs with several run ids while only one of those runs
    survives under the cell's name. Comparing the wrong log against the surviving run
    manufactures a "mismatch" that is really a superseded re-submit. So: prefer a log
    whose run id is a live W&B run; only then fall back to most-eval-points.
    """
    live = [r for r in recs if r.get("run_id") and r["run_id"] in wb_by_id]
    pool = live or recs
    return max(pool, key=lambda r: len(r["evals"])), bool(live)


def fetch_wandb(games):
    """{(game, cell, seed): {...}} and {run_id: (game, cell, seed)} from W&B."""
    api = wandb.Api(timeout=180)
    print("listing runs in {}/{} ...".format(ENTITY, PROJECT), flush=True)
    out, by_id = {}, {}
    for r in api.runs("{}/{}".format(ENTITY, PROJECT), per_page=500):
        m = NAME_RE.match(r.name or "")
        if not m or m.group("game") not in games:
            continue
        key = (m.group("game"), m.group("cell"), int(m.group("seed")))
        curve = []
        for k in EVAL_KEYS:
            rows = []
            for row in r.scan_history(keys=[k, "global_step"], page_size=2000):
                v = row.get(k)
                if v is None:
                    continue
                rows.append(float(v))
            if rows:
                curve = rows
                break
        rec = {"run_id": r.id, "name": r.name, "state": r.state, "curve": curve}
        out[key] = rec
        by_id[r.id] = key
        print("  wandb {:<52} n={}".format(r.name, len(curve)), flush=True)
    return out, by_id


def near_final(v):
    if not v:
        return None
    return st.mean(v[-5:]) if len(v) >= 5 else st.mean(v)


def compare(log_evals, wb_curve, rtol, atol):
    """Element-wise comparison over the overlapping prefix."""
    n = min(len(log_evals), len(wb_curve))
    if n == 0:
        return False, 0, None
    worst, worst_i = 0.0, None
    for i in range(n):
        a, b = log_evals[i], wb_curve[i]
        d = abs(a - b)
        tol = atol + rtol * max(abs(a), abs(b))
        if d > tol and d > worst:
            worst, worst_i = d, i
    return worst_i is None, n, worst_i


def main_effects(cellmeans):
    """Per-level matched-minus-foreign main effect over the factorial cells."""
    eff = {}
    for i in range(3):
        da = [v for c, v in cellmeans.items() if len(c) == 3 and c[i] == "D"]
        pg = [v for c, v in cellmeans.items() if len(c) == 3 and c[i] == "P"]
        if da and pg:
            eff[i] = st.mean(da) - st.mean(pg)
    return eff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+",
                    default=["ms_pacman", "up_n_down", "demon_attack"])
    ap.add_argument("--logdir", default=os.path.expanduser("~/Desktop/echelon_logs"),
                    help="directory holding the per-task stitch_<job>_<task>.out/.err")
    ap.add_argument("--outdir", default="judgement")
    ap.add_argument("--rtol", type=float, default=0.01,
                    help="relative tolerance per eval point (default 1%%)")
    ap.add_argument("--atol", type=float, default=1.0, help="absolute tolerance floor")
    ap.add_argument("--min-evals", type=int, default=8,
                    help="cells with fewer log eval points count as incomplete")
    args = ap.parse_args()

    if not os.path.isdir(args.logdir):
        sys.exit("ERROR: logdir not found: {}".format(args.logdir))
    os.makedirs(args.outdir, exist_ok=True)

    raw_logs = collect_logs(args.logdir)
    found_games = sorted({g for g, _, _ in raw_logs})
    n_files = sum(len(v) for v in raw_logs.values())
    print("parsed {} log files covering {} cells from {} (games: {})".format(
        n_files, len(raw_logs), args.logdir, ", ".join(found_games) or "none"))
    for g in args.games:
        if g not in found_games:
            print("  !! NO LOGS for {} -- every cell will be 'no-log' (unverifiable). "
                  "Fetch them from ISCA: ~/echelon/logs/stitch_*.out|.err".format(g))

    wb, wb_by_id = fetch_wandb(set(args.games))

    # Resolve each cell to the log matching the surviving W&B run (see select_log).
    logs, resolved_live = {}, {}
    for key, recs in raw_logs.items():
        rec, live = select_log(recs, wb_by_id)
        logs[key] = rec
        resolved_live[key] = live
        if len(recs) > 1:
            print("  {} had {} logs ({} re-submits); using {} [{}]".format(
                "/".join(str(x) for x in key), len(recs), len(recs) - 1,
                rec["log"], "run-id matches live W&B run" if live
                else "NO log matches a live run -- superseded ids only"))

    all_rows = []
    summary_lines = ["# Stitching log-verification (SLURM stdout vs W&B)", "",
                     "Ground truth = per-process SLURM stdout. W&B is the value under test.",
                     "`match` within rtol={} atol={} element-wise over the overlapping prefix."
                     .format(args.rtol, args.atol), ""]
    rerun = []

    for game in args.games:
        rows = []
        seeds = sorted({s for (g, _, s) in list(logs) + list(wb) if g == game})
        cells = [c for c in CELLS_ORDER
                 if any(g == game and cc == c for (g, cc, _) in list(logs) + list(wb))]
        for cell in cells:
            for seed in seeds:
                key = (game, cell, seed)
                lrec, wrec = logs.get(key), wb.get(key)

                # Prefer keying by run id: if the log's .err names a run, that is the
                # run this process actually wrote to, regardless of name matching.
                keyed_by = "name"
                if lrec and lrec.get("run_id"):
                    if lrec["run_id"] in wb_by_id:
                        wrec = wb.get(wb_by_id[lrec["run_id"]])
                        keyed_by = "run_id"
                    else:
                        keyed_by = "run_id(unmatched)"

                if lrec is None and wrec is None:
                    continue
                if lrec is None:
                    verdict = "no-log"
                elif wrec is None or not wrec["curve"]:
                    verdict = "no-wandb"
                else:
                    ok, n_cmp, worst_i = compare(lrec["evals"], wrec["curve"],
                                                 args.rtol, args.atol)
                    verdict = "match" if ok else "mismatch"

                lnf = near_final(lrec["evals"]) if lrec else None
                wnf = near_final(wrec["curve"]) if wrec and wrec["curve"] else None
                row = {
                    "game": game, "cell": cell, "level_desc": DESC.get(cell, cell),
                    "seed": seed, "verdict": verdict, "keyed_by": keyed_by,
                    "run_id": (wrec or {}).get("run_id", ""),
                    "log_file": (lrec or {}).get("log", ""),
                    "n_log_evals": len(lrec["evals"]) if lrec else 0,
                    "n_wandb_evals": len(wrec["curve"]) if wrec else 0,
                    "log_near_final": round(lnf, 3) if lnf is not None else "",
                    "wandb_near_final": round(wnf, 3) if wnf is not None else "",
                    "delta": round(wnf - lnf, 3) if (lnf is not None and wnf is not None) else "",
                    "complete": (lrec is not None and len(lrec["evals"]) >= args.min_evals),
                }
                rows.append(row)
                if verdict in ("mismatch", "no-log", "no-wandb"):
                    rerun.append("{}/{}/seed{}  [{}]".format(game, cell, seed, verdict))

        all_rows.extend(rows)
        out_csv = os.path.join(args.outdir, "stitch_verify_{}.csv".format(game))
        cols = ["game", "cell", "level_desc", "seed", "verdict", "keyed_by", "run_id",
                "log_file", "n_log_evals", "n_wandb_evals", "log_near_final",
                "wandb_near_final", "delta", "complete"]
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print("wrote {} ({} rows)".format(out_csv, len(rows)))

        # ---- per-game verdict table + main-effect comparison ----
        counts = defaultdict(int)
        for r in rows:
            counts[r["verdict"]] += 1
        summary_lines += ["## {}".format(game), "",
                          "verdicts: " + ", ".join("{} {}".format(v, counts[v])
                                                   for v in sorted(counts)) or "none", ""]
        if rows:
            summary_lines += ["| cell | matched | seed | verdict | keyed by | log near-final | W&B near-final | delta |",
                              "|---|---|---|---|---|---|---|---|"]
            for r in rows:
                summary_lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    r["cell"], r["level_desc"], r["seed"], r["verdict"], r["keyed_by"],
                    r["log_near_final"], r["wandb_near_final"], r["delta"]))
            summary_lines.append("")

        # Main effects from each source, over complete cells only. The Demon Attack
        # finest-level sign flip is exactly what this table is here to expose.
        log_means, wb_means = {}, {}
        for cell in CELLS_ORDER:
            lv = [near_final(logs[(game, cell, s)]["evals"]) for s in seeds
                  if (game, cell, s) in logs
                  and len(logs[(game, cell, s)]["evals"]) >= args.min_evals]
            wv = [near_final(wb[(game, cell, s)]["curve"]) for s in seeds
                  if (game, cell, s) in wb and wb[(game, cell, s)]["curve"]]
            if lv:
                log_means[cell] = st.mean(lv)
            if wv:
                wb_means[cell] = st.mean(wv)
        le, we = main_effects(log_means), main_effects(wb_means)
        if le or we:
            summary_lines += ["### per-level main effect (matched - foreign)", "",
                              "| level | from logs | from W&B | sign flip? |",
                              "|---|---|---|---|"]
            for i in range(3):
                l, w_ = le.get(i), we.get(i)
                flip = ""
                if l is not None and w_ is not None:
                    flip = "**YES**" if (l > 0) != (w_ > 0) else "no"
                summary_lines.append("| {} | {} | {} | {} |".format(
                    LEVEL[i],
                    "{:+.1f}".format(l) if l is not None else "-",
                    "{:+.1f}".format(w_) if w_ is not None else "-",
                    flip))
            summary_lines.append("")

    out_md = os.path.join(args.outdir, "stitch_verify_summary.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print("wrote {}".format(out_md))

    out_rerun = os.path.join(args.outdir, "stitch_rerun_list.txt")
    with open(out_rerun, "w", encoding="utf-8") as fh:
        fh.write("# Cells that cannot be cited as-is (mismatch / no-log / no-wandb).\n")
        fh.write("# mismatch -> W&B value is wrong for that cell; no-log -> unverifiable.\n\n")
        fh.write("\n".join(rerun) + ("\n" if rerun else ""))
    print("wrote {} ({} cells)".format(out_rerun, len(rerun)))

    print("\n=== verdict totals ===")
    tot = defaultdict(int)
    for r in all_rows:
        tot[(r["game"], r["verdict"])] += 1
    for g in args.games:
        parts = ["{} {}".format(v, n) for (gg, v), n in sorted(tot.items()) if gg == g]
        print("  {:<14} {}".format(g, ", ".join(parts) or "no cells found"))


if __name__ == "__main__":
    main()
