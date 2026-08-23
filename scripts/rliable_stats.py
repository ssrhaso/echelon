"""rliable statistics over the ECHELON transfer ladders.

Emits, for whatever run set you point it at (the existing pull, or the isca-aug26 tag
once those runs land):

  (a) per-condition IQM of human-normalized score with stratified bootstrap CIs over
      the game x seed matrix, plus probability of improvement vs from-scratch;
  (b) per-game median [IQR] cells, LaTeX-ready;
  (d) a HARD-FAIL guard: any (game, condition) cell whose runs mix host or precision
      aborts and names the offending runs.

Two deliberate choices:

  * Run state is NEVER used as a filter. The crashed/failed labels in this project are
    known-wrong -- runs carrying full 10-point eval curves are labelled crashed. Runs
    are included on evidence (a complete eval curve), not on W&B's status field.

  * The guard is the point, not a warning. Mixing host/precision within a game's cells
    is what invalidated the Ms Pac-Man ladder; silently averaging
    across a split stack is the exact failure this is here to make impossible.
    --exclude-confounded downgrades it to "drop that game and say so".

Usage:
    python scripts/rliable_stats.py                       # from judgement/runs_full.csv
    python scripts/rliable_stats.py --exclude-confounded  # skip known-confounded games
    python scripts/rliable_stats.py --pull --tag isca-aug26   # fresh, new runs
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from rliable import metrics as rl_metrics
except ImportError:
    sys.exit("rliable is required:  pip install rliable")

# Canonical Atari-100k (random, human) reference scores.
REF = {
    "breakout":     (1.7, 30.5),
    "demon_attack": (152.1, 1971.0),
    "ms_pacman":    (307.3, 6951.6),
    "up_n_down":    (533.4, 11693.2),
    "asterix":      (210.0, 8503.3),
}

CONDS = ["scratch", "warmstart", "freeze-L0", "freeze-L01", "freeze-L012",
         "freeze-enc", "freeze-all"]
BASELINE = "scratch"


def hns(game, v):
    rnd, hum = REF[game]
    return (v - rnd) / (hum - rnd)


# ---------------------------------------------------------------- loading

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r["game"] not in REF or r["condition"] not in CONDS:
            continue
        # Evidence-based inclusion: a canonical run with a complete eval curve.
        # NOTE: r["state"] is deliberately not consulted.
        if r.get("canonical") != "True" or r.get("complete") != "True":
            continue
        if not r.get("last5_return"):
            continue
        out.append({
            "run_id": r["run_id"], "name": r["name"], "game": r["game"],
            "condition": r["condition"], "seed": int(r["seed"]),
            "score": float(r["last5_return"]),
            "host": r.get("gpu") or "?",          # pre-isca-aug26 proxy for host/stack
            "precision": r.get("precision") or "?",
            "precision_source": "inferred(heuristic)",
        })
    return out


def load_wandb(tag, entity, project):
    """Fresh pull. Prefers the provenance now written into wandb.config by
    nnet/models/model.py (precision / hostname / gpu_model), which for the older runs
    had to be reconstructed from stdout."""
    import re
    import wandb

    PATS = [
        (re.compile(r"^hrvq/atari100k/(?P<g>[a-z_]+)/seed(?P<s>\d+)$"), "scratch"),
        (re.compile(r"^baseline/(?P<g>[a-z_]+)/seed(?P<s>\d+)$"), "scratch"),
        (re.compile(r"^transfer/pongto/(?P<g>[a-z_]+)/(?P<c>[a-zA-Z0-9-]+)/seed(?P<s>\d+)$"), None),
    ]
    EVAL_RE = re.compile(r"(?i)^evaluation-epoch[\\/]\d+[\\/]score$")

    api = wandb.Api(timeout=300)
    runs = api.runs("{}/{}".format(entity, project), per_page=500)
    out = []
    for r in runs:
        game = cond = seed = None
        for pat, fixed in PATS:
            m = pat.match(r.name or "")
            if m:
                game = m.group("g")
                seed = int(m.group("s"))
                cond = fixed or m.group("c")
                break
        if game not in REF or cond not in CONDS:
            continue
        if tag and tag not in (r.tags or []):
            continue
        summary = r.summary._json_dict
        keys = [k for k in summary if EVAL_RE.match(k)]
        if not keys:
            continue
        curve = []
        for row in r.scan_history(keys=[keys[0]], page_size=2000):
            v = row.get(keys[0])
            if v is not None:
                curve.append(float(v))
        if len(curve) < 10:
            continue
        cfg = dict(r.config or {})
        prec = cfg.get("precision")
        out.append({
            "run_id": r.id, "name": r.name, "game": game, "condition": cond,
            "seed": seed, "score": float(np.mean(curve[-5:])),
            "host": cfg.get("hostname") or cfg.get("gpu_model") or "?",
            "precision": str(prec) if prec else "?",
            "precision_source": "wandb.config" if prec else "missing",
        })
        print("  pulled {:<52} last5={:.1f}".format(r.name, out[-1]["score"]), flush=True)
    return out


# ---------------------------------------------------------------- guard

def check_stacks(runs, exclude):
    """Hard-fail on any (game, condition) cell mixing host or precision.

    Also reports game-level splits: the Ms Pac-Man failure was not a mixed *cell* but
    a mixed *game* -- a Windows-hosted scratch baseline compared against a cloud-hosted
    ladder. A per-cell check alone would have passed it.
    """
    cell = defaultdict(list)
    for r in runs:
        cell[(r["game"], r["condition"])].append(r)

    cell_bad = []
    for key, grp in sorted(cell.items()):
        combos = {(r["host"], r["precision"]) for r in grp}
        if len(combos) > 1:
            cell_bad.append((key, grp, combos))

    game = defaultdict(list)
    for r in runs:
        game[r["game"]].append(r)
    game_bad = []
    for g, grp in sorted(game.items()):
        combos = {(r["host"], r["precision"]) for r in grp}
        if len(combos) > 1:
            game_bad.append((g, grp, combos))

    if not cell_bad and not game_bad:
        print("stack guard: OK -- every game's cells share one (host, precision).\n")
        return runs, []

    lines = []
    for (g, c), grp, combos in cell_bad:
        lines.append("  CELL {}/{} mixes {} stacks:".format(g, c, len(combos)))
        for r in sorted(grp, key=lambda x: x["seed"]):
            lines.append("      seed{} {:<10} host={:<28} precision={:<10} ({})".format(
                r["seed"], r["run_id"], r["host"], r["precision"], r["precision_source"]))
    for g, grp, combos in game_bad:
        if any(gg == g for (gg, _), _, _ in cell_bad):
            pass
        lines.append("  GAME {} spans {} distinct (host, precision) combinations:".format(
            g, len(combos)))
        for combo in sorted(combos):
            who = [r for r in grp if (r["host"], r["precision"]) == combo]
            conds = sorted({r["condition"] for r in who})
            lines.append("      {:<28} {:<10}  conditions: {}".format(
                combo[0], combo[1], ", ".join(conds)))

    msg = ("HOST/PRECISION MIXING DETECTED -- aggregation refused.\n"
           + "\n".join(lines)
           + "\n\nEach game must be one complete ladder on one stack. Re-run the game "
             "on a single stack, or pass --exclude-confounded to drop it and continue.")
    bad_games = sorted({g for (g, _), _, _ in cell_bad} | {g for g, _, _ in game_bad})
    if not exclude:
        sys.exit("\nERROR: " + msg)
    print("\nWARNING: " + msg)
    print("\n--exclude-confounded: dropping {}\n".format(", ".join(bad_games)))
    return [r for r in runs if r["game"] not in bad_games], bad_games


# ---------------------------------------------------------------- statistics

def build_matrix(runs):
    """condition -> {game: [hns scores, one per seed]}"""
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        out[r["condition"]][r["game"]].append(hns(r["game"], r["score"]))
    return out


def iqm(values):
    a = np.asarray(values, dtype=float)
    return float(rl_metrics.aggregate_iqm(a.reshape(-1, 1)))


def stratified_bootstrap(per_game, stat, reps, rng):
    """Resample seeds WITH replacement independently within each game, then pool.

    Stratifying by game is what keeps a game with many seeds from dominating the
    resample, and it tolerates the ragged seed counts (n=2..6) this project has --
    rliable's own StratifiedBootstrap needs a rectangular matrix.
    """
    games = sorted(per_game)
    draws = []
    for _ in range(reps):
        pooled = []
        for g in games:
            vals = per_game[g]
            idx = rng.integers(0, len(vals), size=len(vals))
            pooled.extend(np.asarray(vals)[idx])
        draws.append(stat(pooled))
    return np.percentile(draws, [2.5, 97.5])


def prob_improvement(cond_per_game, base_per_game):
    """P(condition > scratch), averaged over the games both cover."""
    games = sorted(set(cond_per_game) & set(base_per_game))
    if not games:
        return None, []
    ps = []
    for g in games:
        x = np.asarray(cond_per_game[g], dtype=float)
        y = np.asarray(base_per_game[g], dtype=float)
        wins = (x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()
        ps.append(wins / (len(x) * len(y)))
    return float(np.mean(ps)), games


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="judgement/runs_full.csv")
    ap.add_argument("--pull", action="store_true", help="pull fresh from W&B instead")
    ap.add_argument("--tag", default=None, help="with --pull, restrict to this W&B tag")
    ap.add_argument("--entity", default="haso-university-of-the-west-of-england")
    ap.add_argument("--project", default="nnet")
    ap.add_argument("--reps", type=int, default=10000, help="bootstrap replications")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-confounded", action="store_true",
                    help="drop host/precision-mixed games instead of aborting")
    ap.add_argument("--outdir", default="judgement")
    args = ap.parse_args()

    runs = load_wandb(args.tag, args.entity, args.project) if args.pull \
        else load_csv(args.csv)
    if not runs:
        sys.exit("no runs loaded")
    print("loaded {} runs across {} games\n".format(
        len(runs), len({r["game"] for r in runs})))

    runs, dropped = check_stacks(runs, args.exclude_confounded)
    if not runs:
        sys.exit("every game was dropped by the stack guard; nothing to aggregate")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mat = build_matrix(runs)
    base = mat.get(BASELINE, {})

    # ---- (a) IQM + CI + probability of improvement ----
    rows = []
    print("=" * 92)
    print("Per-condition IQM of human-normalized score  (stratified bootstrap, {} reps)"
          .format(args.reps))
    print("=" * 92)
    print("{:<12} {:>3} {:>4}  {:>8}  {:>18}   {:>10}".format(
        "condition", "G", "runs", "IQM", "95% CI", "P(>scratch)"))
    for c in CONDS:
        per_game = mat.get(c)
        if not per_game:
            continue
        pooled = [v for g in per_game for v in per_game[g]]
        point = iqm(pooled)
        lo, hi = stratified_bootstrap(per_game, iqm, args.reps, rng)
        poi, poi_games = (None, []) if c == BASELINE else prob_improvement(per_game, base)
        rows.append({
            "condition": c, "n_games": len(per_game), "n_runs": len(pooled),
            "iqm_hns": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "prob_improvement_vs_scratch": "" if poi is None else round(poi, 4),
            "games": "|".join(sorted(per_game)),
        })
        print("{:<12} {:>3} {:>4}  {:>8.4f}  [{:>7.4f}, {:>7.4f}]   {:>10}".format(
            c, len(per_game), len(pooled), point, lo, hi,
            "-" if poi is None else "{:.3f}".format(poi)))
    print()

    with open(os.path.join(args.outdir, "rliable_iqm.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote {}/rliable_iqm.csv".format(args.outdir))

    # ---- (b) per-game median [IQR] ----
    per_cell = defaultdict(list)
    for r in runs:
        per_cell[(r["game"], r["condition"])].append(r["score"])
    games = sorted({r["game"] for r in runs})

    tex = ["% Generated by scripts/rliable_stats.py -- raw return, median [IQR].",
           "% \\rcell{median}{iqr-low}{iqr-high} is defined here so the table is "
           "self-contained.",
           "\\providecommand{\\rcell}[3]{$#1$\\,\\scriptsize[#2,\\,#3]}",
           "\\begin{tabular}{l" + "c" * len(games) + "}",
           "\\toprule",
           "Condition & " + " & ".join(g.replace("_", "\\_") for g in games) + " \\\\",
           "\\midrule"]
    csv_rows = []
    print("\nPer-game median [IQR] of raw return")
    print("-" * 92)
    hdr = "{:<12}".format("condition") + "".join("{:>22}".format(g) for g in games)
    print(hdr)
    for c in CONDS:
        cells, line = [], "{:<12}".format(c)
        for g in games:
            vals = per_cell.get((g, c))
            if not vals:
                cells.append("--")
                line += "{:>22}".format("-")
                continue
            a = np.asarray(vals, dtype=float)
            med = float(np.median(a))
            q1, q3 = np.percentile(a, [25, 75])
            cells.append("\\rcell{{{:.0f}}}{{{:.0f}}}{{{:.0f}}}".format(med, q1, q3))
            line += "{:>22}".format("{:.0f} [{:.0f},{:.0f}] n={}".format(
                med, q1, q3, len(a)))
            csv_rows.append({"game": g, "condition": c, "n": len(a),
                             "median": round(med, 2), "iqr_lo": round(float(q1), 2),
                             "iqr_hi": round(float(q3), 2)})
        print(line)
        tex.append(c + " & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]

    with open(os.path.join(args.outdir, "per_game_median_iqr.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["game", "condition", "n", "median",
                                           "iqr_lo", "iqr_hi"])
        w.writeheader()
        w.writerows(csv_rows)
    with open(os.path.join(args.outdir, "per_game_median_iqr.tex"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")
    print("\nwrote {}/per_game_median_iqr.csv and .tex".format(args.outdir))

    json.dump({"iqm": rows, "per_game": csv_rows, "dropped_games": dropped,
               "reps": args.reps, "tag": args.tag,
               "note": "run.state never used as a filter; crashed/failed labels are "
                       "known-wrong in this project"},
              open(os.path.join(args.outdir, "rliable_stats.json"), "w",
                   encoding="utf-8"), indent=2)
    print("wrote {}/rliable_stats.json".format(args.outdir))


if __name__ == "__main__":
    main()
