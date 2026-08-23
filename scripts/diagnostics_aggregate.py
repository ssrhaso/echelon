"""Aggregate per-checkpoint diagnostics into per-condition paper tables.

Consumes the results.json files written by scripts/run_diagnostics.py (one per
job) and collapses the per-seed rows into per-(game, condition) summaries with
mean +/- std across seeds. Emits:

  - aggregated.csv   long format, one row per (game, condition, level)
  - aggregated.json  same data, nested
  - tables.tex       paste-ready LaTeX fragments for the paper:
                       * Active-Codes |A^(l)| columns        (Table: results_main)
                       * Jaccard / quant-error / drift row    (Table: results_diagnostics)

Per-level diagnostics covered: active codes |A^(l)|, usage %, perplexity,
residual quant error E^(l) (D2), Jaccard-vs-source J^(l) (D1), and codebook
drift vs source (matched cosine / identity rate). The encoder-feature drift
rho_enc (D3) is not yet computed upstream and is emitted as a TODO marker.

Usage:
    python scripts/diagnostics_aggregate.py --root results/diagnostics
    python scripts/diagnostics_aggregate.py --results a/results.json b/results.json
    python scripts/diagnostics_aggregate.py --root results/diagnostics --source_game pong
"""

import argparse
import csv
import glob
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

# Row order for the LaTeX tables (skips conditions absent from the data).
CONDITION_ORDER = [
    "scratch", "warmstart", "freeze-L0", "freeze-L01",
    "freeze-L012", "freeze-enc", "freeze-all",
]
NUM_LEVELS = 3


def load_results(paths):
    """Yield (job_tag, parsed_results) for each results.json path."""
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        yield data.get("tag") or os.path.basename(os.path.dirname(p)), data


def _canon_game(meta, fallback=None):
    return meta.get("game") or fallback


def collect_rows(results_iter, source_game):
    """Group per-seed rows by (game, condition); attach per-level Jaccard-vs-source.

    Jaccard-vs-source is pulled from each job's pairwise matrix: for every target
    row, the entries linking it to a source-game row give J^(l), which we then
    average across the seeds of each condition.
    """
    groups = defaultdict(lambda: defaultdict(list))  # (game,cond) -> metric -> [per-seed]

    for _tag, data in results_iter:
        rows = data.get("rows", [])
        pairwise = data.get("pairwise", [])

        # Map label -> row meta so we can tell source rows from target rows.
        label_game = {r["label"]: _canon_game(r["meta"]) for r in rows}
        source_labels = {lab for lab, g in label_game.items()
                         if (g or "").lower() == source_game.lower()}

        # label -> {level -> jaccard vs source} (mean over source rows present).
        jacc_vs_src = defaultdict(lambda: defaultdict(list))
        for pe in pairwise:
            a, b = pe["a"], pe["b"]
            a_src, b_src = a in source_labels, b in source_labels
            if a_src == b_src:
                continue  # both source or neither -> not a target-vs-source pair
            target = b if a_src else a
            for lv in pe["levels"]:
                jacc_vs_src[target][lv["level"]].append(lv["jaccard"])

        for r in rows:
            game = _canon_game(r["meta"])
            cond = r["meta"].get("condition")
            if game is None or cond is None:
                continue
            if (game or "").lower() == source_game.lower() and cond == "scratch":
                # The source row itself is a reference, not a target condition.
                continue
            key = (game, cond)
            g = groups[key]
            for lv in range(NUM_LEVELS):
                u = r["usage"][lv]
                g[f"active_l{lv}"].append(u["active"])
                g[f"usage_pct_l{lv}"].append(u["usage"] * 100)
                g[f"perplexity_l{lv}"].append(u["perplexity"])
                g[f"residual_err_l{lv}"].append(r["residual_errors"][lv])
                g[f"recon_mse_l{lv}"].append(r["recon"][lv]["mse"])
                if r.get("drift"):
                    g[f"drift_cos_l{lv}"].append(r["drift"][lv]["mean_matched_cos"])
                    g[f"drift_identity_l{lv}"].append(r["drift"][lv]["identity_rate"])
                js = jacc_vs_src.get(r["label"], {}).get(lv)
                if js:
                    g[f"jaccard_src_l{lv}"].append(float(np.mean(js)))
            if r.get("enc_drift"):  # rho_enc (D3): one scalar per checkpoint
                g["rho_enc"].append(r["enc_drift"]["rho_enc_mean"])
            g["_seeds"].append(r["meta"].get("seed"))
    return groups


def _ms(values):
    """(mean, std, n) for a list, or (nan, nan, 0) if empty."""
    if not values:
        return float("nan"), float("nan"), 0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def summarise(groups):
    """(game,cond) -> {metric -> (mean,std,n)}, ordered by CONDITION_ORDER."""
    out = OrderedDict()
    keys = sorted(groups.keys(), key=lambda k: (
        k[0], CONDITION_ORDER.index(k[1]) if k[1] in CONDITION_ORDER else 99))
    for key in keys:
        g = groups[key]
        seeds = [s for s in g.get("_seeds", []) if s is not None]
        stats = {"n_seeds": len(g.get("_seeds", [])), "seeds": sorted(set(seeds))}
        for metric, vals in g.items():
            if metric == "_seeds":
                continue
            stats[metric] = _ms(vals)
        out[key] = stats
    return out


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------

def write_csv(summary, path):
    metrics = []
    for stats in summary.values():
        metrics = [m for m in stats if m not in ("n_seeds", "seeds")]
        break
    base_metrics = sorted({m.rsplit("_l", 1)[0] for m in metrics})
    fields = ["game", "condition", "n_seeds", "level"] + \
             [f"{m}_mean" for m in base_metrics] + [f"{m}_std" for m in base_metrics]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (game, cond), stats in summary.items():
            for lv in range(NUM_LEVELS):
                row = {"game": game, "condition": cond,
                       "n_seeds": stats["n_seeds"], "level": lv}
                for m in base_metrics:
                    mean, std, _ = stats.get(f"{m}_l{lv}", (float("nan"),) * 3)
                    row[f"{m}_mean"] = mean
                    row[f"{m}_std"] = std
                w.writerow(row)
    print(f"[write] {path}")


def write_json(summary, path):
    serial = {f"{g}/{c}": v for (g, c), v in summary.items()}
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"[write] {path}")


def _cell(mean, std, n=None, prec=0, with_std=True):
    # `n` is accepted so callers can splat a (mean, std, n) tuple directly.
    if mean != mean:  # NaN
        return r"\tbd"
    if with_std and std == std:
        return f"{mean:.{prec}f}$\\pm${std:.{prec}f}"
    return f"{mean:.{prec}f}"


def write_latex(summary, path, source_game):
    games = sorted({g for g, _ in summary})
    lines = []
    lines.append("% Auto-generated by scripts/diagnostics_aggregate.py")
    lines.append("% Active-code counts |A^(l)| for the main results table (mean$\\pm$std over seeds).")
    for game in games:
        lines.append(f"% --- {game}: active codes |A^(l)| (L0 / L1 / L2) ---")
        for (g, cond), stats in summary.items():
            if g != game:
                continue
            cells = [_cell(*stats.get(f"active_l{lv}", (float('nan'),) * 3), prec=0)
                     for lv in range(NUM_LEVELS)]
            lines.append(f"\\textsc{{{cond}}} & " + " & ".join(cells)
                         + f"  % n={stats['n_seeds']}")
        lines.append("")

    lines.append("% Diagnostics table: Jaccard J^(l) vs source, quant error E^(l), drift, rho_enc.")
    lines.append("% rho_enc (D3) requires a --source_ckpt in the run; \\tbd if absent.")
    for game in games:
        lines.append(f"% --- {game}: Jaccard(L0/L1/L2) | QuantErr(L0/L1/L2) | drift-cos | rho_enc ---")
        for (g, cond), stats in summary.items():
            if g != game:
                continue
            jacc = [_cell(*stats.get(f"jaccard_src_l{lv}", (float('nan'),) * 3), prec=3)
                    for lv in range(NUM_LEVELS)]
            qerr = [_cell(*stats.get(f"residual_err_l{lv}", (float('nan'),) * 3), prec=4)
                    for lv in range(NUM_LEVELS)]
            cos_mean, cos_std, _ = stats.get("drift_cos_l0", (float("nan"),) * 3)
            drift = _cell(cos_mean, cos_std, prec=3)
            rho = _cell(*stats.get("rho_enc", (float("nan"),) * 3), prec=3)
            lines.append(f"\\textsc{{{cond}}} & " + " & ".join(jacc)
                         + " & " + " & ".join(qerr)
                         + f" & {drift} & {rho}  % n={stats['n_seeds']}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[write] {path}")


def print_console(summary):
    print("\n## Per-condition diagnostics (mean +/- std over seeds)\n")
    for (game, cond), stats in summary.items():
        seeds = stats["seeds"]
        rho = stats.get("rho_enc")
        rstr = f"  rho_enc={rho[0]:.3f}" if rho and rho[0] == rho[0] else ""
        print(f"  {game}/{cond}  (n={stats['n_seeds']}, seeds={seeds}){rstr}")
        for lv in range(NUM_LEVELS):
            a = stats.get(f"active_l{lv}", (float('nan'),) * 3)
            p = stats.get(f"perplexity_l{lv}", (float('nan'),) * 3)
            e = stats.get(f"residual_err_l{lv}", (float('nan'),) * 3)
            j = stats.get(f"jaccard_src_l{lv}")
            jstr = f"  Jacc-src={j[0]:.3f}" if j else ""
            print(f"    L{lv}: active={a[0]:5.1f}+/-{a[1]:4.1f}"
                  f"  perp={p[0]:6.2f}  resid={e[0]:.4f}{jstr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="Directory containing <job>/results.json files (globbed).")
    ap.add_argument("--results", nargs="*", default=None,
                    help="Explicit results.json path(s); combined with --root.")
    ap.add_argument("--source_game", default="pong",
                    help="Canonical game name treated as the transfer source "
                         "(its rows define the Jaccard-vs-source reference).")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write aggregated.{csv,json} + tables.tex. "
                         "Defaults to --root, else current directory.")
    args = ap.parse_args()

    paths = list(args.results or [])
    if args.root:
        paths += sorted(glob.glob(os.path.join(args.root, "*", "results.json")))
        paths += sorted(glob.glob(os.path.join(args.root, "results.json")))
    # De-dupe on the resolved path so the same file passed via both --root and
    # --results (with different separators) is not counted twice.
    seen, deduped = set(), []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    paths = deduped
    if not paths:
        ap.error("no results.json found; pass --root and/or --results")

    print(f"== Aggregating {len(paths)} results file(s) ==")
    for p in paths:
        print(f"  - {p}")

    groups = collect_rows(load_results(paths), args.source_game)
    summary = summarise(groups)
    if not summary:
        ap.error("no (game, condition) groups found in the results files")

    print_console(summary)

    out_dir = args.output_dir or args.root or "."
    os.makedirs(out_dir, exist_ok=True)
    write_csv(summary, os.path.join(out_dir, "aggregated.csv"))
    write_json(summary, os.path.join(out_dir, "aggregated.json"))
    write_latex(summary, os.path.join(out_dir, "tables.tex"), args.source_game)


if __name__ == "__main__":
    main()
