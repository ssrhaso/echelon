"""
Export Evaluation-epoch/0/* metrics from W&B to CSV (ALL games, ALL conditions, ALL periods).

Project: haso-university-of-the-west-of-england/nnet

Supersedes the old breakout-only export. Fixes:
 - classifies baseline/<game>/seed runs (were silently dropped before)
 - records `game` as a first-class column
 - handles BOTH key separators: 'Evaluation-epoch\\0\\score' and 'Evaluation-epoch/0/score'
 - pulls the full per-epoch curve (all eval points), not just the summary

Outputs (under wandb_export/):
  eval_epoch_long.csv     long format: run_id, run_name, game, family, condition, seed, state, step, epoch_idx, metric, value
  eval_epoch_wide.csv     one row per (run, eval point), one column per Evaluation-epoch metric
  eval_epoch_by_period.csv pivot: one row per run, one column per eval period index (score only) -> easy table source
  eval_epoch_summary.csv  per-run final/best/mean/last5 score
  runs_metadata.csv       run discovery + per-run eval point counts
  export_log.txt          run log

W&B API quirk: score and global_step are logged in separate wandb.log() calls,
so each metric is fetched independently and merged on global_step.
"""

import re
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import wandb

ENTITY  = "haso-university-of-the-west-of-england"
PROJECT = "nnet"

OUT_DIR  = Path(__file__).resolve().parent.parent / "wandb_export"
CACHE    = OUT_DIR / "cache"
LOG_PATH = OUT_DIR / "export_log.txt"

OUT_DIR.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

# Match an Evaluation-epoch metric with EITHER separator: \0\name  or  /0/name
EVAL_RE = re.compile(r"^Evaluation-epoch[\\/]0[\\/](.+)$")

# name -> (game, family, condition, seed)
NAME_PATTERNS = [
    (re.compile(r"^hrvq/atari100k/(?P<g>[a-z_]+)/seed(?P<s>\d+)$"),                 "scratch",  "scratch"),
    (re.compile(r"^baseline/(?P<g>[a-z_]+)/seed(?P<s>\d+)$"),                       "scratch",  "scratch"),
    (re.compile(r"^transfer/pongto/(?P<g>[a-z_]+)/warmstart/seed(?P<s>\d+)$"),      "transfer", "warmstart"),
    (re.compile(r"^transfer/pongto/(?P<g>[a-z_]+)/freeze-L0/seed(?P<s>\d+)$"),      "transfer", "freeze-L0"),
    (re.compile(r"^transfer/pongto/(?P<g>[a-z_]+)/freeze-L01/seed(?P<s>\d+)$"),     "transfer", "freeze-L01"),
    (re.compile(r"^transfer/pongto/(?P<g>[a-z_]+)/freeze-L012/seed(?P<s>\d+)$"),    "transfer", "freeze-L012"),
    (re.compile(r"^transfer/pongto/(?P<g>[a-z_]+)/freeze-enc/seed(?P<s>\d+)$"),     "transfer", "freeze-enc"),
    (re.compile(r"^transfer/pongto/(?P<g>[a-z_]+)/freeze-all/seed(?P<s>\d+)$"),     "transfer", "freeze-all"),
]


def classify(name: str):
    for pat, family, cond in NAME_PATTERNS:
        m = pat.match(name or "")
        if m:
            return m.group("g"), family, cond, int(m.group("s"))
    return None, None, None, None


def log(msg, fp):
    line = msg.rstrip()
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def fetch_metric(run, metric_key):
    """Pull (global_step, value) pairs for one metric via scan_history."""
    out = []
    for row in run.scan_history(keys=[metric_key, "global_step"], page_size=2000):
        v = row.get(metric_key)
        s = row.get("global_step")
        if v is None or s is None:
            continue
        out.append((int(s), float(v)))
    out.sort(key=lambda x: x[0])
    return out


def main():
    log_fp = open(LOG_PATH, "w", encoding="utf-8")
    log(f"=== ECHELON eval export (all games) - {datetime.now().isoformat(timespec='seconds')} ===", log_fp)
    log(f"project: {ENTITY}/{PROJECT}", log_fp)
    log(f"out: {OUT_DIR}", log_fp)

    api = wandb.Api(timeout=240)
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    log(f"\ntotal runs in project: {len(all_runs)}", log_fp)

    classified = []
    for r in all_runs:
        game, fam, cond, seed = classify(r.name)
        if game is None:
            continue
        classified.append((r, game, fam, cond, seed))
    log(f"matching naming scheme: {len(classified)}", log_fp)

    grouped = defaultdict(list)
    for r, game, fam, cond, seed in classified:
        grouped[(game, cond)].append(seed)
    log("\nRuns by (game, condition):", log_fp)
    for (game, cond), seeds in sorted(grouped.items()):
        log(f"  {game:<13} {cond:<12} n={len(seeds):>2}  seeds={sorted(seeds)}", log_fp)

    long_rows  = []
    wide_rows  = []
    meta_rows  = []
    final_rows = []

    log("\npulling per-run eval histories...", log_fp)
    for i, (r, game, fam, cond, seed) in enumerate(classified, 1):
        # discover this run's eval-epoch metric keys (handles either separator)
        try:
            eval_keys = sorted(k for k in r.summary.keys() if EVAL_RE.match(k))
        except Exception as e:
            log(f"  WARN summary {r.id}: {e}", log_fp)
            eval_keys = []

        # cache key includes the resolved metric keys so stale caches are invalidated
        cache_pkl = CACHE / f"{r.id}.pkl"
        per_metric = None
        source = None
        if cache_pkl.exists():
            try:
                with open(cache_pkl, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("keys") == eval_keys and blob.get("state") == r.state:
                    per_metric = blob["data"]
                    source = "cache"
            except Exception:
                per_metric = None

        if per_metric is None:
            per_metric = {}
            try:
                for k in eval_keys:
                    per_metric[k] = fetch_metric(r, k)
                source = "api"
                with open(cache_pkl, "wb") as f:
                    pickle.dump({"keys": eval_keys, "state": r.state, "data": per_metric}, f)
            except Exception as e:
                log(f"  [{i}/{len(classified)}] {r.id} ERROR: {e}", log_fp)
                continue

        n_eval = sum(len(v) for v in per_metric.values())
        try:
            runtime_h = float(r.summary.get("_runtime", 0)) / 3600.0
        except Exception:
            runtime_h = None

        meta_rows.append({
            "run_id": r.id, "run_name": r.name, "game": game, "family": fam,
            "condition": cond, "seed": seed, "state": r.state,
            "created_at": r.created_at, "runtime_hours": runtime_h,
            "n_eval_points": int(n_eval), "source": source,
        })
        log(f"  [{i:>3}/{len(classified)}] {r.id}  {game:<13} {cond:<12} seed={seed} "
            f"eval_pts={int(n_eval):>4} ({source})", log_fp)

        # long-format rows + per-period index (period within this run, ordered by step)
        score_pairs = []
        for k, pairs in per_metric.items():
            metric_name = EVAL_RE.match(k).group(1)
            for period_idx, (step, val) in enumerate(pairs):
                long_rows.append({
                    "run_id": r.id, "run_name": r.name, "game": game, "family": fam,
                    "condition": cond, "seed": seed, "state": r.state,
                    "step": step, "epoch_idx": period_idx, "metric": metric_name, "value": val,
                })
            if metric_name == "score":
                score_pairs = pairs

        # wide rows: one per (run, step) with a column per metric
        by_step = defaultdict(dict)
        for k, pairs in per_metric.items():
            metric_name = EVAL_RE.match(k).group(1)
            for step, val in pairs:
                by_step[step][metric_name] = val
        for period_idx, step in enumerate(sorted(by_step)):
            row = {"run_id": r.id, "run_name": r.name, "game": game, "family": fam,
                   "condition": cond, "seed": seed, "state": r.state,
                   "step": step, "epoch_idx": period_idx}
            row.update(by_step[step])
            wide_rows.append(row)

        if score_pairs:
            scores = pd.Series([v for _, v in score_pairs])
            last5 = scores.tail(5)
            final_rows.append({
                "run_id": r.id, "run_name": r.name, "game": game, "family": fam,
                "condition": cond, "seed": seed, "state": r.state,
                "n_evals": int(len(scores)),
                "final_step": int(score_pairs[-1][0]),
                "final_score": float(scores.iloc[-1]),
                "best_score": float(scores.max()),
                "mean_score": float(scores.mean()),
                "last5_mean": float(last5.mean()),
                "last5_std": float(last5.std()) if len(last5) > 1 else 0.0,
            })

    # ---- write outputs ----
    pd.DataFrame(meta_rows).sort_values(["game", "condition", "seed"]).to_csv(
        OUT_DIR / "runs_metadata.csv", index=False)

    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df = long_df.sort_values(["game", "condition", "seed", "run_id", "metric", "step"])
    long_df.to_csv(OUT_DIR / "eval_epoch_long.csv", index=False)

    wide_df = pd.DataFrame(wide_rows)
    if not wide_df.empty:
        wide_df = wide_df.sort_values(["game", "condition", "seed", "run_id", "step"])
    wide_df.to_csv(OUT_DIR / "eval_epoch_wide.csv", index=False)

    # by-period pivot (score only): rows = run, cols = period_0..period_N
    score_long = long_df[long_df["metric"] == "score"] if not long_df.empty else long_df
    if not score_long.empty:
        by_period = (score_long
                     .pivot_table(index=["game", "family", "condition", "seed", "run_id", "run_name", "state"],
                                  columns="epoch_idx", values="value", aggfunc="first")
                     .reset_index())
        by_period.columns = [f"period_{c}" if isinstance(c, int) else c for c in by_period.columns]
        by_period = by_period.sort_values(["game", "condition", "seed"])
        by_period.to_csv(OUT_DIR / "eval_epoch_by_period.csv", index=False)

    pd.DataFrame(final_rows).sort_values(["game", "condition", "seed"]).to_csv(
        OUT_DIR / "eval_epoch_summary.csv", index=False)

    log("\n=== outputs ===", log_fp)
    for name in ["eval_epoch_long.csv", "eval_epoch_wide.csv",
                 "eval_epoch_by_period.csv", "eval_epoch_summary.csv", "runs_metadata.csv"]:
        p = OUT_DIR / name
        if p.exists():
            log(f"  {p.name:<28} {p.stat().st_size/1024:>8.1f} KB", log_fp)
    log(f"\ntotal eval points exported: {len(long_rows)}", log_fp)
    log_fp.close()


if __name__ == "__main__":
    main()
