# W&B results snapshot

Pulled from `haso-university-of-the-west-of-england/nnet` via `pull_wandb.py` on 2026-04-21.

## Contents

```
results/
├── summary.csv                          all 31 runs, one row each
├── pong_baselines/                      6 seeds (hrvq/atari100k/pong/*)
│   ├── scores.csv                       wide: seed × score_1..10
│   ├── trajectories.json                {seed: [[env_step, score], ...]}
│   └── seed0.json ... seed5.json        per-run dump
├── breakout_baselines/                  6 seeds (hrvq/atari100k/breakout/*)
│   └── (same layout as pong_baselines)
├── transfer_pong_to_breakout/           18 runs (transfer/pongto/breakout/*)
│   ├── condition_stats.csv              per-condition mean/std across seeds
│   ├── all_scores.csv                   every (condition, seed) in one table
│   ├── trajectories.json                {condition/seed: [...]}
│   └── <condition>/                     one folder per freezing condition
│       ├── scores.csv
│       └── seed0.json, seed1.json, seed2.json
└── archive/                             1 retired run(s)
```

## Schema notes

- Trajectories are `[env_step, score]` pairs pulled from `Evaluation-epoch\0\score`
  (older runs: `Evaluation-epoch/0/score`). `env_step` comes from W&B `_step`.
- `mean_last5` = mean of the last 5 eval points per run.
- `summary.csv` state values: `finished`, `failed`, `crashed`. `failed` pong seeds
  still have eval data logged before the failure — inspect before using.

## Regenerate

```
python pull_wandb.py
```
Rebuilds the presentation layer. Per-run fetches are cached in `.cache/` so
reruns are fast and resumable — delete `.cache/<run_id>.json` to force a refresh.
