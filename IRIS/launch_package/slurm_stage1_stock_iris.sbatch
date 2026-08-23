#!/bin/bash
#SBATCH --job-name=iris_stage1_stock
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage1_stock_%j.out
#SBATCH --error=logs/stage1_stock_%j.err
# ---------------------------------------------------------------------------
# STAGE 1 - stock-IRIS baseline (single-level tokenizer), 1 game, 1 seed, full budget.
# Purpose: validate the env/ARM build and establish the published-IRIS ballpark.
# This runs the SINGLE-LEVEL configuration of the ported tokenizer (hrvq_num_codes=[512]),
# which exercises the identical encode/loss/decode code path as stock IRIS. For a
# bit-for-bit pristine baseline instead, `git checkout` the upstream clone HEAD
# (tag `upstream-iris`, the first commit in this repo) and run plain `src/main.py`.
# ---------------------------------------------------------------------------
set -euo pipefail
module load cuda || true
source .venv_gpu/bin/activate

GAME=${GAME:-BreakoutNoFrameskip-v4}
SEED=${SEED:-0}

python src/main.py \
  env.train.id=${GAME} \
  common.device=cuda:0 \
  common.seed=${SEED} \
  common.epochs=600 \
  tokenizer.hrvq_num_codes='[512]' \
  tokenizer.hrvq_commitment_costs='[1.0]' \
  wandb.mode=online wandb.group=stage1_stock wandb.name=stage1_${GAME}_s${SEED}

# ACCEPTANCE: final return in the published-Breakout ballpark (see ACCEPTANCE_CRITERIA.md).
# A score far below ballpark => env/ARM problem, NOT a model problem.
