#!/bin/bash
#SBATCH --job-name=iris_stage3_hrvq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3_hrvq_%j.out
#SBATCH --error=logs/stage3_hrvq_%j.err
# ---------------------------------------------------------------------------
# STAGE 3 - HRVQ-scratch: 3-level residual HRVQ trained from scratch (NO transfer,
# NO freezing). Same game/seed/budget as Stage 1. Purpose: show the 3-level graft
# is sound and the cascade is justified -- the final score should land within
# tolerance of the Stage-1 baseline (see ACCEPTANCE_CRITERIA.md).
#
# Uses the default tokenizer config (hrvq_num_codes=[512,512,512]); no port CLI
# flags needed. The freeze/transfer machinery (port_cli.py, transfer.py) is for
# LATER transfer experiments, not this from-scratch run.
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
  wandb.mode=online wandb.group=stage3_hrvq_scratch wandb.name=stage3_${GAME}_s${SEED}

# ACCEPTANCE: (a) per-level live-code fraction stays non-degenerate early in
# training (Stage-2 gate, see ACCEPTANCE_CRITERIA.md) and (b) final return within
# tolerance of Stage 1. If L1/L2 go dead -> cascade collapse finding (expected to
# be possible: NO EMA / no dead-code revival was ported, by design).
