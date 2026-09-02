#!/bin/bash
# SLURM job array: TRM recursive-dynamics pilot, 2 Atari100k games x 3 seeds.
# Compares against the existing hrvq_baseline runs (isca_scratch.slurm), which
# share every setting except dynamics_core. Tasks are game-major: game =
# GAMES[i/3], seed = i%3.
#
#   sbatch -A <account> experiments/isca_trm.slurm

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE   # overridden by -A on the sbatch line
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive                    # one node per job: per-run wandb service socket
#SBATCH --array=0-5%3                  # 6 tasks, 3 in flight
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/trm_%A_%a.out
#SBATCH -e logs/trm_%A_%a.err

set -u

GAMES=(boxing qbert)

i=$SLURM_ARRAY_TASK_ID
GAME=${GAMES[$((i/3))]}
SEED=$((i%3))

# Environment built by experiments/isca_setup.sh
module load Anaconda3 2>/dev/null || true
source activate "$HOME/echelon-env" 2>/dev/null || conda activate "$HOME/echelon-env"
cd "$HOME/echelon"

export env_name="atari100k-${GAME}" run_name=trm_dynamics
export override_config='{"precision":"bfloat16","dynamics_core":"trm"}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Task-unique temp and wandb directories, so concurrent tasks stay isolated
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME seed=$SEED core=trm | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
python main.py --wandb --seed "$SEED" --eval_period_epoch 5 --keep_last_k 3 \
  --log_figure_period_epoch 9999 \
  --wandb_name "trm/${GAME}/seed${SEED}"
