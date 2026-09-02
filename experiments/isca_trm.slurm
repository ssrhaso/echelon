#!/bin/bash
# SLURM job array: TRM recursive-dynamics pilot, 2 Atari100k games x 3 seeds.
# Compares against the completed scratch baselines (runs_inventory.csv:
# ms_pacman 6 seeds last5~823, breakout 6 seeds last5~37), which share every
# setting except dynamics_core. Tasks are game-major: game = GAMES[i/3],
# seed = i%3.
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

GAMES=(ms_pacman breakout)

i=$SLURM_ARRAY_TASK_ID
GAME=${GAMES[$((i/3))]}
SEED=$((i%3))

# Environment built by experiments/isca_setup.sh. Call its python directly and
# fail loudly — the silent fallback to the system Anaconda is how job 2365050
# died (activation "succeeds" even when the env is gutted).
PY="$HOME/echelon-env/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY missing — rebuild with experiments/isca_setup.sh" >&2
    exit 1
fi
cd "$HOME/echelon"

# float32 to match the completed scratch baselines (the atari100k default);
# mixing precision across a comparison is what invalidated the Ms Pac-Man ladder
export env_name="atari100k-${GAME}" run_name=trm_dynamics
export override_config='{"dynamics_core":"trm"}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Task-unique temp and wandb directories, so concurrent tasks stay isolated
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME seed=$SEED core=trm | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
"$PY" main.py --wandb --seed "$SEED" --eval_period_epoch 5 --keep_last_k 3 \
  --log_figure_period_epoch 9999 \
  --wandb_name "trm/${GAME}/seed${SEED}"
