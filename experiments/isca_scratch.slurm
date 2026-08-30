#!/bin/bash
# SLURM job array: HRVQ scratch baselines, 20 Atari100k games x 3 seeds.
# Tasks are game-major, so games complete in order: game = GAMES[i/3], seed = i%3.
#
#   sbatch -A <account> experiments/isca_scratch.slurm

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE   # overridden by -A on the sbatch line
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive                    # one node per job: per-run wandb service socket
#SBATCH --array=0-59%6                 # 60 tasks, 2 games in flight
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/scratch_%A_%a.out
#SBATCH -e logs/scratch_%A_%a.err

set -u

GAMES=(gopher kung_fu_master private_eye jamesbond boxing krull \
       alien amidar assault asterix bank_heist battle_zone chopper_command \
       crazy_climber frostbite hero kangaroo qbert road_runner seaquest)

i=$SLURM_ARRAY_TASK_ID
GAME=${GAMES[$((i/3))]}
SEED=$((i%3))

# Environment built by experiments/isca_setup.sh
module load Anaconda3 2>/dev/null || true
source activate "$HOME/echelon-env" 2>/dev/null || conda activate "$HOME/echelon-env"
cd "$HOME/echelon"

export env_name="atari100k-${GAME}" run_name=hrvq_baseline
export override_config='{"precision":"bfloat16"}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Task-unique temp and wandb directories, so concurrent tasks stay isolated
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME seed=$SEED | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
python main.py --wandb --seed "$SEED" --eval_period_epoch 5 --keep_last_k 3 \
  --log_figure_period_epoch 9999 \
  --wandb_name "baseline/${GAME}/seed${SEED}"
