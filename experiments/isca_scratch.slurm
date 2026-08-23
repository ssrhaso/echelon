#!/bin/bash
# ISCA (Exeter HPC) job-array: HRVQ scratch baselines across the remaining Atari100k games.
# 20 games x 3 seeds = 60 array tasks, GAME-MAJOR: a game's 3 seeds are contiguous indices
# (0,1,2 = game0; 3,4,5 = game1; ...), so games complete in order.
#   game = GAMES[i/3],  seed = i%3.
#
# Launch (pass the account from your working stitch script so it matches):
#   cd ~/echelon
#   ACCT=$(grep -m1 '#SBATCH -A' experiments/isca_stitch.slurm | awk '{print $3}')
#   sbatch -A "$ACCT" experiments/isca_scratch.slurm
# Watch: squeue --me ; tail -f logs/scratch_*_*.out

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE          # overridden by -A on the sbatch line
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive          # whole node per job: cures wandb service-socket cross-contamination
#SBATCH --array=0-59%6       # 60 tasks (20 games x 3 seeds); %6 = 2 games in flight, in order
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/scratch_%A_%a.out
#SBATCH -e logs/scratch_%A_%a.err

set -u

# --- games (game-major): high-signal first (blowout + transfer candidates), then the rest ---
GAMES=(gopher kung_fu_master private_eye jamesbond boxing krull \
       alien amidar assault asterix bank_heist battle_zone chopper_command \
       crazy_climber frostbite hero kangaroo qbert road_runner seaquest)

i=$SLURM_ARRAY_TASK_ID
GAME=${GAMES[$((i/3))]}
SEED=$((i%3))

# --- env (match isca_stitch.slurm) ---
module load Anaconda3 2>/dev/null || true
source activate "$HOME/echelon-env" 2>/dev/null || conda activate "$HOME/echelon-env"
cd "$HOME/echelon"

export env_name="atari100k-${GAME}" run_name=hrvq_baseline
export override_config='{"precision":"bfloat16"}'   # bf16: A100-friendly, matches existing baselines
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-task isolation (unique temp + wandb dir) so concurrent tasks never cross-contaminate wandb.
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME seed=$SEED | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
python main.py --wandb --seed "$SEED" --eval_period_epoch 5 --keep_last_k 3 \
  --log_figure_period_epoch 9999 \
  --wandb_name "baseline/${GAME}/seed${SEED}"
