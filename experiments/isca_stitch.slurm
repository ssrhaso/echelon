#!/bin/bash
# ISCA (Exeter HPC) job-array submission for the codebook-stitching matrix.
# 7 cells x 3 seeds = 21 array tasks; each task is ONE (cell, seed) on its own GPU.
#
# GAME selects the eval game + its matched donor codebook. Defaults to demon_attack;
# override per game (donor must be staged at transfer_ckpt/<GAME>_donor.ckpt):
#   sbatch --export=ALL,GAME=demon_attack experiments/isca_stitch.slurm
#   sbatch --export=ALL,GAME=ms_pacman    experiments/isca_stitch.slurm
#   sbatch --export=ALL,GAME=up_n_down    experiments/isca_stitch.slurm
#
# One-time setup first: bash experiments/isca_setup.sh   (stages pong + all game donors)
# Watch:                squeue -u $USER   |   tail -f logs/stitch_*_*.out

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE         # TODO: sacctmgr -n show assoc user=$USER format=account%40
#SBATCH --time=24:00:00                      # one run fits a full GPU in <24h
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive          # whole node per job: the only fully reliable cure for the
                             # wandb service-socket cross-contamination on shared nodes.
#SBATCH --array=0-20%8                        # 21 tasks (7 cells x 3 seeds), max 8 concurrent
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/stitch_%A_%a.out
#SBATCH -e logs/stitch_%A_%a.err

set -u

# --- eval game + donors (GAME overridable via --export=ALL,GAME=...) ---
GAME="${GAME:-demon_attack}"
PONG_CKPT="transfer_ckpt/pong_seed5_best.ckpt"          # foreign donor (same for every game)
MATCH_CKPT="transfer_ckpt/${GAME}_donor.ckpt"           # matched donor = this game's own codebook
SRC="pong=${PONG_CKPT},match=${MATCH_CKPT}"

# --- matrix: cells as L0.L1.L2 donors. match = this game's codebook, pong = foreign.
# Priority order: single-level injections + ceiling first, then the pair cells.
CELLS=(DPP                    PDP                    PPD                    DDD                    DDP                    PDD                    DPD)
MAPS=("0:match,1:pong,2:pong" "0:pong,1:match,2:pong" "0:pong,1:pong,2:match" "0:match,1:match,2:match" "0:match,1:match,2:pong" "0:pong,1:match,2:match" "0:match,1:pong,2:match")

i=$SLURM_ARRAY_TASK_ID
CELL=${CELLS[$((i/3))]}
MAP=${MAPS[$((i/3))]}
SEED=$((i%3))

# --- env (match isca_setup.sh) ---
module load Anaconda3 2>/dev/null || true
source activate "$HOME/echelon-env" 2>/dev/null || conda activate "$HOME/echelon-env"
cd "$HOME/echelon"

# fail fast if the matched donor for this game isn't staged
if [ ! -f "$MATCH_CKPT" ]; then
    echo "ERROR: matched donor for GAME=$GAME not found at $MATCH_CKPT - run isca_setup.sh first." >&2
    exit 1
fi

export env_name="atari100k-${GAME}" run_name=hrvq_stitch
export override_config='{"precision":"bfloat16"}'   # bf16 across all stitch cells: A100-friendly, matches freeze-L012 baselines
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-task isolation: two array tasks can land on one node (2 GPUs/node); a shared
# TMPDIR/cwd makes concurrent wandb runs cross-contaminate. Unique temp + wandb dir
# per task keeps each run's logging fully separate (belt-and-suspenders with --exclusive).
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1                            # stream training logs live to the SLURM .out

echo "=== array task $i | game=$GAME | cell=$CELL seed=$SEED | map=$MAP | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
python main.py --wandb --seed "$SEED" --eval_period_epoch 5 --keep_last_k 3 \
  --log_figure_period_epoch 9999 \
  --codebook_map "$MAP" --codebook_sources "$SRC" \
  --wandb_name "transfer/stitch/${GAME}/${CELL}/seed${SEED}"
