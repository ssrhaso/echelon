#!/bin/bash
# SLURM job array for the transfer sweep: array task N runs row N of the manifest
# CSV (0-indexed, header excluded) on one GPU.
#
# Stage the source checkpoint with experiments/isca_setup.sh before submitting,
# and keep one precision throughout a manifest.

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE   # overridden by -A on the sbatch line
#SBATCH --time=24:00:00                # ISCA hard cap on partition gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   # one GPU per task, so 2 tasks/node on the A100s
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/transfer_%A_%a.out
#SBATCH -e logs/transfer_%A_%a.err

set -u

MANIFEST="${MANIFEST:-experiments/isca_transfer_manifest.csv}"
i=$SLURM_ARRAY_TASK_ID

# Environment built by experiments/isca_setup.sh. Activated before the manifest is
# parsed, which needs the environment's python.
module load Anaconda3 2>/dev/null || true
source activate "$HOME/echelon-env" 2>/dev/null || conda activate "$HOME/echelon-env"
cd "$HOME/echelon"

# Row i of the manifest is line i+2, the header being line 1
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1
fi
ROW=$(awk -v n=$((i + 2)) 'NR==n' "$MANIFEST")
if [ -z "$ROW" ]; then
    echo "ERROR: manifest has no row for array task $i" >&2; exit 1
fi

# The manifest is RFC4180: override_config and freeze_levels are quoted because
# they contain commas, so parse with python rather than cut or IFS.
eval "$(python - "$MANIFEST" "$i" <<'PY'
import csv, sys, shlex
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
r = rows[int(sys.argv[2])]
assert int(r["idx"]) == int(sys.argv[2]), "manifest idx column does not match row order"
for k in ("game", "condition", "seed", "precision", "tag", "wandb_name", "env_name",
          "run_name", "freeze_levels", "freeze_encoder", "transfer_checkpoint",
          "override_config"):
    print("ROW_%s=%s" % (k.upper(), shlex.quote(r[k])))
# .get: manifests predating the whole-model-transfer baseline have no such column
print("ROW_TRANSFER_ALL=%s" % shlex.quote(r.get("transfer_all", "0")))
PY
)"

export env_name="$ROW_ENV_NAME"
export run_name="$ROW_RUN_NAME"

# Precision is set only here; nnet/models/model.py records it in wandb.config
export override_config="$ROW_OVERRIDE_CONFIG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

export WANDB_TAGS="$ROW_TAG"

# Task-unique temp directory: two tasks share a node, and a shared wandb service
# socket cross-contaminates concurrent runs. Prefer node-local scratch, whose root
# is not always writable, so probe candidates and fall back to $HOME/tmp.
TASK_SUFFIX="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
JOB_TMP=""
for cand in "${SLURM_TMPDIR:-}" "/local/${SLURM_JOB_ID}" "/local/${USER}" "/local" \
            "/scratch/${USER}" "$HOME/tmp"; do
    [ -n "$cand" ] || continue
    try="${cand}/${TASK_SUFFIX}"
    if mkdir -p "$try" 2>/dev/null && [ -w "$try" ]; then
        JOB_TMP="$try"; break
    fi
done
if [ -z "$JOB_TMP" ]; then
    echo "ERROR: no writable temp root found (tried SLURM_TMPDIR, /local/*, /scratch, \$HOME/tmp)" >&2
    exit 1
fi
export TMPDIR="$JOB_TMP"
export WANDB_DIR="$JOB_TMP"
export WANDB_CACHE_DIR="$JOB_TMP/wandb-cache"
export WANDB_CONFIG_DIR="$JOB_TMP/wandb-config"
export WANDB_DATA_DIR="$JOB_TMP/wandb-data"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
trap 'rm -rf "$JOB_TMP"' EXIT

echo "=== transfer task $i | ${ROW_GAME} / ${ROW_CONDITION} / seed ${ROW_SEED} ==="
echo "    host      : $(hostname)"
echo "    gpu       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "    precision : ${ROW_PRECISION}  (override_config=${ROW_OVERRIDE_CONFIG})"
echo "    wandb_name: ${ROW_WANDB_NAME}"
echo "    tmp       : ${JOB_TMP}"

ARGS=(--wandb --seed "$ROW_SEED" --eval_period_epoch 5 --keep_last_k 3
      --log_figure_period_epoch 9999
      --wandb_name "$ROW_WANDB_NAME")

if [ -n "$ROW_TRANSFER_CHECKPOINT" ]; then
    if [ ! -f "$ROW_TRANSFER_CHECKPOINT" ]; then
        echo "ERROR: transfer source not staged at $ROW_TRANSFER_CHECKPOINT; run experiments/isca_setup.sh" >&2
        exit 1
    fi
    ARGS+=(--transfer_checkpoint "$ROW_TRANSFER_CHECKPOINT")
fi
if [ "$ROW_TRANSFER_ALL" = "1" ]; then
    ARGS+=(--transfer_all)
fi
if [ -n "$ROW_FREEZE_LEVELS" ]; then
    ARGS+=(--freeze_levels "$ROW_FREEZE_LEVELS")
fi
if [ "$ROW_FREEZE_ENCODER" = "1" ]; then
    ARGS+=(--freeze_encoder)
fi

echo "    launch    : python main.py ${ARGS[*]}"
echo
python main.py "${ARGS[@]}"
