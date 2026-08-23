#!/bin/bash
# ISCA (Exeter HPC) job array for the isca-aug26 sprint: the four target-game ladders
# (Ms Pac-Man, Asterix, Up'n'Down, Demon Attack), 7 cells x 5 seeds each.
#
# Each array task runs exactly ONE row of the manifest CSV on ONE GPU, so a 2-GPU A100
# node hosts two independent runs. Row N of the manifest (0-indexed, header excluded)
# is array task N -- see scripts/gen_isca_manifest.py for the ordering guarantee.
#
# Do not submit this by hand: scripts/isca_launch.py validates the manifest first
# (single precision, staged donor, protected source untouched) and prints/submits the
# sbatch lines.
#   python scripts/isca_launch.py --manifest experiments/isca_aug26_manifest.csv          # dry run
#   python scripts/isca_launch.py --manifest experiments/isca_aug26_manifest.csv --submit
#
# Watch: squeue --me ; tail -f logs/aug26_*.out

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE   # overridden by -A on the sbatch line
#SBATCH --time=24:00:00                # ISCA hard cap on partition gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   # ONE GPU per task => 2 tasks/node on the A100s
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/aug26_%A_%a.out
#SBATCH -e logs/aug26_%A_%a.err

set -u

MANIFEST="${MANIFEST:-experiments/isca_aug26_manifest.csv}"
i=$SLURM_ARRAY_TASK_ID

# --- env (matches experiments/isca_setup.sh) ---
# Done BEFORE the manifest is parsed: parsing needs the env's python, and the login
# node's bare `python` is not guaranteed to exist.
module load Anaconda3 2>/dev/null || true
source activate "$HOME/echelon-env" 2>/dev/null || conda activate "$HOME/echelon-env"
cd "$HOME/echelon"   # /lustre/home/$USER/echelon -- checkpoints + logs persist here

# --- read row i of the manifest (skip header; row i is line i+2) ---
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1
fi
ROW=$(awk -v n=$((i + 2)) 'NR==n' "$MANIFEST")
if [ -z "$ROW" ]; then
    echo "ERROR: manifest has no row for array task $i" >&2; exit 1
fi

# The manifest is RFC4180: override_config and freeze_levels are quoted because they
# contain commas. Parse with python rather than cut/IFS, which would split inside them.
eval "$(python - "$MANIFEST" "$i" <<'PY'
import csv, sys, shlex
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
r = rows[int(sys.argv[2])]
assert int(r["idx"]) == int(sys.argv[2]), "manifest idx column does not match row order"
for k in ("game", "condition", "seed", "precision", "tag", "wandb_name", "env_name",
          "run_name", "freeze_levels", "freeze_encoder", "transfer_checkpoint",
          "override_config"):
    print("ROW_%s=%s" % (k.upper(), shlex.quote(r[k])))
# Added with the weight-transfer baseline. Read with .get so this script still
# parses the original aug26 manifest, which has no such column.
print("ROW_TRANSFER_ALL=%s" % shlex.quote(r.get("transfer_all", "0")))
PY
)"

export env_name="$ROW_ENV_NAME"
export run_name="$ROW_RUN_NAME"

# Precision. This env var is the ONLY thing that sets precision in this codebase;
# it is now also written into wandb.config at startup by nnet/models/model.py, so no
# future audit has to reconstruct it from stdout.
export override_config="$ROW_OVERRIDE_CONFIG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

export WANDB_TAGS="$ROW_TAG"

# Per-task isolation. Two tasks share a node here (no --exclusive, unlike
# isca_stitch.slurm), so every path wandb might contend on gets a task-unique
# directory -- the service socket lives under TMPDIR and a shared one is what
# cross-contaminated concurrent runs previously.
#
# Node-local scratch is preferred (fast, wiped at job end), but the root of /local is
# NOT user-writable on the ISCA gpu nodes -- creating /local/$USER fails with
# EACCES and killed the first pilot in 90s. So probe candidates in order and take the
# first that actually works, falling back to $HOME/tmp, which is what the working
# isca_stitch.slurm / isca_scratch.slurm have always used.
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
echo "    tmp root  : ${JOB_TMP}"
export TMPDIR="$JOB_TMP"
export WANDB_DIR="$JOB_TMP"
export WANDB_CACHE_DIR="$JOB_TMP/wandb-cache"
export WANDB_CONFIG_DIR="$JOB_TMP/wandb-config"
export WANDB_DATA_DIR="$JOB_TMP/wandb-data"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
trap 'rm -rf "$JOB_TMP"' EXIT

echo "=== isca-aug26 task $i | ${ROW_GAME} / ${ROW_CONDITION} / seed ${ROW_SEED} ==="
echo "    host      : $(hostname)"
echo "    gpu       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "    precision : ${ROW_PRECISION}  (override_config=${ROW_OVERRIDE_CONFIG})"
echo "    wandb_name: ${ROW_WANDB_NAME}"
echo "    tmp       : ${JOB_TMP}"

# --- assemble launch args from the row ---
ARGS=(--wandb --seed "$ROW_SEED" --eval_period_epoch 5 --keep_last_k 3
      --log_figure_period_epoch 9999
      --wandb_name "$ROW_WANDB_NAME")

if [ -n "$ROW_TRANSFER_CHECKPOINT" ]; then
    if [ ! -f "$ROW_TRANSFER_CHECKPOINT" ]; then
        echo "ERROR: transfer source not staged at $ROW_TRANSFER_CHECKPOINT -- run experiments/isca_setup.sh" >&2
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
