#!/bin/bash
# One-time cluster setup for the sweeps, run on a login node or in a short
# interactive session. Builds the conda environment on shared storage, installs
# dependencies and stages the source checkpoint for the array tasks to reuse.
#
#   bash experiments/isca_setup.sh
#
# The two lines marked SITE may need the cluster's own module and conda names.
set -e

# /tmp on login nodes is small; route pip temp and cache onto shared storage so
# the multi-GB torch wheels do not run out of space.
export TMPDIR="$HOME/tmp"
export PIP_CACHE_DIR="$HOME/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# SITE: the conda module name may differ (module avail anaconda)
module load Anaconda3 2>/dev/null || module load anaconda3 2>/dev/null || true

ENVDIR="$HOME/echelon-env"
REPO="$HOME/echelon"

# 1. conda env (Python 3.11; torch has no wheels for 3.14)
if [ ! -d "$ENVDIR" ]; then
    conda create -y -p "$ENVDIR" python=3.11
fi
# SITE: if `conda activate` errors, source the cluster's conda.sh first
source activate "$ENVDIR" 2>/dev/null || conda activate "$ENVDIR"

# 2. repo
if [ ! -d "$REPO" ]; then
    git clone https://github.com/ssrhaso/echelon.git "$REPO"
fi
cd "$REPO"
git pull --ff-only || true

# 3. deps (ale-py bundles the ROMs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install gymnasium ale-py opencv-python wandb tqdm av tensorboard pyyaml

# 4. patch the dm_control import so the repo loads without mujoco
python - <<'PY'
p = "nnet/envs/__init__.py"; s = open(p).read()
if "try:\n    from . import dm_control" not in s:
    open(p, "w").write(s.replace("from . import dm_control",
        "try:\n    from . import dm_control\nexcept ImportError:\n    pass")); print("patched dm_control")
else:
    print("dm_control already patched")
PY

# 5. W&B auth, written to ~/.netrc and inherited by the array tasks. Paste the key
#    at the prompt, or export WANDB_API_KEY before running this script.
wandb login

# 6. stage the source checkpoint named by the sweep definition
python - <<'PY'
import glob, os, shutil, yaml, wandb
cfg = yaml.safe_load(open("experiments/transfer_freezing.yaml"))
dst = cfg["transfer_checkpoint"]
ref = cfg["transfer_checkpoint_artifact"]
if os.path.isfile(dst):
    print("have", dst)
else:
    root = "transfer_ckpt/_dl_" + os.path.basename(dst).replace(".ckpt", "")
    src = max(glob.glob(os.path.join(wandb.Api().artifact(ref).download(root=root), "*.ckpt")),
              key=os.path.getsize)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print("staged", dst, "from", ref)
PY

# 7. logs dir for SLURM stdout/stderr
mkdir -p logs

# 8. sanity
python -c "import torch; print('torch', torch.__version__, 'CUDA-build', torch.version.cuda)"
echo "Setup complete. Submit a sweep with, e.g.:"
echo "  sbatch -A <account> experiments/isca_transfer.slurm"
echo "  sbatch -A <account> experiments/isca_scratch.slurm"
