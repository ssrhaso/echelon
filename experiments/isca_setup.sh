#!/bin/bash
# One-time ISCA setup for the codebook-stitching matrix.
# Run ONCE on a login node (lightweight: conda env + pip + downloads, NO training)
# or inside a short interactive session. Builds the env on the shared filesystem so
# every array task in isca_stitch.slurm can reuse it.
#
#   bash experiments/isca_setup.sh
#
# Adjust the two CLUSTER-SPECIFIC lines marked TODO to ISCA's module/conda names.
set -e

# /tmp on login nodes is tiny; route pip temp + cache onto lustre (huge quota)
# so the multi-GB torch/CUDA wheels don't blow up with "No space left on device".
export TMPDIR="$HOME/tmp"
export PIP_CACHE_DIR="$HOME/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# --- TODO(ISCA): module name for conda may differ (module avail anaconda) ---
module load Anaconda3 2>/dev/null || module load anaconda3 2>/dev/null || true

ENVDIR="$HOME/echelon-env"
REPO="$HOME/echelon"

# 1. conda env (Python 3.11 -- torch has no wheels for 3.14)
if [ ! -d "$ENVDIR" ]; then
    conda create -y -p "$ENVDIR" python=3.11
fi
# --- TODO(ISCA): if `conda activate` errors, source the cluster's conda.sh first ---
source activate "$ENVDIR" 2>/dev/null || conda activate "$ENVDIR"

# 2. repo
if [ ! -d "$REPO" ]; then
    git clone https://github.com/ssrhaso/echelon.git "$REPO"
fi
cd "$REPO"
git pull --ff-only || true

# 3. deps (ale-py bundles ROMs; no AutoROM needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install gymnasium ale-py opencv-python wandb tqdm av tensorboard pyyaml

# 4. patch dm_control import so the repo loads without mujoco
python - <<'PY'
p = "nnet/envs/__init__.py"; s = open(p).read()
if "try:\n    from . import dm_control" not in s:
    open(p, "w").write(s.replace("from . import dm_control",
        "try:\n    from . import dm_control\nexcept ImportError:\n    pass")); print("patched dm_control")
else:
    print("dm_control already patched")
PY

# 5. W&B auth (writes ~/.netrc; array tasks inherit it). Paste your key at the prompt,
#    or `export WANDB_API_KEY=...` before running this script.
wandb login

# 6. stage every pinned donor (foreign Pong + each game's matched codebook)
python - <<'PY'
import yaml, wandb, shutil, os, glob
cfg = yaml.safe_load(open("experiments/codebook_stitching.yaml")); api = wandb.Api()
items = [(cfg["foreign_donor"]["ckpt"], cfg["foreign_donor"]["artifact"])]
items += [(g["ckpt"], g["artifact"]) for g in cfg["games"].values()]
for dst, ref in items:
    if os.path.isfile(dst):
        print("have", dst); continue
    sub = "transfer_ckpt/_dl_" + os.path.basename(dst).replace(".ckpt", "")
    d = api.artifact(ref).download(root=sub)
    src = max(glob.glob(os.path.join(d, "*.ckpt")), key=os.path.getsize)
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(src, dst)
    print("staged", dst, "<-", ref)
PY

# 7. logs dir for SLURM stdout/stderr
mkdir -p logs

# 8. sanity
python -c "import torch; print('torch', torch.__version__, 'CUDA-build', torch.version.cuda)"
echo "ISCA setup complete. Submit a game's matrix with, e.g.:"
echo "  sbatch --export=ALL,GAME=ms_pacman experiments/isca_stitch.slurm"
echo "  sbatch --export=ALL,GAME=up_n_down experiments/isca_stitch.slurm"
