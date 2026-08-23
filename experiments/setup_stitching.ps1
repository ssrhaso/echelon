# ECHELON Codebook-Stitching Experiment Setup (Pong x Demon Attack -> eval Demon Attack)
#
# NOTE (superseded): the multi-game stitching experiment runs on ISCA via SLURM --
# experiments/isca_setup.sh (stages all donors) + experiments/isca_stitch.slurm
# (game-parameterized: `sbatch --export=ALL,GAME=<game> ...`). This Windows/Vast.ai
# launcher targets the original single-game DA matrix and predates the multi-game
# manifest in codebook_stitching.yaml; prefer the ISCA flow.
#
# Clone the repo on a fresh GPU box and run the stitching matrix with one command.
#
# Usage (run from anywhere - the script chdirs to the repo root):
#   powershell -ExecutionPolicy Bypass -File experiments\setup_stitching.ps1            # setup + print every launch command
#   powershell -ExecutionPolicy Bypass -File experiments\setup_stitching.ps1 -Run       # setup + run the WHOLE matrix
#   powershell -ExecutionPolicy Bypass -File experiments\setup_stitching.ps1 -Only DPP  # setup + run one cell (all its seeds)
#   powershell -ExecutionPolicy Bypass -File experiments\setup_stitching.ps1 -Only DPP -Seed 1   # one cell, one seed
#   powershell -ExecutionPolicy Bypass -File experiments\setup_stitching.ps1 -SkipSetup -Run     # skip env setup, just run
#
# Matrix + donor definitions live in experiments\codebook_stitching.yaml.

param(
    [switch]$Run,
    [string]$Only = "",
    [int]$Seed = -1,
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"

Write-Host "ECHELON codebook-stitching setup (Pong x Demon Attack -> DA)" -ForegroundColor Cyan

# Resolve repo root (parent of this script's folder) and chdir there so all
# relative paths (nnet/, main.py, transfer_ckpt/) resolve no matter where invoked.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot" -ForegroundColor DarkGray

$ConfigFile = Join-Path $PSScriptRoot "codebook_stitching.yaml"
$ConfigFileFwd = $ConfigFile -replace '\\', '/'

if (-not $SkipSetup) {
    Write-Host "[1/7] Installing torch + torchvision (CUDA 12.8)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

    Write-Host "[2/7] Installing Atari + training deps..."
    pip install gymnasium ale-py opencv-python wandb tqdm av tensorboard pyyaml autorom

    Write-Host "[3/7] Downloading Atari ROMs..."
    try { AutoROM --accept-license } catch {
        Write-Host "  AutoROM failed, retrying via python -m AutoROM..." -ForegroundColor Yellow
        try { python -m AutoROM --accept-license } catch {
            Write-Host "  AutoROM still failing - install ROMs manually if training errors." -ForegroundColor Red
        }
    }

    Write-Host "[4/7] Logging into Weights & Biases..."
    python -m wandb login

    Write-Host "[5/7] Patching nnet/envs/__init__.py to tolerate missing dm_control..."
    $envInit = "nnet/envs/__init__.py"
    $content = Get-Content $envInit -Raw
    if ($content -notmatch "try:\s*\r?\n\s*from \. import dm_control") {
        $patched = $content -replace "from \. import dm_control", "try:`n    from . import dm_control`nexcept ImportError:`n    pass"
        Set-Content -Path $envInit -Value $patched -NoNewline
        Write-Host "  patched."
    } else { Write-Host "  already patched, skipping." }

    Write-Host "[6/7] Disabling sleep/hibernate on AC..."
    try { powercfg /change standby-timeout-ac 0; powercfg /change hibernate-timeout-ac 0 } catch {
        Write-Host "  powercfg not available (non-Windows host?), skipping." -ForegroundColor Yellow
    }

    Write-Host "[7/7] Downloading + staging BOTH donor checkpoints from W&B..."
    python -c @"
import yaml, wandb, shutil, os, glob
cfg = yaml.safe_load(open('$ConfigFileFwd'))
api = wandb.Api()
for name, art_ref in cfg['donor_artifacts'].items():
    dst = cfg['donors'][name]
    if os.path.isfile(dst):
        print(f'  {name}: already staged at {dst}'); continue
    d = api.artifact(art_ref).download(root=os.path.join('transfer_ckpt', f'_dl_{name}'))
    ckpts = glob.glob(os.path.join(d, '*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f'No .ckpt in artifact {art_ref}')
    src = max(ckpts, key=os.path.getsize)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(f'  {name}: staged -> {dst}')
"@
}

# ---- Sanity: both donors present ----
$donorCheck = python -c @"
import yaml, os
cfg = yaml.safe_load(open('$ConfigFileFwd'))
missing = [f'{n}={p}' for n, p in cfg['donors'].items() if not os.path.isfile(p)]
print('MISSING:' + ','.join(missing) if missing else 'OK')
"@
if ($donorCheck -ne "OK") {
    Write-Host "Donor checkpoints missing ($donorCheck). Re-run without -SkipSetup." -ForegroundColor Red
    exit 1
}

# ---- Parse the matrix YAML into a flat job list ----
# Line: name|seed|map|env_name|run_name|eval_p|keep_k|precision|sources|description
$parser = @"
import yaml
cfg = yaml.safe_load(open('$ConfigFileFwd'))
env_name = cfg['env_name']; run_name = cfg['run_name']
base = cfg.get('base_args', {})
eval_p = base.get('eval_period_epoch', 5); keep_k = base.get('keep_last_k', 3)
prec = base.get('precision', 'float32')
sources = ','.join(f'{n}={p}' for n, p in cfg['donors'].items())
for exp in cfg['experiments']:
    for seed in exp['seeds']:
        print('|'.join(str(x) for x in [exp['name'], seed, exp['map'], env_name,
              run_name, eval_p, keep_k, prec, sources, exp.get('description','')]))
"@
$jobLines = python -c $parser
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to parse $ConfigFile" -ForegroundColor Red; exit 1 }

$jobs = @()
foreach ($line in $jobLines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $p = $line.Split('|')
    $jobs += [pscustomobject]@{
        Name = $p[0]; Seed = [int]$p[1]; Map = $p[2]; EnvName = $p[3]; RunName = $p[4]
        EvalPeriod = [int]$p[5]; KeepLastK = [int]$p[6]; Precision = $p[7]
        Sources = $p[8]; Description = $p[9]
    }
}

# ---- Precision env (DA = bf16/3070 profile, keep the pair internally consistent) ----
$firstPrec = ($jobs | Select-Object -First 1).Precision
if ($firstPrec -ne "float32") {
    $env:override_config = '{"precision":"' + $firstPrec + '"}'
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}
$env:env_name = ($jobs | Select-Object -First 1).EnvName
$env:run_name = ($jobs | Select-Object -First 1).RunName

function Get-LaunchArgs($job) {
    $launch = @(
        "main.py", "--wandb",
        "--seed", $job.Seed,
        "--eval_period_epoch", $job.EvalPeriod,
        "--keep_last_k", $job.KeepLastK,
        "--log_figure_period_epoch", 9999,
        "--codebook_map", $job.Map,
        "--codebook_sources", $job.Sources,
        "--wandb_name", "transfer/stitch/$($job.EnvName.Split('-')[-1])/$($job.Name)/seed$($job.Seed)"
    )
    return ,$launch
}

# ---- Filter ----
if ($Only -ne "") {
    $jobs = $jobs | Where-Object { $_.Name -eq $Only }
    if ($jobs.Count -eq 0) { Write-Host "No cell named '$Only' in $ConfigFile" -ForegroundColor Red; exit 1 }
}
if ($Seed -ge 0) { $jobs = $jobs | Where-Object { $_.Seed -eq $Seed } }

Write-Host ""
Write-Host "Setup complete. Matrix: $($jobs.Count) run(s)." -ForegroundColor Green
Write-Host "  env_name = $env:env_name | run_name = $env:run_name | precision = $firstPrec" -ForegroundColor Yellow
if ($env:override_config) { Write-Host "  override_config = $env:override_config" -ForegroundColor Yellow }
Write-Host ""

if ($Run -or $Only -ne "") {
    Write-Host "Running $($jobs.Count) stitching run(s)..." -ForegroundColor Cyan
    foreach ($job in $jobs) {
        $launchArgs = Get-LaunchArgs $job
        Write-Host ""
        Write-Host ">>> [$($job.Name) seed=$($job.Seed)] $($job.Description)" -ForegroundColor Cyan
        Write-Host "    python $($launchArgs -join ' ')" -ForegroundColor DarkGray
        & python @launchArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: $($job.Name) seed=$($job.Seed) (exit $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "Continuing to next run..." -ForegroundColor Yellow
        }
    }
    Write-Host "`nAll requested stitching runs finished." -ForegroundColor Green
} else {
    Write-Host "Set the session env vars once, then copy-paste any launch line:" -ForegroundColor Cyan
    Write-Host "  `$env:env_name = `"$env:env_name`""
    Write-Host "  `$env:run_name = `"$env:run_name`""
    if ($env:override_config) {
        Write-Host "  `$env:override_config = '$env:override_config'"
        Write-Host "  `$env:PYTORCH_CUDA_ALLOC_CONF = `"$env:PYTORCH_CUDA_ALLOC_CONF`""
    }
    Write-Host ""
    foreach ($job in $jobs) {
        Write-Host "# [$($job.Name) seed=$($job.Seed)] $($job.Description)"
        Write-Host "python $((Get-LaunchArgs $job) -join ' ')"
        Write-Host ""
    }
    Write-Host "Run the whole matrix with -Run, or one cell with -Only <NAME> [-Seed N]." -ForegroundColor Yellow
}
